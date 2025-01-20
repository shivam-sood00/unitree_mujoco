import random
import time
import sys

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml

def quat_to_rot(quat):
    w, x, y, z = quat

    R = np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x ** 2 + y ** 2)]
    ])

    return R

with open(f"go2.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    policy_path = config["policy_path"]
    xml_path = config["xml_path"]

    control_type = config["control_type"]

    simulation_duration = config["simulation_duration"]
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]

    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)

    default_angles = np.array(config["default_angles"], dtype=np.float32)

    lin_vel_scale = config["lin_vel_scale"]
    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale = config["action_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

    num_actions = config["num_actions"]
    num_obs = config["num_obs"]
    
    cmd = np.array(config["cmd_init"], dtype=np.float32)

    torque_limit = config["torque_limit"]

# policy to unitree mapping
# ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FR_hip_joint',
# 'FR_thigh_joint', 'FR_calf_joint', 'RL_hip_joint', 'RL_thigh_joint',
# 'RL_calf_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']
mapping = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
mapping_contact = [1, 0, 3, 2]

#map the default angles to gym angles:
default_angles_gym = np.array([default_angles[mapping[i]] for i in range(12)], dtype=np.float32)

class RLPolicy:
    # Define the RL policy variables
    high_state = None
    low_state = None
    policy = None
    lin_vel = torch.zeros(3, dtype=torch.float32)
    ang_vel = torch.zeros(3, dtype=torch.float32)
    projected_gravity = torch.zeros(3, dtype=torch.float32)
    dof_pos = torch.zeros(12, dtype=torch.float32)
    dof_vel = torch.zeros(12, dtype=torch.float32)
    torques = torch.zeros(12, dtype=torch.float32)
    command = [0.2, 0.0, 0.0]
    step_counter = 0
    prev_action = torch.zeros(12, dtype=torch.float32)
    action_zero = torch.zeros(12, dtype=torch.float32)

    dof_vel_limits = torch.tensor([30, 30, 20, 30, 30, 20, 30, 30, 20, 30, 30, 20], dtype=torch.float32)

    def load_policy(self, model_path):
        self.policy = torch.jit.load(model_path)
        self.policy.eval()

    def _compute_torques(self, actions, dt=simulation_dt):
        # if self.step_counter % control_decimation == 0:        
        if control_type == "torque":
            self.torques = actions * action_scale

        elif control_type == "position":
            # target_pos = torch.zeros(12, dtype=torch.float32)
            # target_pos = actions * action_scale 
            # # print("Target pos:", target_pos)

            # target_vel = torch.zeros(12, dtype=torch.float32)
            # self.torques = self.pd_control(target_pos, self.dof_pos, kps, target_vel, self.dof_vel, kds)
            # print("Pos actions", actions * action_scale)
            self.torques =  actions* action_scale
            # import pdb; pdb.set_trace()
        return self.torques

    def pd_control(self, target_q, q, kp, target_dq, dq, kd):
        for i in range(12):
            self.torques[i] = kp[i] * (target_q[i] - q[i]) + kd[i] * (target_dq[i] - dq[i])
        return self.torques
    
    def get_action(self):
        obs = self.get_obs()
        if obs is None:
            return torch.zeros(12, dtype=torch.float32)
        with torch.no_grad():
            action = self.policy(obs)
        return action, self._compute_torques(action)
        # return self._compute_torques(self.action_zero)

    def get_obs(self):
        if self.low_state is None:
            return None
        # self.step_counter += 1
        self.ang_vel = torch.tensor([self.low_state.imu_state.gyroscope[0], self.low_state.imu_state.gyroscope[1],
                                     self.low_state.imu_state.gyroscope[2]], dtype=torch.float32)

        quat = self.low_state.imu_state.quaternion
        self.projected_gravity = torch.tensor(quat_to_rot(quat) @ np.array([0, 0, -1]), dtype=torch.float32)

        dof_pos = torch.tensor(list(map(lambda x: x.q, self.low_state.motor_state))[:12], dtype=torch.float32)
        dof_vel = torch.tensor(list(map(lambda x: x.dq, self.low_state.motor_state))[:12], dtype=torch.float32)
        for i in range(12):
            self.dof_pos[mapping[i]] = dof_pos[i] - default_angles[mapping[i]]
            self.dof_vel[mapping[i]] = dof_vel[i]

        commands = torch.tensor([self.command[0] * lin_vel_scale, self.command[1] * lin_vel_scale,
                                 self.command[2] * ang_vel_scale], dtype=torch.float32)
        # commands = torch.tensor([3., 0., 0.], dtype=torch.float32)

        # print(self.high_state.velocity)
        lin_vel = np.array(self.high_state.velocity, dtype=np.float32).T @ quat_to_rot(quat)
        self.lin_vel = torch.tensor(lin_vel, dtype=torch.float32)

        if num_obs == 42:
            return torch.cat((self.projected_gravity,
                              commands,
                              (self.dof_pos) * dof_pos_scale,
                              self.dof_vel * dof_vel_scale,
                              self.prev_action,
                              ), dim=-1)
        if num_obs == 48:
            return torch.cat((self.lin_vel * lin_vel_scale,
                              self.ang_vel * ang_vel_scale,
                              self.projected_gravity,
                              commands,
                              (self.dof_pos) * dof_pos_scale,
                              self.dof_vel * dof_vel_scale,
                              self.prev_action,
                              ), dim=-1)


    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg

    def HighStateHandler(self, msg: SportModeState_):
        self.high_state = msg


rl_policy = RLPolicy()

if __name__ == "__main__":

    # Initialization
    # ChannelFactoryInitialize(1, "lo")
    if len(sys.argv) < 2:
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(0, sys.argv[1])

    hight_state_suber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    low_state_suber = ChannelSubscriber("rt/lowstate", LowState_)

    hight_state_suber.Init(rl_policy.HighStateHandler, 10)
    low_state_suber.Init(rl_policy.LowStateHandler, 10)

    low_cmd_puber = ChannelPublisher("rt/lowcmd", LowCmd_)
    low_cmd_puber.Init()
    crc = CRC()

    cmd = unitree_go_msg_dds__LowCmd_()
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0

    rl_policy.load_policy(policy_path)


    stand_first_counter = 0
    while stand_first_counter < 10000:
        for i in range(12):
            cmd.motor_cmd[i].q = default_angles[i]
            cmd.motor_cmd[i].kp = kps[i]
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kd = kds[i]
            cmd.motor_cmd[i].tau = 0.0

        stand_first_counter += 1
        cmd.crc = crc.Crc(cmd)

        # Publish message
        low_cmd_puber.Write(cmd)
    step_start = time.perf_counter()

    # Main loop
    start = time.time()
    print("Starting the Policy")
    while time.time() - start < simulation_duration:
        start_time = time.time()

        # if RLPolicy.step_counter % control_decimation == 0:

        raw_action, calculated_action = rl_policy.get_action()
        for j in range(control_decimation):
            if control_type == 'position':
                for i, a in enumerate(mapping):
                    cmd.motor_cmd[i].q = calculated_action[a] + default_angles[i]
                    cmd.motor_cmd[i].kp = kps[i]
                    cmd.motor_cmd[i].dq = 0.0
                    cmd.motor_cmd[i].kd = kds[i]
                    cmd.motor_cmd[i].tau = 0.0
            else:
                for i, a in enumerate(mapping):
                    cmd.motor_cmd[i].q = 0.
                    cmd.motor_cmd[i].kp = 0.0
                    cmd.motor_cmd[i].dq = 0.0
                    cmd.motor_cmd[i].kd = 0.
                    cmd.motor_cmd[i].tau = calculated_action[a]
        RLPolicy.prev_action = raw_action
        RLPolicy.step_counter += 1
        
        cmd.crc = crc.Crc(cmd)

        # Publish message
        low_cmd_puber.Write(cmd)

        # Sleep
        end_time = time.time()
        time.sleep(max(0., simulation_dt - (end_time - start_time)))
