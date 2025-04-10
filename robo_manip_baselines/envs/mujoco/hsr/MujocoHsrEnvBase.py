from os import path

import mujoco
import numpy as np
from gymnasium.spaces import Box, Dict

from robo_manip_baselines.common import ArmConfig
from robo_manip_baselines.teleop import SpacemouseInputDevice

from ..MujocoEnvBase import MujocoEnvBase


class MujocoHsrEnvBase(MujocoEnvBase):
    default_camera_config = {
        "azimuth": -135.0,
        "elevation": -45.0,
        "distance": 1.8,
        "lookat": [0.5, 0.0, 0.2],
    }
    observation_space = Dict(
        {
            "joint_pos": Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float64),
            "joint_vel": Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float64),
            "wrench": Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
        }
    )

    def setup_robot(self, init_qpos):
        self.init_qpos[: len(init_qpos)] = init_qpos
        self.init_qvel[:] = 0.0

        mujoco.mj_kinematics(self.model, self.data)

        self.body_config_list = [
            ArmConfig(
                arm_urdf_path=path.join(
                    path.dirname(__file__), "../../assets/common/robots/hsr/hsr.urdf"
                ),
                arm_root_pose=self.get_body_pose("base_link"),
                ik_eef_joint_id=5,
                arm_joint_idxes=np.arange(5),
                gripper_joint_idxes=np.array([5]),
                gripper_joint_idxes_in_gripper_joint_pos=np.array([0]),
                eef_idx=0,
                init_arm_joint_pos=self.init_qpos[0:5],
                init_gripper_joint_pos=np.zeros(1),
            )
        ]

    def setup_input_device(self, input_device_name, motion_manager, overwrite_kwargs):
        if input_device_name == "spacemouse":
            InputDeviceClass = SpacemouseInputDevice
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid input device key: {input_device_name}"
            )

        default_kwargs = self.get_input_device_kwargs(input_device_name)

        return [
            InputDeviceClass(
                motion_manager.body_manager_list[0],
                **{**default_kwargs, **overwrite_kwargs},
            )
        ]

    def get_input_device_kwargs(self, input_device_name):
        return {}

    def _get_obs(self):
        arm_joint_name_list = [
            "torso_lift_joint",
            "arm_flex_joint",
            "arm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
        ]
        gripper_joint_name = "hand_motor_joint"

        arm_joint_pos = np.array(
            [self.data.joint(joint_name).qpos[0] for joint_name in arm_joint_name_list]
        )
        arm_joint_vel = np.array(
            [self.data.joint(joint_name).qvel[0] for joint_name in arm_joint_name_list]
        )
        gripper_joint_pos = np.array([self.data.joint(gripper_joint_name).qpos[0]])
        gripper_joint_vel = np.zeros(1)
        force = self.data.sensor("force_sensor").data.flat.copy()
        torque = self.data.sensor("torque_sensor").data.flat.copy()

        return {
            "joint_pos": np.concatenate(
                (arm_joint_pos, gripper_joint_pos), dtype=np.float64
            ),
            "joint_vel": np.concatenate(
                (arm_joint_vel, gripper_joint_vel), dtype=np.float64
            ),
            "wrench": np.concatenate((force, torque), dtype=np.float64),
        }
