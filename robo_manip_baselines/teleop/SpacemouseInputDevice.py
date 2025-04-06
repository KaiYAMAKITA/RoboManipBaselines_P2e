import numpy as np
import pinocchio as pin

from .InputDeviceBase import InputDeviceBase


class SpacemouseInputDevice(InputDeviceBase):
    """Spacemouse for teleoperation input device."""

    def __init__(self, arm_manager, pos_scale=1e-2, rpy_scale=5e-3, gripper_scale=5.0):
        super().__init__()

        self.arm_manager = arm_manager
        self.pos_scale = pos_scale
        self.rpy_scale = rpy_scale
        self.gripper_scale = gripper_scale

    def connect(self):
        if self.connected:
            return

        self.connected = True

        import pyspacemouse

        pyspacemouse.open()

    def read(self):
        if not self.connected:
            raise RuntimeError(f"[{self.__class__.__name__}] Device is not connected.")

        # Empirically, you can call read repeatedly to get the latest device state
        import pyspacemouse

        for i in range(10):
            self.state = pyspacemouse.read()

    def set_command_data(self):
        # Set arm command
        delta_pos = self.pos_scale * np.array(
            [
                -1.0 * self.state.y,
                self.state.x,
                self.state.z,
            ]
        )
        delta_rpy = self.rpy_scale * np.array(
            [
                -1.0 * self.state.roll,
                -1.0 * self.state.pitch,
                -2.0 * self.state.yaw,
            ]
        )

        target_se3 = self.arm_manager.target_se3.copy()
        target_se3.translation += delta_pos
        target_se3.rotation = pin.rpy.rpyToMatrix(*delta_rpy) @ target_se3.rotation

        self.arm_manager.set_command_eef_pose(target_se3)

        # Set gripper command
        gripper_joint_pos = self.arm_manager.get_command_gripper_joint_pos().copy()

        if self.state.buttons[0] > 0 and self.state.buttons[-1] <= 0:
            gripper_joint_pos += self.gripper_scale
        elif self.state.buttons[-1] > 0 and self.state.buttons[0] <= 0:
            gripper_joint_pos -= self.gripper_scale

        self.arm_manager.set_command_gripper_joint_pos(gripper_joint_pos)
