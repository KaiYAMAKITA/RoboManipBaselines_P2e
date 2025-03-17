import gymnasium as gym

from robo_manip_baselines.common import GraspPhaseBase


class GraspPhase(GraspPhaseBase):
    def set_target(self):
        self.set_target_open()


class OperationMujocoUR5eToolbox(object):
    def setup_env(self):
        self.env = gym.make(
            "robo_manip_baselines/MujocoUR5eToolboxEnv-v0", render_mode="human"
        )

    def get_pre_motion_phases(self):
        return [GraspPhase(self)]
