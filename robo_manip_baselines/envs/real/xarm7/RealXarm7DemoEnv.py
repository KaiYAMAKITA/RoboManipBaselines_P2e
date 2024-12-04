import numpy as np

from .RealXarm7EnvBase import RealXarm7EnvBase


class RealXarm7DemoEnv(RealXarm7EnvBase):
    def __init__(
        self,
        robot_ip,
        camera_ids,
        **kwargs,
    ):
        RealXarm7EnvBase.__init__(
            self,
            robot_ip,
            camera_ids,
            init_qpos=np.deg2rad([0.0, -30.0, 0.0, 45.0, 0.0, 75.0, 0.0, 800.0]),
            **kwargs,
        )

    def modify_world(self, world_idx=None, cumulative_idx=None):
        """Modify simulation world depending on world index."""
        # TODO: Automatically set world index according to task variations
        if world_idx is None:
            world_idx = 0
            # world_idx = cumulative_idx % 2
        return world_idx
