import cv2
import matplotlib.pylab as plt
import numpy as np
import torch
import os 
from robo_manip_baselines.common import RolloutBase, denormalize_data, normalize_data
import sys
from .P2ePolicy import P2ePolicy
import importlib
from torch.utils.tensorboard import SummaryWriter
from matplotlib.backends.backend_agg import FigureCanvasAgg

import yaml
import types
from attrdict import AttrDict

from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "Plan2Explore",
    "../third_party/SimpleDreamer/dreamer/algorithms/plan2explore.py"
)
plan2explore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan2explore)
Plan2Explore = plan2explore.Plan2Explore

def load_config(config_path: str) -> AttrDict:
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return AttrDict(config)

class RolloutP2e(RolloutBase): 

    def setup_policy(self):
        
        #self.args.
        # Print policy information
        self.print_policy_info()
        print(
            f"  - obs steps: {self.model_meta_info['data']['n_obs_steps']}, action steps: {self.model_meta_info['data']['n_action_steps']}"
        )

        #define P2E

        self.config = load_config("../third_party/SimpleDreamer/dreamer/configs/p2e-dmc-walker-walk.yml")
        self.p2e = Plan2Explore(
            observation_shape=(3, 480, 640),
            discrete_action_bool=False,
            action_size=7,
            writer=SummaryWriter(log_dir="/tmp"),
            device="cuda",
            config=self.config,
        )

        self.policy = P2ePolicy(p2e=self.p2e)

        
        self.args.world_idx_list = [0 for i in range(2)]
        self.args.save_rollout = True
        self.args.auto_exit = True
        self.args.max_duration = 10.0

        
        #load checkpoint
        self.load_ckpt()

    def setup_plot(self):
        fig_ax = plt.subplots(
            2,
            len(self.camera_names),
            figsize=(13.5, 6.0),
            dpi=60,
            squeeze=False,
            constrained_layout=True,
        )
        super().setup_plot(fig_ax)

    def reset_variables(self):
        super().reset_variables()

        self.state_buf = None
        self.images_buf = None
        self.policy_action_buf = None

    def get_state(self):
        # Get latest value
        if len(self.state_keys) == 0:
            state = np.zeros(0, dtype=np.float32)
        else:
            state = np.concatenate(
                [
                    self.motion_manager.get_data(state_key, self.obs)
                    for state_key in self.state_keys
                ]
            )

        state = normalize_data(state, self.model_meta_info["state"])
        state = torch.tensor(state, dtype=torch.float32)

        # Store and return
        if self.state_buf is None:
            self.state_buf = [
                state for _ in range(self.model_meta_info["data"]["n_obs_steps"])
            ]
        else:
            self.state_buf.pop(0)
            self.state_buf.append(state)

        state = torch.stack(self.state_buf, dim=0)[torch.newaxis].to(self.device)

        return state

    def get_images(self):
        # Get latest value
        images = []
        for camera_name in self.camera_names:
            image = self.info["rgb_images"][camera_name]

            image = np.moveaxis(image, -1, -3)
            image = torch.tensor(image.copy(), dtype=torch.uint8)
            image = self.image_transforms(image)

            images.append(image)

        # Store and return
        if self.images_buf is None:
            self.images_buf = [
                [image for _ in range(self.model_meta_info["data"]["n_obs_steps"])]
                for image in images
            ]
        else:
            for single_images_buf, image in zip(self.images_buf, images):
                single_images_buf.pop(0)
                single_images_buf.append(image)

        images = torch.stack(
            [
                torch.stack(single_images_buf, dim=0)[torch.newaxis].to(self.device)
                for single_images_buf in self.images_buf
            ]
        )

        return images

    def infer_policy(self):
        # Infer
        if self.policy_action_buf is None or len(self.policy_action_buf) == 0:
            state = self.get_state()
            images = self.get_images()

            #何も考えず一旦ここをP2E仕様にする
            action = self.policy(state, images)[0]
            self.policy_action_buf = list(
                action.cpu().detach().numpy().astype(np.float64)
            )

        # Store action
        self.policy_action = denormalize_data(
            self.policy_action_buf.pop(0), self.model_meta_info["action"]
        )
        self.policy_action_list = np.concatenate(
            [self.policy_action_list, self.policy_action[np.newaxis]]
        )

    def draw_plot(self):
        # Clear plot
        for _ax in np.ravel(self.ax):
            _ax.cla()
            _ax.axis("off")

        # Plot images
        self.plot_images(self.ax[0, 0 : len(self.camera_names)])

        # Plot action
        self.plot_action(self.ax[1, 0])

        # Finalize plot
        self.canvas.draw()
        cv2.imshow(
            self.policy_name,
            cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR),
        )

    def reset(self):
        if not hasattr(self, "modified_episode_index"):
            self.modified_episode_index = 0
        # Reset plot
        if not self.args.no_plot:
            for _ax in np.ravel(self.ax):
                _ax.cla()
                _ax.axis("off")

            self.canvas = FigureCanvasAgg(self.fig)
            self.canvas.draw()
            cv2.imshow(
                self.policy_name,
                cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR),
            )

        # Reset motion manager
        self.motion_manager.reset()

        # Reset data manager
        self.data_manager.reset()

        # Reset environment
        self.env.unwrapped.world_random_scale = self.args.world_random_scale
        
        
        if self.data_manager.episode_idx == len(self.args.world_idx_list):
            self.data_manager.episode_idx = 0
                
        world_idx = self.args.world_idx_list[self.data_manager.episode_idx]
        self.data_manager.setup_env_world(world_idx)
        self.obs, self.info = self.env.reset(seed=self.args.seed)
        self.reward = 0
        msg = f"[{self.__class__.__name__}] Reset environment. demo_name: {self.demo_name}, world_idx: {self.data_manager.world_idx}, episode_idx: {self.data_manager.episode_idx}"
        if self.require_task_desc:
            msg += f", task desc: {self.args.task_desc}"
        
        self.modified_episode_index += 1

        # Reset phase manager
        self.phase_manager.reset()

        # Reset variables
        self.reset_variables()

    def get_data_filename(self):
        
        if not hasattr(self, "filename"):
            self.filename = None

        if self.filename is None:
            
            self.filename = super().get_data_filename()
            
        else:
            dirname = os.path.dirname(self.filename)
            #self.filename = os.path.join(
            #    dirname,
            #    f"{self.demo_name}_world{self.data_manager.world_idx:0>1}_{self.data_manager.episode_idx:0>3}.rmb",
            #)
            self.filename = os.path.join(
                dirname,
                f"{self.demo_name}_world{self.data_manager.world_idx:0>1}_{self.modified_episode_index:0>3}.rmb",
            )
        return self.filename