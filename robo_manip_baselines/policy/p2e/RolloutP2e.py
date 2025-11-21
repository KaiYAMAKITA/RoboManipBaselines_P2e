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
import torch.nn.functional as F

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
            observation_shape=(3, 64, 64),
            discrete_action_bool=False,
            action_size=7,
            writer=SummaryWriter(log_dir="/tmp"),
            device="cuda",
            config=self.config,
        )

        self.policy = P2ePolicy(p2e=self.p2e)

        
        self.args.world_idx_list = [0 for i in range(5)]
        self.args.save_rollout = True
        self.args.auto_exit = True
        self.args.max_duration = 10.0
        self.args.no_render = True
        
        
        #load checkpoint
        #self.load_ckpt()
        self.device = torch.device("cuda")

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


        assert len(self.camera_names) == 1

        image = self.info["rgb_images"][self.camera_names[0]]


        # --- ① 横を480に中央クロップ ---
        H, W, C = image.shape
        target_w = 480
        left = (W - target_w) // 2
        right = left + target_w
        image_crop = image[:, left:right, :]     # shape: (480, 480, 3)

        # --- ② cv2 で 64×64 にリサイズ ---
        image = cv2.resize(
            image_crop,
            (64, 64),           # (width, height)
            interpolation=cv2.INTER_AREA
        )



        image = np.moveaxis(image, -1, -3).copy()
        image = torch.tensor(image, dtype=torch.uint8)
        image = self.image_transforms(image)[torch.newaxis].to(self.device)
        """for camera_name in self.camera_names:
            image = self.info["rgb_images"][camera_name]

            image = np.moveaxis(image, -1, -3)
            image = torch.tensor(image.copy(), dtype=torch.uint8)
            image = self.image_transforms(image)[torch.newaxis].to(self.device)

            images.append(image)
"""
        # Store and return
        """if self.images_buf is None:
            self.images_buf = [
                [image for _ in range(self.model_meta_info["data"]["n_obs_steps"])]
                for image in images
            ]
        else:
            for single_images_buf, image in zip(self.images_buf, images):
                single_images_buf.pop(0)
                single_images_buf.append(image)
"""
        """images = torch.stack(
            [
                torch.stack(single_images_buf, dim=0)[torch.newaxis].to(self.device)
                for single_images_buf in self.images_buf
            ]
        )"""
        



        return image

    def infer_policy(self):
        # Infer
        #if self.policy_action_buf is None or len(self.policy_action_buf) == 0:
        #stagte = self.get_state()
        image = self.get_images()
        #何も考えず一旦ここをP2E仕様にする
        #action = self.policy(state, images)[0] #バッチサイズ１のバッチのうち一つ
        # environment interaction

        
        
        embedded_observation = self.policy.p2e.encoder(image)
        self.deterministic = self.policy.p2e.rssm.recurrent_model(
            self.posterior, self.action, self.deterministic
        )
        embedded_observation = embedded_observation.reshape(1, -1)
        _, self.posterior = self.policy.p2e.rssm.representation_model(
            embedded_observation, self.deterministic
        )
        self.action = self.policy.p2e.intrinsic_actor(self.posterior, self.deterministic).detach()
        ###

        # Store action
        self.policy_action = denormalize_data(
            self.action[0].cpu().detach().numpy().astype(np.float64), self.model_meta_info["action"]
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


    def get_data_filename(self):
        
        if not hasattr(self, "filename"):
            self.filename = None

        if self.filename is None:
            self.modified_episode_index = 0
            self.filename = super().get_data_filename()
            
        else:
            dirname = os.path.dirname(self.filename)
            #self.filename = os.path.join(
            #    dirname,
            #    f"{self.demo_name}_world{self.data_manager.world_idx:0>1}_{self.data_manager.episode_idx:0>3}.rmb",
            #)
            self.modified_episode_index += 1
            self.filename = os.path.join(
                dirname,
                f"{self.demo_name}_world{self.data_manager.world_idx:0>1}_{self.modified_episode_index:0>5}.rmb",
            )
        return self.filename
    
    def run(self):
        
        self.reset_flag = True
        self.quit_flag = False
        self.inference_duration_list = []
        self.data_manager.episode_idx = 0
        
        while True:
            if self.reset_flag:
                self.reset()
                self.reset_flag = False

            self.phase_manager.pre_update()

            env_action = np.concatenate(
                [
                    self.motion_manager.get_command_data(key)
                    for key in self.env.unwrapped.command_keys_for_step
                ]
            )
            
            if self.args.save_rollout and self.phase_manager.is_phase("RolloutPhase"):
                self.record_data()

            self.obs, self.reward, _, _, self.info = self.env.step(env_action)

            self.phase_manager.post_update()

            self.key = cv2.waitKey(1)
            self.phase_manager.check_transition()

            if self.key == 27:  # escape key
                self.quit_flag = True
            if self.quit_flag:
                break
            
        
        if self.args.result_filename is not None:
            #print(
            #    f"[{self.__class__.__name__}] Save the rollout results: {self.args.result_filename}"
            #)
            with open(self.args.result_filename, "w") as result_file:
                yaml.dump(self.result, result_file)

        self.print_statistics()

    def reset_variables(self):
        super().reset_variables()

        #enviroment interaction
        self.posterior, self.deterministic = self.policy.p2e.rssm.recurrent_model_input_init(1)
        self.action = torch.zeros(1, self.policy.p2e.action_size).to(self.device)

        self.score = 0
        self.score_lst = np.array([])
        self.done = False
        ###