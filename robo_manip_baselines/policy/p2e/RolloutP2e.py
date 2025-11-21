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
import argparse
import time

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

    checkpoint_dir = ""

    def __init__(self, log_dir=""):
        self.checkpoint_dir = log_dir
        super().__init__()

    def setup_args(self, parser=None, argv=None):

        if parser is None:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                exit_on_error=False, #for debug
            )

        parser.add_argument(
            "--checkpoint", type=str, required=True, help="checkpoint file"
        )

        parser.add_argument(
            "--world_idx",
            type=int,
            default=0,
            help="world index (if '--world_idx_list' option is specified, it takes precedence)",
        )
        parser.add_argument(
            "--world_idx_list",
            type=int,
            nargs="*",
            default=None,
            help="list of world indexes",
        )
        parser.add_argument(
            "--world_idx_repeat_count",
            type=int,
            default=1,
            help="number of times to repeat world indexes",
        )
        parser.add_argument(
            "--world_random_scale",
            nargs="+",
            type=float,
            default=None,
            help="random scale of simulation world (no randomness by default)",
        )

        parser.add_argument(
            "--skip",
            type=int,
            help="step interval to infer policy",
        )
        parser.add_argument(
            "--skip_draw",
            type=int,
            help="step interval to draw the plot",
        )

        parser.add_argument("--seed", type=int, default=-1, help="random seed")

        parser.add_argument(
            "--no_render",
            action="store_true",
            help="whether to disable simulation rendering",
        )
        parser.add_argument(
            "--no_plot", action="store_true", help="whether to disable policy plot"
        )
        parser.add_argument(
            "--win_xy_plot",
            type=int,
            nargs=2,
            help="xy position of window to plot policy information",
        )

        parser.add_argument(
            "--wait_before_start",
            action="store_true",
            help="whether to wait a key input before starting motion",
        )
        parser.add_argument(
            "--auto_exit",
            action="store_true",
            help="whether to automatically exit from rollout",
        )
        parser.add_argument(
            "--max_duration",
            type=float,
            default=30.0,
            help=(
                "maximum rollout duration for automatic exit [s] "
                "(used only when '--auto_exit' option is enabled)"
            ),
        )

        parser.add_argument(
            "--save_rollout",
            action="store_true",
            help="whether to save rollout data",
        )

        parser.add_argument(
            "--result_filename",
            type=str,
            default=None,
            help="File path (*.yaml) to save rollout results (default: do not save)",
        )

        parser.add_argument(
            "--save_last_image",
            action="store_true",
            help="whether to save the observation image of the last frame",
        )
        parser.add_argument(
            "--output_image_dir",
            type=str,
            default=".",
            help=(
                "directory to save the output image (default: current directory, "
                "used only when '--output_image_dir' option is enabled)."
            ),
        )

        parser.add_argument(
            "--demo_name", type=str, default="", help="demonstration name"
        )
        parser.add_argument(
            "--target_task", type=str, default=None, help="target task name"
        )
        if self.require_task_desc:
            parser.add_argument(
                "--task_desc", type=str, required=True, help="task description"
            )

        self.set_additional_args(parser)

        if argv is None:
            argv = sys.argv
        self.args = parser.parse_args(argv[1:])
        

        if self.args.world_idx_list is None:
            self.args.world_idx_list = [self.args.world_idx]
        self.args.world_idx_list *= self.args.world_idx_repeat_count

        if self.args.world_random_scale is not None:
            self.args.world_random_scale = np.array(self.args.world_random_scale)

        if self.args.seed < 0:
            self.args.seed = int(time.time()) % (2**32)

    def setup_policy(self):
        
        #self.args.
        # Print policy information
        self.print_policy_info()
        print(
            f"  - obs steps: {self.model_meta_info['data']['n_obs_steps']}, action steps: {self.model_meta_info['data']['n_action_steps']}"
        )

        #define P2E

        self.config = load_config("../third_party/SimpleDreamer/dreamer/configs/p2e-dmc-walker-walk.yml")
        self.config.operation.log_dir = self.checkpoint_dir
        self.p2e = Plan2Explore(
            observation_shape=(3, 64, 64),
            discrete_action_bool=False,
            action_size=7,
            writer=SummaryWriter(log_dir="/tmp"),
            device="cuda",
            config=self.config,
            log_dir=self.checkpoint_dir
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