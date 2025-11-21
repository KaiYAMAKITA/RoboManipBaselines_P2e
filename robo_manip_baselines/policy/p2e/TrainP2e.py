import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from robo_manip_baselines.common import TrainBase
import importlib
import os
import random
import numpy as np
import pickle
import datetime
import copy

try:
    #from .P2eDataset import P2eDataset
    from .P2ePolicy import P2ePolicy
    from .RolloutP2e import RolloutP2e
    from ..data.CachedDataset import CachedDataset
    from ..utils.FileUtils import find_rmb_files
    from ..data.RmbData import RmbData
    from ..utils.DataUtils import get_skipped_data_seq
    from ..utils.FileUtils import find_rmb_files
    from ..data.DataKey import DataKey
except:
    #from robo_manip_baselines.policy.p2e.P2eDataset import P2eDataset
    from robo_manip_baselines.policy.p2e.P2ePolicy import P2ePolicy
    from robo_manip_baselines.policy.p2e.RolloutP2e import RolloutP2e
    from robo_manip_baselines.common.data.CachedDataset import CachedDataset
    from robo_manip_baselines.common.utils.FileUtils import find_rmb_files
    from robo_manip_baselines.policy.p2e.P2eDataset import build_p2e_attrdict_dataset
    from robo_manip_baselines.common.data.DataKey import DataKey
    from robo_manip_baselines.common.data.RmbData import RmbData
    from robo_manip_baselines.common.utils.DataUtils import get_skipped_data_seq
    from robo_manip_baselines.common.utils.FileUtils import find_rmb_files

import sys, os
repo_root = "../third_party/SimpleDreamer"
sys.path.insert(0, repo_root)
import importlib.util

spec = importlib.util.spec_from_file_location(
    "Plan2Explore",
    "../third_party/SimpleDreamer/dreamer/algorithms/plan2explore.py"
)
plan2explore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan2explore)
Plan2Explore = plan2explore.Plan2Explore

from attrdict import AttrDict
import yaml

def load_config(config_path: str) -> AttrDict:
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return AttrDict(config)

class TrainP2e(TrainBase):
    #DatasetClass = P2eDataset
    operation_parent_module_str = "robo_manip_baselines.envs.operation"
    policy_parent_module_str = "robo_manip_baselines.policy"
    policy_choices = [
        "Mlp",
        "Sarnn",
        "Act",
        "MtAct",
        "DiffusionPolicy",
        "DiffusionPolicy3d",
        "CnnMlp",
        "CnnReservoir",
        "CnnDwTcn",
        "P2e"
    ]
    max_buffer_num = 15
    def __init__(self):
        #self.stash = [sys.argv[sys.argv.index("--checkpoint")], sys.argv[sys.argv.index("--checkpoint")+1]]
        #del sys.argv[sys.argv.index("--checkpoint")+1]
        #del sys.argv[sys.argv.index("--checkpoint")]
        #sys.argv += ["--dataset_dir", "dummy"]
        sys.argv += ["--dataset_dir", "policy/p2e/test"]
        super().__init__()
        
        #self.setup_policy()
        
        self.setup_env()
        
        
        

    def setup_args(self, parser=None, argv=None):
        if parser is None:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                exit_on_error=False, #for debug
            )

        parser.add_argument(
            "--dataset_dir",
            type=str,
            required=True,
            help="dataset directory",
        )
        parser.add_argument(
            "--checkpoint_dir",
            type=str,
            default=None,
            help="checkpoint directory",
        )

        parser.add_argument(
            "--enable_rmb_cache",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Whether to enable data caching in RmbData. This uses RAM heavily, so it should be disabled on computers with small RAM.",
        )
        parser.add_argument(
            "--use_cached_dataset",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Whether to use CachedDataset. When enabling this, make sure that non-reproducible processes such as data augmentation are not performed on the original dataset.",
        )

        parser.add_argument(
            "--state_keys",
            type=str,
            nargs="*",
            default=[DataKey.MEASURED_JOINT_POS],
            choices=DataKey.MEASURED_DATA_KEYS,
            help="state data keys",
        )
        parser.add_argument(
            "--action_keys",
            type=str,
            nargs="+",
            default=[DataKey.COMMAND_JOINT_POS],
            choices=DataKey.COMMAND_DATA_KEYS,
            help="action data keys",
        )
        parser.add_argument(
            "--camera_names",
            type=str,
            nargs="+",
            default=["front"],
            help="camera names",
        )

        parser.add_argument(
            "--num_data",
            type=int,
            default=None,
            help="number of data files to use for learning (by default all files in the dataset_dir are used)",
        )
        parser.add_argument(
            "--train_ratio", type=float, default=0.8, help="ratio of train data"
        )
        parser.add_argument(
            "--val_ratio", type=float, default=None, help="ratio of validation data"
        )

        parser.add_argument(
            "--norm_type",
            type=str,
            default="gaussian",
            choices=["gaussian", "limits"],
            help="normalization type",
        )
        parser.add_argument(
            "--state_aug_std",
            type=float,
            default=0.0,
            help="Standard deviation of random noise added to state",
        )
        parser.add_argument(
            "--action_aug_std",
            type=float,
            default=0.0,
            help="Standard deviation of random noise added to action",
        )
        parser.add_argument(
            "--image_aug_std",
            type=float,
            default=0.0,
            help="Standard deviation of random noise added to images",
        )
        parser.add_argument(
            "--image_aug_erasing_scale",
            type=float,
            default=0.0,
            help="Scale of random erasing applied to the images",
        )
        parser.add_argument(
            "--image_aug_color_scale",
            type=float,
            default=0.0,
            help="Scale of color random noise added to the images",
        )
        parser.add_argument(
            "--image_aug_affine_scale",
            type=float,
            default=0.0,
            help="Scale of affine random noise added to the images",
        )

        parser.add_argument(
            "--skip",
            type=int,
            default=3,
            help="skip interval of data sequence (set 1 for no skip)",
        )

        parser.add_argument("--batch_size", type=int, help="batch size")
        parser.add_argument("--num_epochs", type=int, help="number of epochs")
        parser.add_argument("--lr", type=float, help="learning rate")
        parser.add_argument(
            "--num_workers", type=int, default=4, help="number of workers in dataloader"
        )

        parser.add_argument("--seed", type=int, default=42, help="random seed")
        self.set_additional_args(parser)
        

        if argv is None:
            argv = sys.argv
        
        self.args = parser.parse_args(argv[1:])
        #self.args, remaining_argv = parser.parse_known_args(argv[1:]) #for debug
        #sys.argv = [sys.argv[0]] + remaining_argv

        
        # Set checkpoint directory if it is not specified
        if self.args.checkpoint_dir is None:
            dataset_dirname = os.path.basename(os.path.normpath(self.args.dataset_dir))
            dataset_dirname = sys.argv[sys.argv.index("--env")+1]
            checkpoint_dirname = "{}_{}_{:%Y%m%d_%H%M%S}".format(
                dataset_dirname, self.policy_name, datetime.datetime.now()
            )
            self.args.checkpoint_dir = os.path.normpath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "checkpoint",
                    self.policy_name,
                    checkpoint_dirname,
                )
            )


    def setup_env(self):
        #case1
        from robo_manip_baselines.bin.Rollout import RolloutMain
        
        
        #envarg = sys.argv[sys.argv.index("--env")+1]
        self.stash = ["--checkpoint", "policy/p2e/model_meta_info.pkl"]
        #sys.argv = [sys.argv[0]] + ["P2e", envarg] + self.stash
        sys.argv = [sys.argv[0]] + self.stash
        
        sys.argv += ["--no_render", "--no_plot"]
        
        self.setup_rollout()
        self.rollout.policy.p2e.save(epoch=0)
        #self.replay_buffer = RolloutMain()
        self.args.checkpoint = self.args.checkpoint_dir

    def set_additional_args(self, parser):
        parser.set_defaults(enable_rmb_cache=True)

        parser.set_defaults(batch_size=32)
        parser.set_defaults(num_epochs=100)
        parser.set_defaults(lr=1e-5)

        
        parser.add_argument(
            "--policy", default="P2e", type=str, help="policy"
        )

        parser.add_argument(
            "--env", type=str, help="environment"
        )
        parser.add_argument(
            "--weight_decay", type=float, default=1e-4, help="weight decay"
        )

        parser.add_argument(
            "--hidden_dim_list",
            type=int,
            nargs="+",
            default=[512, 512],
            help="Dimension list of hidden layers",
        )
        parser.add_argument(
            "--state_feature_dim",
            type=int,
            default=512,
            help="Dimension of state feature",
        )
        parser.add_argument(
            "--n_obs_steps",
            type=int,
            default=1,
            help="number of steps in the observation sequence to input in the policy",
        )
        parser.add_argument(
            "--n_action_steps",
            type=int,
            default=1,
            help="number of steps in the action sequence to output from the policy",
        )
        parser.add_argument(
            "--config", type=str, help="configuration file"
        )


    def setup_model_meta_info(self):
        super().setup_model_meta_info()

        self.model_meta_info["data"]["n_obs_steps"] = self.args.n_obs_steps
        self.model_meta_info["data"]["n_action_steps"] = self.args.n_action_steps

    def setup_policy(self):
        # Set policy args
        self.model_meta_info["policy"]["args"] = {
            "n_obs_steps": self.args.n_obs_steps,
            "n_action_steps": self.args.n_action_steps,
            "hidden_dim_list": self.args.hidden_dim_list,
            "state_feature_dim": self.args.state_feature_dim,
        }

        #define P2E
        self.config = load_config("../third_party/SimpleDreamer/dreamer/configs/p2e-dmc-walker-walk.yml")
        self.config.operation.log_dir = self.args.checkpoint_dir
        """self.p2e = Plan2Explore(
            observation_shape=(3, 64, 64),
            discrete_action_bool=False,
            action_size=7,
            writer=SummaryWriter(log_dir="/tmp"),
            device="cuda",
            config=self.config,
            log_dir=self.args.checkpoint_dir,
        )

        self.policy = P2ePolicy(p2e=self.p2e)
        self.policy.p2e.save(epoch=0)"""

        # Print policy information
        #self.print_policy_info()
        #print(
        #    f"  - obs steps: {self.args.n_obs_steps}, action steps: {self.args.n_action_steps}"
        #)

    def train_loop(self):
        
        #rb_path = os.path.join(self.args.checkpoint_dir, "replay_buffer")
        #os.makedirs(rb_path, exist_ok=True)
        self.args.dataset_dir = None
        self.environment_interaction()
        pbar = tqdm(total = self.args.num_epochs * self.config.parameters.dreamer.collect_interval, desc="Training progress")
        for epoch in range(self.args.num_epochs):
            #collect replay buffer
            for iteration in range(self.config.parameters.dreamer.collect_interval):
                #for data in self.train_dataloader:
                #    print("aa")
                self.data = self.make_data() #rmbからp2eに使える形にデータ変換を行う
                for i in self.data.keys():
                    self.data[i] = self.data[i].to("cuda")
                """
                posterior, deterministics = self.policy.p2e.dynamic_learning(self.data)
                self.policy.p2e.behavior_learning(
                    self.policy.p2e.actor,
                    self.policy.p2e.critic,
                    self.policy.p2e.actor_optimizer,
                    self.policy.p2e.critic_optimizer,
                    posterior,
                    deterministics,
                )

                self.policy.p2e.behavior_learning(
                    self.policy.p2e.intrinsic_actor,
                    self.policy.p2e.intrinsic_critic,
                    self.policy.p2e.intrinsic_actor_optimizer,
                    self.policy.p2e.intrinsic_critic_optimizer,
                    posterior,
                    deterministics,
                )"""

                posterior, deterministics = self.rollout.policy.p2e.dynamic_learning(self.data)
                self.rollout.policy.p2e.behavior_learning(
                    self.rollout.policy.p2e.actor,
                    self.rollout.policy.p2e.critic,
                    self.rollout.policy.p2e.actor_optimizer,
                    self.rollout.policy.p2e.critic_optimizer,
                    posterior,
                    deterministics,
                )

                self.rollout.policy.p2e.behavior_learning(
                    self.rollout.policy.p2e.intrinsic_actor,
                    self.rollout.policy.p2e.intrinsic_critic,
                    self.rollout.policy.p2e.intrinsic_actor_optimizer,
                    self.rollout.policy.p2e.intrinsic_critic_optimizer,
                    posterior,
                    deterministics,
                )


                pbar.update(1)
                #print(f"{self.config.parameters.dreamer.collect_interval * epoch + iteration} iteration finished")
                self.rollout.policy.p2e.save(epoch=epoch * self.config.parameters.dreamer.collect_interval + iteration)
                
            self.environment_interaction()
            
                
            # Save current checkpoint
            if epoch % max(self.args.num_epochs // 10, 1) == 0:
                self.save_current_ckpt(f"epoch{epoch:0>5}")

        pbar.close()
        # Save last checkpoint
        self.save_current_ckpt("last")

        # Save best checkpoint
        self.save_best_ckpt()


    def make_data(self):
        
        data = build_p2e_attrdict_dataset(
            self.all_filenames,
            self.model_meta_info,
            enable_rmb_cache=self.args.enable_rmb_cache,
            device="cuda",

        )
        return data

    def setup_rollout(self):
        env_utils_spec = importlib.util.spec_from_file_location(
            "EnvUtils",
            os.path.join(os.path.dirname(__file__), "../..", "common/utils/EnvUtils.py"),
        )
        env_utils_module = importlib.util.module_from_spec(env_utils_spec)
        env_utils_spec.loader.exec_module(env_utils_module)

        if "Isaac" in self.args.env:
            from isaacgym import (
                gymapi,  # noqa: F401
                gymtorch,  # noqa: F401
                gymutil,  # noqa: F401
            )

        # This includes pytorch import, so it must be later than isaac import
        from robo_manip_baselines.common import camel_to_snake, remove_prefix

        operation_module = importlib.import_module(
            f"{self.operation_parent_module_str}.Operation{self.args.env}"
        )
        OperationEnvClass = getattr(operation_module, f"Operation{self.args.env}")

        policy_module = importlib.import_module(
            f"{self.policy_parent_module_str}.{camel_to_snake(self.args.policy)}"
        )
        RolloutPolicyClass = getattr(policy_module, f"Rollout{self.args.policy}")

        # The order of parent classes must not be changed in order to maintain the method resolution order (MRO)
        class Rollout(OperationEnvClass, RolloutPolicyClass):
            @property
            def policy_name(self):
                return remove_prefix(RolloutPolicyClass.__name__, "Rollout")

        if self.args.config is None:
            config = {}
        else:
            with open(self.args.config, "r") as f:
                config = yaml.safe_load(f)


        self.rollout = Rollout(**config, log_dir=self.args.checkpoint_dir)
        #self.rollout.checkpoint_dir = self.args.checkpoint_dir


    def run_rollout(self):
        self.rollout.run()
    
    def environment_interaction(self):
        #ロールアウトをする
        self.run_rollout()

        if self.args.dataset_dir is None:
            self.args.dataset_dir, _ = os.path.split(self.rollout.filename)

        #収録したリプレイバッファのフォルダに存在するすべてのリプレイバッファからデータを作る
        self.all_filenames = find_rmb_files(self.args.dataset_dir, num_files=self.args.num_data)
        #データセットの数が一定を超えていたら古いものから削除する
        if len(self.all_filenames) > self.max_buffer_num:
            self.all_filenames = self.all_filenames.sort()[-self.max_buffer_num:]
        #このデータセット作成の部分でrmbのデータをp2e用に変換する

        self.all_filenames.sort()
        self.make_data()


    def setup_dataset(self):
        #if self.all_filenames == []:
        #    return
        if self.args.enable_rmb_cache and self.args.use_cached_dataset:
            raise ValueError(
                f"[{self.__class__.__name__}] Both 'enable_rmb_cache' and 'use_cached_dataset' options cannot be True at the same time."
            )

        # Get file list
        num_files = len(self.all_filenames)
        train_num = max(int(np.clip(self.args.train_ratio, 0.0, 1.0) * num_files), 1)
        if self.args.val_ratio is None:
            val_num = max(num_files - train_num, 1)
        else:
            val_num = max(int(np.clip(self.args.val_ratio, 0.0, 1.0) * num_files), 1)
        train_filenames = self.all_filenames[:train_num]
        val_filenames = self.all_filenames[-1 * val_num :]

        # Set data stats
        self.set_data_stats()

        # Make dataloader
        #self.train_dataloader = self.make_dataloader(train_filenames, shuffle=True)
        #self.val_dataloader = self.make_dataloader(val_filenames, shuffle=False)

        self.data = build_p2e_attrdict_dataset(
            self.all_filenames,
            self.model_meta_info,
            enable_rmb_cache=self.args.enable_rmb_cache,
            device="cuda",
        )
        
        # Setup tensorboard
        #print(f"a {self.args.checkpoint}")
        self.writer = SummaryWriter(self.args.checkpoint_dir)

        # Print dataset information
        #self.print_dataset_info()

    def set_data_stats(self):
        # Load dataset
        all_state = []
        all_action = []
        rgb_image_example = None
        depth_image_example = None
        episode_len_list = []
        #if self.all_filenames == []:
        #    return
        for filename in self.all_filenames:
            with RmbData(filename) as rmb_data:
                episode_len = rmb_data[DataKey.TIME][:: self.args.skip].shape[0]
                episode_len_list.append(episode_len)

                # Load state
                if len(self.args.state_keys) == 0:
                    state = np.zeros((episode_len, 0), dtype=np.float64)
                else:
                    state = np.concatenate(
                        [
                            get_skipped_data_seq(rmb_data[key][:], key, self.args.skip)
                            for key in self.args.state_keys
                        ],
                        axis=1,
                    )
                all_state.append(state)

                # Load action
                if len(self.args.action_keys) == 0:
                    action = np.zeros((episode_len, 0), dtype=np.float64)
                else:
                    action = np.concatenate(
                        [
                            get_skipped_data_seq(rmb_data[key][:], key, self.args.skip)
                            for key in self.args.action_keys
                        ],
                        axis=1,
                    )
                all_action.append(action)

                # Load image
                if rgb_image_example is None:
                    rgb_image_example = {
                        camera_name: rmb_data[DataKey.get_rgb_image_key(camera_name)][0]
                        for camera_name in self.args.camera_names
                        if DataKey.get_rgb_image_key(camera_name) in rmb_data
                    }
                if depth_image_example is None:
                    depth_image_example = {
                        camera_name: rmb_data[DataKey.get_depth_image_key(camera_name)][
                            0
                        ]
                        for camera_name in self.args.camera_names
                        if DataKey.get_depth_image_key(camera_name) in rmb_data
                    }
        all_state = np.concatenate(all_state, dtype=np.float64)
        all_action = np.concatenate(all_action, dtype=np.float64)

        self.model_meta_info["state"].update(self.calc_stats_from_seq(all_state))
        self.model_meta_info["action"].update(self.calc_stats_from_seq(all_action))
        self.model_meta_info["image"].update(
            {
                "rgb_example": rgb_image_example,
                "depth_example": depth_image_example,
            }
        )
        self.model_meta_info["data"].update(
            {
                "mean_episode_len": np.mean(episode_len_list),
                "min_episode_len": np.min(episode_len_list),
                "max_episode_len": np.max(episode_len_list),
            }
        )

    def setup_rmb_files(self):
        
        #if self.args.dataset_dir is "dummy":
        #    self.all_filenames = []
        #else:
        self.all_filenames = find_rmb_files(
            self.args.dataset_dir, num_files=self.args.num_data
        )
        print(f"polsstht{self.args.dataset_dir}")
        
        #self.args.dataset_dirがself.op.get_data_filename()になるようにすればいい。
        #毎度更新されるself.op.get_data_filename()をどこかに保存しておけるようにしておきたい。どうする
        random.shuffle(self.all_filenames)

    def run(self):
        # Save model meta info
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        model_meta_info_path = os.path.join(
            self.args.checkpoint_dir, "model_meta_info.pkl"
        )
        with open(model_meta_info_path, "wb") as f:
            pickle.dump(self.model_meta_info, f)
        print(
            f"[{self.__class__.__name__}] Save model meta info: {model_meta_info_path}"
        )

        # Train loop
        print(
            f"[{self.__class__.__name__}] Train with saving checkpoints: {self.args.checkpoint_dir}"
        )
        self.best_ckpt_info = {"loss": np.inf, "epoch": -1}
        self.train_loop()


    def update_best_ckpt(self, epoch_summary, policy=None):
        if policy is None:
            policy = self.rollout.policy

        if epoch_summary["loss"] < self.best_ckpt_info["loss"]:
            self.best_ckpt_info = {
                "epoch": epoch_summary["epoch"],
                "loss": epoch_summary["loss"],
                "state_dict": copy.deepcopy(policy.state_dict()),
            }

    def save_current_ckpt(self, ckpt_suffix, policy=None):
        if policy is None:
            policy = self.rollout.policy

        ckpt_path = os.path.join(self.args.checkpoint_dir, f"policy_{ckpt_suffix}.ckpt")
        torch.save(policy.state_dict(), ckpt_path)

if __name__ == "__main__":
    p2e = Plan2Explore()