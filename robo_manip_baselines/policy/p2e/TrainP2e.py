import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from robo_manip_baselines.common import TrainBase
import importlib
import os
import numpy as np


try:
    #from .P2eDataset import P2eDataset
    from .P2ePolicy import P2ePolicy
    from .RolloutP2e import RolloutP2e
    from ..data.CachedDataset import CachedDataset
    from ..utils.FileUtils import find_rmb_files
except:
    #from robo_manip_baselines.policy.p2e.P2eDataset import P2eDataset
    from robo_manip_baselines.policy.p2e.P2ePolicy import P2ePolicy
    from robo_manip_baselines.policy.p2e.RolloutP2e import RolloutP2e
    from robo_manip_baselines.common.data.CachedDataset import CachedDataset
    from robo_manip_baselines.common.utils.FileUtils import find_rmb_files
    from robo_manip_baselines.policy.p2e.P2eDataset import build_p2e_attrdict_dataset

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
    def __init__(self):
        
        self.stash = [sys.argv[sys.argv.index("--checkpoint")], sys.argv[sys.argv.index("--checkpoint")+1]]
        del sys.argv[sys.argv.index("--checkpoint")+1]
        del sys.argv[sys.argv.index("--checkpoint")]
        super().__init__()
        
        self.setup_policy()
        
        self.setup_env()
        
        
        
    def setup_env(self):
        #case1
        from robo_manip_baselines.bin.Rollout import RolloutMain
        
        
        envarg = sys.argv[sys.argv.index("--env")+1]
        #sys.argv = [sys.argv[0]] + ["P2e", envarg] + self.stash
        sys.argv = [sys.argv[0]] + self.stash

        self.setup_rollout()
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
        print("checkpoint dir", self.args.checkpoint_dir)
        self.config.operation.log_dir = self.args.checkpoint_dir
        print("uhhosu", self.config.operation.log_dir)
        self.p2e = Plan2Explore(
            observation_shape=(3, 64, 64),
            discrete_action_bool=False,
            action_size=7,
            writer=SummaryWriter(log_dir="/tmp"),
            device="cuda",
            config=self.config,
            log_dir=self.args.checkpoint_dir,
        )

        self.policy = P2ePolicy(p2e=self.p2e)
        self.policy.p2e.save(epoch=0)

        """
        # Construct policy
        self.policy = P2ePolicy(
            len(self.model_meta_info["state"]["example"]),
            len(self.model_meta_info["action"]["example"]),
            len(self.args.camera_names),
            **self.model_meta_info["policy"]["args"],
        )"""
        #self.policy.cuda()

        """
        # Construct optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        """

        # Print policy information
        self.print_policy_info()
        print(
            f"  - obs steps: {self.args.n_obs_steps}, action steps: {self.args.n_action_steps}"
        )

    def train_loop(self):
        
        #rb_path = os.path.join(self.args.checkpoint_dir, "replay_buffer")
        #os.makedirs(rb_path, exist_ok=True)
        self.args.dataset_dir = None
        self.environment_interaction()
        for epoch in tqdm(range(self.args.num_epochs)):
            # Run train step
            """a = 0
            self.policy.train()
            batch_result_list = []
            for data in self.train_dataloader:
                a = a + 1
                print(a)
                self.optimizer.zero_grad()
                pred_action = self.policy(*[d.cuda() for d in data[0:2]])
                loss = F.l1_loss(pred_action, data[2].cuda())
                loss.backward()
                self.optimizer.step()
                batch_result_list.append(self.detach_batch_result({"loss": loss}))
            self.log_epoch_summary(batch_result_list, "train", epoch)

            # Run validation step
            with torch.inference_mode():
                self.policy.eval()
                batch_result_list = []
                for data in self.val_dataloader:
                    pred_action = self.policy(*[d.cuda() for d in data[0:2]])
                    loss = F.l1_loss(pred_action, data[2].cuda())
                    batch_result_list.append(self.detach_batch_result({"loss": loss}))
                epoch_summary = self.log_epoch_summary(batch_result_list, "val", epoch)

                # Update best checkpoint
                self.update_best_ckpt(epoch_summary)
            """
            
            #self.replay_buffer.args.checkpoint = os.path.join(self.args.checkpoint_dir, "dummy")
            
            replay_buffer_num = 30
            #if len(os.listdir(rb_path)) >= replay_buffer_num:
            #    os.remove(os.path.join(rb_path, os.listdir(rb_path).sort()[:len(os.listdir(rb_path))-replay_buffer_num]))

            #collect replay buffer
            for iteration in range(self.config.parameters.dreamer.collect_interval):
                #for data in self.train_dataloader:
                #    print("aa")
                self.data = self.make_data() #rmbからp2eに使える形にデータ変換を行う
            
                posterior, deterministics = self.policy.dynamic_learning(self.data)
                self.policy.p2e.behavior_learning(
                    self.policy.p2e.actor,
                    self.policy.p2e.critic,
                    self.policy.p2e.actor_optimizer,
                    self.policy.p2e.critic_optimizer,
                    posterior,
                    deterministics,
                )

                self.policy.p2e.behavior_learning(
                    self.intrinsic_actor,
                    self.intrinsic_critic,
                    self.intrinsic_actor_optimizer,
                    self.intrinsic_critic_optimizer,
                    posterior,
                    deterministics,
                )
                print(f"f{iteration} iteration finished")
                

            self.environment_interaction()
            
                





            # Save current checkpoint
            if epoch % max(self.args.num_epochs // 10, 1) == 0:
                self.save_current_ckpt(f"epoch{epoch:0>3}")

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

        """parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="This is a meta argument parser for the rollout switching between different policies and environments. The actual arguments are handled by another internal argument parser.",
            fromfile_prefix_chars="@",
            add_help=False,
        )
        parser.add_argument(
            "policy",
            type=str,
            nargs="?",
            default=None,
            choices=self.policy_choices,
            help="policy",
        )
        parser.add_argument(
            "env",
            type=str,
            help="environment",
            nargs="?",
            default=None,
            choices=env_utils_module.get_env_names(
                operation_parent_module_str=self.operation_parent_module_str
            ),
        )
        parser.add_argument("--config", type=str, help="configuration file")
        parser.add_argument(
            "-h",
            "--help",
            action="store_true",
            help="Show this help message and continue",
        )

        self.args, remaining_argv = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + remaining_argv
        if self.args.policy is None or self.args.env is None:
            parser.print_help()
            sys.exit(1)
        elif self.args.help:
            parser.print_help()
            print("\n================================\n")
            sys.argv += ["--help"]"""
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


        self.rollout = Rollout(**config)


    def run_rollout(self):
        self.rollout.run()
    
    def environment_interaction(self):
        #ロールアウトをする
        self.run_rollout()
        print("rollout finished")
        print(self.args.dataset_dir)
        if self.args.dataset_dir is None:
            self.args.dataset_dir, _ = os.path.split(self.rollout.filename)
        print(self.args.dataset_dir)
        #収録したリプレイバッファのフォルダに存在するすべてのリプレイバッファからデータを作る
        self.all_filenames = find_rmb_files(self.args.dataset_dir, num_files=self.args.num_data)
        print(self.all_filenames)
        #このデータセット作成の部分でrmbのデータをp2e用に変換する
        self.make_data()


    def setup_dataset(self):
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
        print("ohha")

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
    """
    def make_dataloader(self, filenames, shuffle=True):
        dataset = self.DatasetClass(
            filenames, self.model_meta_info, self.args.enable_rmb_cache
        )
        dataset = build_p2e_attrdict_dataset(
            filenames,
            self.model_meta_info,
            enable_rmb_cache=self.args.enable_rmb_cache,
            device="cpu",
        )

        if self.args.use_cached_dataset:
            dataset = CachedDataset(dataset)

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=self.args.num_workers,
            persistent_workers=True,
            prefetch_factor=4,
        )
    
        return dataloader
    """


if __name__ == "__main__":
    p2e = Plan2Explore()