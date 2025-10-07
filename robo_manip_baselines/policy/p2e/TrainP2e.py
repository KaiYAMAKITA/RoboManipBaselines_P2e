import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
from robo_manip_baselines.common import TrainBase
import importlib
import os


try:
    from .P2eDataset import P2eDataset
    from .P2ePolicy import P2ePolicy
    from .RolloutP2e import RolloutP2e
except:
    from robo_manip_baselines.policy.p2e.P2eDataset import P2eDataset
    from robo_manip_baselines.policy.p2e.P2ePolicy import P2ePolicy
    from robo_manip_baselines.policy.p2e.RolloutP2e import RolloutP2e

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
    DatasetClass = P2eDataset
    operation_parent_module_str = "robo_manip_baselines.envs.operation"
    policy_parent_module_str = "robo_manip_baselines.policy"
    def __init__(self):
        super().__init__()
        print(f"TrainP2e {sys.argv}")

        #self.setup_args_all()
        self.setup_policy()
        self.setup_env()
        
        # sys.argvにダミーの--checkpointを追加
        #import sys
        #if "--checkpoint" not in sys.argv:
        #    sys.argv += ["--checkpoint", ""]
        #self.excluded_args = []
        #print(f"sys.argv: {sys.argv}")
        #sys.argvからrolloutに必要なargumentを削除する
        #arg = sys.argv.index("--dataset_dir")
        #del sys.argv[arg:arg+2]

        #これをsetup_env内でどうにかする
        #self.replay_buffer = RolloutP2e()
        #self.replay_buffer.args.save_rollout = True
    def setup_env(self):
        if "--checkpoint" not in sys.argv:
            sys.argv += ["--checkpoint", ""]
        self.excluded_args = []
        print(f"sys.argv: {sys.argv}")
        from robo_manip_baselines.common import camel_to_snake, remove_prefix
        env_utils_spec = importlib.util.spec_from_file_location(
            "EnvUtils",
            os.path.join(os.path.dirname(__file__), "../..", "common/utils/EnvUtils.py"),
        )
        print(f"これ実行されますかね： {self.args}")
        env_utils_module = importlib.util.module_from_spec(env_utils_spec)
        env_utils_spec.loader.exec_module(env_utils_module)
        self.operation_parent_module_str = "robo_manip_baselines.envs.operation"
        self.policy_parent_module_str = "robo_manip_baselines.policy"
        if self.args.env is not None:
            self.operation_module = importlib.import_module(
                f"{self.operation_parent_module_str}.Operation{self.args.env}"
            )
            self.OperationEnvClass = getattr(self.operation_module, f"Operation{self.args.env}")
            self.policy_module = importlib.import_module(
                f"{self.policy_parent_module_str}.{camel_to_snake(self.args.policy)}"
            )
            self.RolloutPolicyClass = getattr(self.policy_module, f"Rollout{self.args.policy}")
        else:
            assert False, "Please specify --environment"
        class Rollout(self.OperationEnvClass, self.RolloutPolicyClass):
            @property
            def policy_name(self):
                return remove_prefix(self.RolloutPolicyClass.__name__, "Rollout")
            
        if self.args.config is None:
            config = {}
        else:
            with open(self.args.config, "r") as f:
                config = yaml.safe_load(f)
        
        self.replay_buffer = Rollout(**config)
        self.replay_buffer.args.save_rollout = True

    def setup_policy(self):
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

    def set_additional_args(self, parser):
        parser.set_defaults(enable_rmb_cache=True)

        parser.set_defaults(batch_size=32)
        parser.set_defaults(num_epochs=40)
        parser.set_defaults(lr=1e-5)
        print("これが実行されている")
        #Rollout.pyを踏襲
        parser.add_argument(
            "--env", type=str, default=None, help="environment"
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

        # Construct policy
        self.policy = P2ePolicy(
            len(self.model_meta_info["state"]["example"]),
            len(self.model_meta_info["action"]["example"]),
            len(self.args.camera_names),
            **self.model_meta_info["policy"]["args"],
        )
        self.policy.cuda()

        # 暫定
        # self.policy = Plan2Explore(...)

        # for environment interaction
        # self.rollout = Rollout(...)  # from RolloutP2e.py
        # self.rollout.run === environment_interaction

        # Construct optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        # Print policy information
        self.print_policy_info()
        print(
            f"  - obs steps: {self.args.n_obs_steps}, action steps: {self.args.n_action_steps}"
        )

    def train_loop(self):
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
            
            replay_buffer_num = 20
            dataset_dir = "ddir"
            if len(os.listdir(dataset_dir)) >= replay_buffer_num:
                os.remove(os.path.join(dataset_dir, os.listdir(dataset_dir).sort()[:len(os.listdir(dataset_dir))-replay_buffer_num]))

            #collect replay buffer
            for _ in range(3):
                self.replay_buffer.run(self.policy, self.args.num_envs)
                #parser.get_parser(), args1 = parser.parse_args(["--save_rollout", "True"])としたいが、よくわからないので便宜上直接代入することとすru







            # Save current checkpoint
            if epoch % max(self.args.num_epochs // 10, 1) == 0:
                self.save_current_ckpt(f"epoch{epoch:0>3}")

        # Save last checkpoint
        self.save_current_ckpt("last")

        # Save best checkpoint
        self.save_best_ckpt()

if __name__ == "__main__":
    p2e = Plan2Explore()