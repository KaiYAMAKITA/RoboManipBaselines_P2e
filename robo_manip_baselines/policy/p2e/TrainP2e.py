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
        
        self.stash = [sys.argv[sys.argv.index("--checkpoint")], sys.argv[sys.argv.index("--checkpoint")+1]]
        del sys.argv[sys.argv.index("--checkpoint")+1]
        del sys.argv[sys.argv.index("--checkpoint")]
        super().__init__()
        
        self.setup_policy()
        
        self.setup_env()
        
        
        
    def setup_env(self):
        #case1
        from robo_manip_baselines.bin.Rollout import RolloutMain
        
        print(f"]/]/]/]/]/ {sys.argv}")
        envarg = sys.argv[sys.argv.index("--env")+1]
        sys.argv = [sys.argv[0]] + ["P2e", envarg] + self.stash
        #sys.argv += ["--checkpoint", self.args.checkpoint_dir]
        
        # checkpointはreplaybuffer用
        # checkpoint=dirはその他
        self.replay_buffer = RolloutMain()
        print(f"これをみよ {self.replay_buffer.args}")
        print(sys.argv)
        self.replay_buffer.args.checkpoint = self.args.checkpoint_dir
        #print("sdasda")
        #print(self.replay_buffer.args)
        self.args.dataset_dir = os.path.join(self.args.checkpoint_dir, "replay_buffer")
        #print(f"sdf{self.args.dataset_dir}")

        self.replay_buffer.args.save_rollout = True
        self.replay_buffer.args.auto_exit = True
        

    def set_additional_args(self, parser):
        print("aaaaaaaっs")
        parser.set_defaults(enable_rmb_cache=True)

        parser.set_defaults(batch_size=32)
        parser.set_defaults(num_epochs=40)
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
        print("dd")


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
        self.p2e = Plan2Explore(
            observation_shape=(3, 480, 640),
            discrete_action_bool=False,
            action_size=7,
            writer=SummaryWriter(log_dir="/tmp"),
            device="cuda",
            config=self.config,
        )

        self.policy = P2ePolicy(p2e=self.p2e)

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
        rb_path = os.path.join(self.args.checkpoint_dir, "replay_buffer")
        os.makedirs(rb_path, exist_ok=True)
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
            
            self.replay_buffer.args.checkpoint = os.path.join(self.args.checkpoint_dir, "dummy")
            
            replay_buffer_num = 20
            if len(os.listdir(rb_path)) >= replay_buffer_num:
                os.remove(os.path.join(rb_path, os.listdir(rb_path).sort()[:len(os.listdir(rb_path))-replay_buffer_num]))

            #collect replay buffer
            print(self.config)
            for iteration in range(self.config.parameters.dreamer.train_iterations):
                if iteration % self.config.parameters.dreamer.collect_interval == 0:
                    for _ in range(self.config.parameters.dreamer.batch_size):
                        print("ここまで来た。")
                        self.replay_buffer.run() #これはenvironment_interactionを利用する想定。この中でp2eの学習に必要な要素が揃う。
                        #parser.get_parser(), args1 = parser.parse_args(["--save_rollout", "True"])としたいが、よくわからないので便宜上直接代入することとすru
                        print("ここまでは？")
                    #setup_dataset
                    self.setup_dataset()
                #dataの型が合わない（done, rewardはともかくnextobservationをどう入れるか）
                #plan2exploreの333行目
                #rollout-> run -> reward
                for data in self.train_dataloader:
                    self.policy.update(data[0], data[1], data[2])
                
                







            # Save current checkpoint
            if epoch % max(self.args.num_epochs // 10, 1) == 0:
                self.save_current_ckpt(f"epoch{epoch:0>3}")

        # Save last checkpoint
        self.save_current_ckpt("last")

        # Save best checkpoint
        self.save_best_ckpt()

if __name__ == "__main__":
    p2e = Plan2Explore()