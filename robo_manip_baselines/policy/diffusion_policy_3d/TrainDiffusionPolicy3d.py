import argparse
import copy

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from omegaconf import OmegaConf
from tqdm import tqdm

from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
from diffusion_policy_3d.policy.dp3 import DP3
from robo_manip_baselines.common import TrainBase

from .DiffusionPolicy3dDataset import DiffusionPolicy3dDataset


class TrainDiffusionPolicy3d(TrainBase):
    DatasetClass = DiffusionPolicy3dDataset

    def setup_args(self):
        super().setup_args()

    def set_additional_args(self, parser):
        parser.set_defaults(enable_rmb_cache=True)

        parser.set_defaults(norm_type="limits")

        parser.set_defaults(batch_size=128)
        parser.set_defaults(num_epochs=3000)
        parser.set_defaults(lr=1e-4)

        parser.add_argument(
            "--weight_decay", type=float, default=1e-6, help="weight decay"
        )

        parser.add_argument(
            "--use_ema",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable or disable exponential moving average (EMA)",
        )

        parser.add_argument(
            "--use_pc_color",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Enable or disable color information of pointcloud",
        )

        parser.add_argument(
            "--horizon", type=int, default=16, help="prediction horizon"
        )
        parser.add_argument(
            "--n_obs_steps",
            type=int,
            default=2,
            help="number of steps in observation to input in the policy",
        )
        parser.add_argument(
            "--n_action_steps",
            type=int,
            default=8,
            help="number of steps in the action to output from the policy",
        )

        parser.add_argument(
            "--num_points",
            type=int,
            default=512,
            help="number of points in one pointcloud.",
        )

        parser.add_argument(
            "--image_size",
            type=int,
            nargs=2,
            default=[84, 84],
            help="Image size (width, height) to be resized before crop. In the case of multiple image inputs, it is assumed that all images share the same size.",
        )
        parser.add_argument(
            "--image_crop_size",
            type=int,
            nargs=2,
            default=[80, 80],
            help="Image size (width, height) to be cropped after resize. In the case of multiple image inputs, it is assumed that all images share the same size.",
        )
        parser.add_argument(
            "--encoder_output_dim",
            type=int,
            nargs=1,
            default=64,
            help="number of output dimensions of encoder in policy.",
        )

    def setup_model_meta_info(self):
        super().setup_model_meta_info()

        self.model_meta_info["data"]["image_size"] = self.args.image_size
        self.model_meta_info["data"]["image_crop_size"] = self.args.image_crop_size
        self.model_meta_info["data"]["horizon"] = self.args.horizon
        self.model_meta_info["data"]["n_obs_steps"] = self.args.n_obs_steps
        self.model_meta_info["data"]["n_action_steps"] = self.args.n_action_steps
        self.model_meta_info["data"]["num_points"] = self.args.num_points
        self.model_meta_info["data"]["n_point_dim"] = 6 if self.args.use_pc_color else 3

        self.model_meta_info["policy"]["use_ema"] = self.args.use_ema

    def get_extra_norm_config(self):
        if self.args.norm_type == "limits":
            return {
                "out_min": -1.0,
                "out_max": 1.0,
            }
        else:
            return super().get_extra_norm_config()

    def setup_policy(self):
        # Set policy args
        shape_meta = OmegaConf.create(
            {
                "obs": {"point_cloud": {}},
                "action": {"shape": [len(self.model_meta_info["action"]["example"])]},
            }
        )
        if len(self.args.state_keys) > 0:
            shape_meta["obs"]["state"] = {
                "shape": [len(self.model_meta_info["state"]["example"])],
                "type": "low_dim",
            }
        pointcloud_shape = (
            self.model_meta_info["data"]["num_points"],
            self.model_meta_info["data"]["n_point_dim"],
        )
        shape_meta["obs"]["point_cloud"] = {
            "shape": pointcloud_shape,
            "type": "point_cloud",
        }
        pointcloud_encoder_conf = OmegaConf.create(
            {
                "in_channels": self.model_meta_info["data"]["n_point_dim"],
                "out_channels": self.args.encoder_output_dim,
                "use_layernorm": True,
                "final_norm": "layernorm",
                "normal_channel": False,
            }
        )
        self.model_meta_info["policy"]["args"] = {
            "shape_meta": shape_meta,
            "horizon": self.args.horizon,
            "n_action_steps": self.args.n_action_steps,
            "n_obs_steps": self.args.n_obs_steps,
            "num_inference_steps": 10,
            "obs_as_global_cond": True,
            "crop_shape": self.args.image_crop_size[::-1],  # (height, width)
            "diffusion_step_embed_dim": 128,
            "down_dims": [512, 1024, 2048],
            "kernel_size": 5,
            "n_groups": 8,
            "pointcloud_encoder_cfg": pointcloud_encoder_conf,
            "use_pc_color": self.args.use_pc_color,
            "encoder_output_dim": self.args.encoder_output_dim,
        }
        self.model_meta_info["policy"]["noise_scheduler_args"] = {
            "beta_end": 0.02,
            "beta_schedule": "squaredcos_cap_v2",
            "beta_start": 0.0001,
            "clip_sample": True,
            "num_train_timesteps": 100,
            "set_alpha_to_one": True,
            "prediction_type": "sample",
            "steps_offset": 0,
        }

        # Construct policy
        noise_scheduler = DDIMScheduler(
            **self.model_meta_info["policy"]["noise_scheduler_args"]
        )
        self.policy = DP3(
            noise_scheduler=noise_scheduler,
            **self.model_meta_info["policy"]["args"],
        )

        # Construct exponential moving average (EMA)
        if self.args.use_ema:
            self.ema_policy = copy.deepcopy(self.policy)
            self.ema = EMAModel(
                model=self.ema_policy,
                update_after_step=0,
                inv_gamma=1.0,
                power=0.75,
                min_value=0.0,
                max_value=0.9999,
            )

        # Construct optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.95, 0.999),
            eps=1e-8,
        )
        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=(len(self.train_dataloader) * self.args.num_epochs),
        )

        # Transfer to device
        self.policy.cuda()
        if self.args.use_ema:
            self.ema_policy.cuda()
        optimizer_to(self.optimizer, "cuda")

        # Print policy information
        self.print_policy_info()
        print(f"  - use ema: {self.args.use_ema}")
        print(
            f"  - horizon: {self.args.horizon}, obs steps: {self.args.n_obs_steps}, action steps: {self.args.n_action_steps}"
        )
        print(
            f"  - image size: {self.args.image_size}, image crop size: {self.args.image_crop_size}"
        )

    def train_loop(self):
        for epoch in tqdm(range(self.args.num_epochs)):
            # Run train step
            batch_result_list = []
            for data in self.train_dataloader:
                loss, _ = self.policy.compute_loss(dict_apply(data, lambda x: x.cuda()))
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
                if self.args.use_ema:
                    self.ema.step(self.policy)
                batch_result_list.append(
                    self.detach_batch_result(
                        {"loss": loss, "lr": self.lr_scheduler.get_last_lr()[0]}
                    )
                )
            self.log_epoch_summary(batch_result_list, "train", epoch)

            # Run validation step
            if self.args.use_ema:
                policy = self.ema_policy
            else:
                policy = self.policy
            policy.eval()
            with torch.inference_mode():
                batch_result_list = []
                for data in self.val_dataloader:
                    loss, _ = policy.compute_loss(dict_apply(data, lambda x: x.cuda()))
                    batch_result_list.append(self.detach_batch_result({"loss": loss}))
                epoch_summary = self.log_epoch_summary(batch_result_list, "val", epoch)

                # Update best checkpoint
                self.update_best_ckpt(epoch_summary, policy=policy)
            policy.train()

            # Save current checkpoint
            if epoch % max(self.args.num_epochs // 10, 1) == 0:
                self.save_current_ckpt(f"epoch{epoch:0>4}", policy=policy)

        # Save last checkpoint
        self.save_current_ckpt("last", policy=policy)

        # Save best checkpoint
        self.save_best_ckpt()
