import os

import numpy as np
import torch
from eipl.utils import LossScheduler
from torchvision.transforms import v2
from tqdm import tqdm

from robo_manip_baselines.common import DataKey, TrainBase
from robo_manip_baselines.sarnn import RmbSarnnDataset, SarnnPolicy


class TrainSarnn(TrainBase):
    policy_name = "SARNN"
    policy_dir = os.path.join(os.path.dirname(__file__), "..")
    DatasetClass = RmbSarnnDataset

    def __init__(self):
        super().__init__()

        self.setup_image_transforms()

    def setup_args(self):
        super().setup_args()

        # Check action keys
        if len(self.args.action_keys) > 0:
            raise ValueError(
                f"[{self.__class__.__name__}] action_keys must be empty: {self.args.action_keys}"
            )

        # Set image size list
        def refine_size_list(size_list):
            if len(size_list) == 2:
                return [tuple(size_list)] * len(self.args.camera_names)
            else:
                assert len(size_list) == len(self.args.camera_names) * 2
                return [
                    (size_list[i], size_list[i + 1])
                    for i in range(0, len(size_list), 2)
                ]

        self.args.image_crop_size_list = refine_size_list(
            self.args.image_crop_size_list
        )
        self.args.image_size_list = refine_size_list(self.args.image_size_list)

    def set_additional_args(self, parser):
        for action in parser._actions:
            if action.dest == "state_keys":
                action.choices = DataKey.MEASURED_DATA_KEYS + DataKey.COMMAND_DATA_KEYS
            elif action.dest == "action_keys":
                action.choices = []

        parser.set_defaults(state_keys=[DataKey.COMMAND_JOINT_POS])
        parser.set_defaults(action_keys=[])

        parser.set_defaults(state_aug_std=0.2)
        parser.set_defaults(image_aug_std=0.02)
        parser.set_defaults(image_aug_color_scale=1.0)
        parser.add_argument(
            "--image_aug_erasing_scale",
            type=float,
            default=1.0,
            help="Scale of random erasing applied to the images",
        )

        parser.set_defaults(skip=6)

        parser.set_defaults(batch_size=16)
        parser.set_defaults(num_epochs=8000)
        parser.set_defaults(lr=1e-4)

        parser.add_argument(
            "--image_loss_scale",
            type=float,
            default=0.1,
            help="Scale of image loss",
        )
        parser.add_argument(
            "--attention_loss_scale",
            type=float,
            default=0.1,
            help="Scale of attention loss",
        )

        parser.add_argument(
            "--image_crop_size_list",
            type=int,
            nargs="+",
            default=[280, 280],
            help="List of image crop size before resize. Specify a 2-dimensional array if all images have the same size, or an array of <number-of-images> * 2 dimensions if the size differs for each individual image.",
        )
        parser.add_argument(
            "--image_size_list",
            type=int,
            nargs="+",
            default=[64, 64],
            help="List of image size. Specify a 2-dimensional array if all images have the same size, or an array of <number-of-images> * 2 dimensions if the size differs for each individual image.",
        )
        parser.add_argument(
            "--num_attentions",
            type=int,
            default=5,
            help="Number of spacial attention points",
        )
        parser.add_argument(
            "--lstm_hidden_dim",
            type=int,
            default=50,
            help="Dimension of hidden state of LSTM",
        )

    def setup_model_meta_info(self):
        super().setup_model_meta_info()

        self.model_meta_info["image"]["aug_erasing_scale"] = (
            self.args.image_aug_erasing_scale
        )
        self.model_meta_info["data"]["image_crop_size_list"] = (
            self.args.image_crop_size_list
        )
        self.model_meta_info["data"]["image_size_list"] = self.args.image_size_list

    def setup_policy(self):
        # Set policy args
        self.model_meta_info["policy"]["args"] = {
            "image_size_list": self.args.image_size_list,
            "num_attentions": self.args.num_attentions,
            "lstm_hidden_dim": self.args.lstm_hidden_dim,
        }

        # Construct policy
        self.policy = SarnnPolicy(
            len(self.model_meta_info["state"]["example"]),
            len(self.args.camera_names),
            **self.model_meta_info["policy"]["args"],
        )
        self.policy.cuda()

        # Construct optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.args.lr, eps=1e-7
        )

        # Print policy information
        self.print_policy_info()
        print(f"  - image crop size list: {self.args.image_crop_size_list}")
        print(f"  - image size list: {self.args.image_size_list}")
        print(f"  - num attentions: {self.args.num_attentions}")

    def setup_image_transforms(self):
        image_transform_list = []

        if self.model_meta_info["image"]["aug_erasing_scale"] > 0.0:
            scale = self.model_meta_info["image"]["aug_erasing_scale"]
            image_transform_list.append(v2.RandomErasing(p=0.5 * scale))

        if self.model_meta_info["image"]["aug_color_scale"] > 0.0:
            scale = self.model_meta_info["image"]["aug_color_scale"]
            image_transform_list.append(
                v2.ColorJitter(
                    brightness=0.4 * scale,
                    contrast=0.4 * scale,
                    saturation=0.4 * scale,
                    hue=0.05 * scale,
                )
            )

        if self.model_meta_info["image"]["aug_affine_scale"] > 0.0:
            scale = self.model_meta_info["image"]["aug_affine_scale"]
            image_transform_list.append(
                v2.RandomAffine(
                    degrees=4.0 * scale,
                    translate=(0.05 * scale, 0.05 * scale),
                    scale=(1.0 - 0.1 * scale, 1.0 + 0.1 * scale),
                )
            )

        if self.model_meta_info["image"]["aug_std"] > 0.0:
            image_transform_list.append(
                v2.GaussianNoise(sigma=self.model_meta_info["image"]["aug_std"])
            )

        if len(image_transform_list) == 0:
            self.image_transforms = None
        else:
            self.image_transforms = v2.Compose(image_transform_list)

            print(
                f"[{self.__class__.__name__}] Augment the target image for reconstruction."
            )
            image_transform_str_list = [
                f"{image_transform.__class__.__name__}"
                for image_transform in self.image_transforms.transforms
            ]
            print(f"  - image transforms: {image_transform_str_list}")

    def train_loop(self):
        self.attention_loss_scheduler = LossScheduler(decay_end=1000, curve_name="s")

        best_ckpt_info = {"loss": np.inf}
        for epoch in tqdm(range(self.args.num_epochs)):
            # Run train step
            self.policy.train()
            batch_result_list = []
            for data in self.train_dataloader:
                self.optimizer.zero_grad()
                loss = self.calc_loss(*data)
                loss.backward()
                self.optimizer.step()
                batch_result_list.append(self.detach_batch_result({"loss": loss}))
            self.log_epoch_summary(batch_result_list, "train", epoch)

            # Run validation step
            with torch.inference_mode():
                self.policy.eval()
                batch_result_list = []
                for data in self.val_dataloader:
                    loss = self.calc_loss(*data)
                    batch_result_list.append(self.detach_batch_result({"loss": loss}))
                epoch_summary = self.log_epoch_summary(batch_result_list, "val", epoch)

                # Update best checkpoint
                best_ckpt_info = self.update_best_ckpt(best_ckpt_info, epoch_summary)

            # Save current checkpoint
            if epoch % max(self.args.num_epochs // 10, 1) == 0:
                self.save_current_ckpt(f"epoch{epoch:0>4}")

        # Save last checkpoint
        self.save_current_ckpt("last")

        # Save best checkpoint
        self.save_best_ckpt(best_ckpt_info)

    def calc_loss(
        self,
        state_seq,  # (batch_size, episode_len, state_dim)
        image_seq_list,  # (num_images, batch_size, episode_len, 3, width, height)
        mask_seq,  # (batch_size, episode_len)
    ):
        state_seq = state_seq.cuda()
        image_seq_list = [image_seq.cuda() for image_seq in image_seq_list]
        mask_seq = mask_seq.cuda()

        # Augment data
        aug_image_seq_list = []
        for image_seq in image_seq_list:
            if self.image_transforms is None:
                aug_image_seq = image_seq
            else:
                aug_image_seq = self.image_transforms(image_seq)
            aug_image_seq_list.append(aug_image_seq)
        aug_state_seq = state_seq + self.model_meta_info["state"][
            "aug_std"
        ] * torch.randn_like(state_seq)

        # Forward policy along the time sequence
        num_images = len(image_seq_list)
        lstm_state = None
        predicted_state_seq = []
        predicted_image_seq_list = [[] * num_images]
        attention_seq_list = [[] * num_images]
        predicted_attention_seq_list = [[] * num_images]
        for time_idx in range(len(state_seq[0]) - 1):
            (
                predicted_state,
                predicted_image_list,
                attention_list,
                predicted_attention_list,
                lstm_state,
            ) = self.policy(
                aug_state_seq[:, time_idx],
                [aug_image_seq[:, time_idx] for aug_image_seq in aug_image_seq_list],
                lstm_state,
            )
            predicted_state_seq.append(predicted_state)
            for image_idx in range(num_images):
                predicted_image_seq_list[image_idx].append(
                    predicted_image_list[image_idx]
                )
                attention_seq_list[image_idx].append(attention_list[image_idx])
                predicted_attention_seq_list[image_idx].append(
                    predicted_attention_list[image_idx]
                )

        # Permute the dimensions so that the batch size is at the top
        predicted_state_seq = torch.permute(
            torch.stack(predicted_state_seq), (1, 0, 2)
        )  # (batch_size, episode_len, state_dim)
        for image_idx in range(num_images):
            predicted_image_seq_list[image_idx] = torch.permute(
                torch.stack(predicted_image_seq_list[image_idx]),
                (1, 0, 2, 3, 4),
            )  # (batch_size, episode_len, 3, width, height)
            attention_seq_list[image_idx] = torch.permute(
                torch.stack(attention_seq_list[image_idx]),
                (1, 0, 2, 3),
            )  # (batch_size, episode_len, num_attentions, 2)
            predicted_attention_seq_list[image_idx] = torch.permute(
                torch.stack(predicted_attention_seq_list[image_idx]),
                (1, 0, 2, 3),
            )  # (batch_size, episode_len, num_attentions, 2)

        # Calculate loss
        criterion = torch.nn.MSELoss(reduction="none")

        state_loss = torch.mean(criterion(predicted_state_seq, state_seq[:, 1:]), dim=2)
        state_loss = torch.sum(state_loss * mask_seq[:, 1:]) / torch.sum(
            mask_seq[:, 1:]
        )

        image_loss_list = []
        attention_loss_list = []
        for image_idx in range(num_images):
            image_loss = torch.mean(
                criterion(
                    predicted_image_seq_list[image_idx],
                    image_seq_list[image_idx][:, 1:],
                ),
                dim=(2, 3, 4),
            )
            image_loss = torch.sum(image_loss * mask_seq[:, 1:]) / torch.sum(
                mask_seq[:, 1:]
            )
            image_loss_list.append(image_loss)

            attention_loss = torch.mean(
                criterion(
                    predicted_attention_seq_list[image_idx][:, :-1],
                    attention_seq_list[image_idx][:, 1:],
                ),
                dim=(2, 3),
            )
            attention_loss = torch.sum(attention_loss * mask_seq[:, 1:-1]) / torch.sum(
                mask_seq[:, 1:-1]
            )
            attention_loss_list.append(attention_loss)

        loss = (
            state_loss
            + self.args.image_loss_scale * torch.sum(torch.stack(image_loss_list))
            + self.attention_loss_scheduler(self.args.attention_loss_scale)
            * torch.sum(torch.stack(attention_loss_list))
        )

        return loss


if __name__ == "__main__":
    train = TrainSarnn()
    train.run()
    train.close()
