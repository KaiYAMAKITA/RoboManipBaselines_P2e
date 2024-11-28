#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import os
import sys
import shutil
import random
import torch
import numpy as np
import argparse
from pathlib import Path

from tqdm import tqdm
import torch.optim as optim
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from eipl.utils import EarlyStopping, check_args, set_logdir, normalization

class TrainSarnn(object):
    def __init__(self):
        self.setup_args()

        self.setup_dataset()

        self.setup_policy()

    def setup_args(self):
        parser = argparse.ArgumentParser(description="Train SARNN")

        parser.add_argument("--data_dir", type=str, required=True)
        parser.add_argument("--model", type=str, default="sarnn")
        parser.add_argument("--epoch", type=int, default=10000)
        parser.add_argument("--batch_size", type=int, default=5)
        parser.add_argument("--rec_dim", type=int, default=50)
        parser.add_argument("--k_dim", type=int, default=5)
        parser.add_argument("--random_seed", type=int, required=False)
        parser.add_argument("--front_img_loss", type=float, default=0.1)
        parser.add_argument("--side_img_loss", type=float, default=0.1)
        parser.add_argument("--joint_loss", type=float, default=1.0)
        parser.add_argument("--wrench_loss", type=float, default=1.0)
        parser.add_argument("--front_pt_loss", type=float, default=0.1)
        parser.add_argument("--side_pt_loss", type=float, default=0.1)
        parser.add_argument("--no_side_image", action="store_true")
        parser.add_argument("--no_wrench", action="store_true")
        parser.add_argument("--with_mask", action="store_true")
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--heatmap_size", type=float, default=0.1)
        parser.add_argument("--temperature", type=float, default=1e-4)
        parser.add_argument("--stdev", type=float, default=0.1)
        parser.add_argument("--log_dir", default="log/")
        parser.add_argument("--vmin", type=float, default=0.0)
        parser.add_argument("--vmax", type=float, default=1.0)
        parser.add_argument("--device", type=int, default=0)
        parser.add_argument("--compile", action="store_true")
        parser.add_argument("--tag", help="Tag name for snap/log sub directory")

        args = parser.parse_args()
        self.args = check_args(args)

    def setup_dataset(self):
        # fix seed
        if self.args.random_seed is not None:
            random.seed(self.args.random_seed)
            np.random.seed(self.args.random_seed)
            torch.manual_seed(self.args.random_seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        # calculate the noise level (variance) from the normalized range
        stdev = self.args.stdev * (self.args.vmax - self.args.vmin)

        # set device id
        if self.args.device >= 0:
            self.device = "cuda:{}".format(self.args.device)
        else:
            self.device = "cpu"

        # copy bound files
        data_dir = Path(self.args.data_dir)
        minmax = [self.args.vmin, self.args.vmax]
        self.log_dir_path = set_logdir("./" + self.args.log_dir, self.args.tag)
        bound_files = sorted(data_dir.glob("*_bounds.npy"))
        for bound_file in bound_files:
            shutil.copy(bound_file, os.path.join(self.log_dir_path, bound_file.name))

        # load train data files
        train_data_dir = data_dir / "train"
        joint_bounds = np.load(data_dir / "action_bounds.npy")
        joints_raw = np.load(sorted(train_data_dir.glob("**/actions.npy"))[0])
        joints = normalization(joints_raw, joint_bounds, minmax)
        if not self.args.no_wrench:
            wrench_bounds = np.load(data_dir / "wrench_bounds.npy")
            wrenches_raw = np.load(sorted(train_data_dir.glob("**/wrenches.npy"))[0])
            wrenches = normalization(wrenches_raw, wrench_bounds, minmax)
        front_images_raw = np.load(sorted(train_data_dir.glob("**/front_images.npy"))[0])
        front_images = normalization(front_images_raw.transpose(0, 1, 4, 2, 3), (0, 255), minmax)
        if not self.args.no_side_image:
            side_images_raw = np.load(sorted(train_data_dir.glob("**/side_images.npy"))[0])
            side_images = normalization(side_images_raw.transpose(0, 1, 4, 2, 3), (0, 255), minmax)
        masks = np.load(sorted(train_data_dir.glob("**/masks.npy"))[0])

        if (not self.args.no_side_image) and (not self.args.no_wrench):
            assert not self.args.with_mask, "with_mask option is not supported for the model with side_image and wrench."
            from multimodal_robot_model.sarnn import MultimodalDatasetWithSideimageAndWrench
            train_dataset = MultimodalDatasetWithSideimageAndWrench(
                front_images,
                side_images,
                joints,
                wrenches,
                device=self.device,
                stdev=stdev
            )
        elif self.args.no_side_image and self.args.no_wrench:
            if self.args.with_mask:
                from multimodal_robot_model.sarnn import MultimodalDatasetWithMask
                train_dataset = MultimodalDatasetWithMask(
                    front_images,
                    joints,
                    masks,
                    device=self.device,
                    stdev=stdev
                )
            else:
                from eipl.data import MultimodalDataset
                train_dataset = MultimodalDataset(
                    front_images,
                    joints,
                    device=self.device,
                    stdev=stdev
                )
        else:
            raise AssertionError(f"Not asserted (no_side_image, no_wrench): {(self.args.no_side_image, self.args.no_wrench)}")
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=False,
        )

        # load test data files
        test_data_dir = data_dir / "test"
        joints_raw = np.load(sorted(test_data_dir.glob("**/actions.npy"))[0])
        joints = normalization(joints_raw, joint_bounds, minmax)
        if not self.args.no_wrench:
            wrenches_raw = np.load(sorted(test_data_dir.glob("**/wrenches.npy"))[0])
            wrenches = normalization(wrenches_raw, wrench_bounds, minmax)
        front_images_raw = np.load(sorted(test_data_dir.glob("**/front_images.npy"))[0])
        front_images = normalization(front_images_raw.transpose(0, 1, 4, 2, 3), (0, 255), minmax)
        if not self.args.no_side_image:
            side_images_raw = np.load(sorted(test_data_dir.glob("**/side_images.npy"))[0])
            side_images = normalization(side_images_raw.transpose(0, 1, 4, 2, 3), (0, 255), minmax)
        masks = np.load(sorted(test_data_dir.glob("**/masks.npy"))[0])

        if (not self.args.no_side_image) and (not self.args.no_wrench):
            test_dataset = MultimodalDatasetWithSideimageAndWrench(
                front_images,
                side_images,
                joints,
                wrenches,
                device=self.device,
                stdev=None
            )
        elif self.args.no_side_image and self.args.no_wrench:
            if self.args.with_mask:
                from multimodal_robot_model.sarnn import MultimodalDatasetWithMask
                test_dataset = MultimodalDatasetWithMask(
                    front_images,
                    joints,
                    masks,
                    device=self.device,
                    stdev=stdev
                )
            else:
                from eipl.data import MultimodalDataset
                test_dataset = MultimodalDataset(
                    front_images,
                    joints,
                    device=self.device,
                    stdev=None
                )
        else:
            raise AssertionError(f"Not asserted (no_side_image, no_wrench): {(self.args.no_side_image, self.args.no_wrench)}")
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=False,
        )

    def setup_policy(self):
        joint_dim = self.train_loader.dataset[0][0][1].shape[-1]

        # define model
        if (not self.args.no_side_image) and (not self.args.no_wrench):
            from multimodal_robot_model.sarnn import SARNNwithSideimageAndWrench
            self.model = SARNNwithSideimageAndWrench(
                rec_dim=self.args.rec_dim,
                joint_dim=joint_dim,
                wrench_dim=6,
                k_dim=self.args.k_dim,
                heatmap_size=self.args.heatmap_size,
                temperature=self.args.temperature,
                im_size=[64, 64],
            )
        elif self.args.no_side_image and self.args.no_wrench:
            from eipl.model import SARNN
            self.model = SARNN(
                rec_dim=self.args.rec_dim,
                joint_dim=joint_dim,
                k_dim=self.args.k_dim,
                heatmap_size=self.args.heatmap_size,
                temperature=self.args.temperature,
                im_size=[64, 64],
            )
        else:
            raise AssertionError(f"Not asserted (no_side_image, no_wrench): {(self.args.no_side_image, self.args.no_wrench)}")

        # torch.compile makes PyTorch code run faster
        if self.args.compile:
            torch.set_float32_matmul_precision("high")
            self.model = torch.compile(self.model)

        # set optimizer
        self.optimizer = optim.Adam(self.model.parameters(), eps=1e-07, lr=self.args.lr)

        # load trainer/tester class
        if (not self.args.no_side_image) and (not self.args.no_wrench):
            from multimodal_robot_model.sarnn import fullBPTTtrainerWithSideimageAndWrench, Loss
            loss_weights = [
                {
                    Loss.FRONT_IMG: self.args.front_img_loss,
                    Loss.SIDE_IMG: self.args.side_img_loss,
                    Loss.JOINT: self.args.joint_loss,
                    Loss.WRENCH: self.args.wrench_loss,
                    Loss.FRONT_PT: self.args.front_pt_loss,
                    Loss.SIDE_PT: self.args.side_pt_loss
                }[loss] for loss in Loss
            ]
            self.trainer = fullBPTTtrainerWithSideimageAndWrench(
                self.model, self.optimizer, loss_weights=loss_weights, device=self.device
            )
        elif self.args.no_side_image and self.args.no_wrench:
            loss_weights = [self.args.front_img_loss, self.args.joint_loss, self.args.front_pt_loss]
            if self.args.with_mask:
                from multimodal_robot_model.sarnn import fullBPTTtrainerWithMask
                self.trainer = fullBPTTtrainerWithMask(self.model, self.optimizer, loss_weights=loss_weights, device=self.device)
            else:
                from eipl.tutorials.airec.sarnn.libs.fullBPTT import fullBPTTtrainer
                self.trainer = fullBPTTtrainer(self.model, self.optimizer, loss_weights=loss_weights, device=self.device)
        else:
            raise AssertionError(f"Not asserted (no_side_image, no_wrench): {(self.args.no_side_image, self.args.no_wrench)}")

    def run(self):
        save_name = os.path.join(self.log_dir_path, "SARNN.pth")
        writer = SummaryWriter(log_dir=self.log_dir_path, flush_secs=30)
        early_stop = EarlyStopping(patience=1000)

        with tqdm(range(self.args.epoch)) as pbar_epoch:
            for epoch in pbar_epoch:
                # train and test
                train_loss = self.trainer.process_epoch(self.train_loader)
                with torch.no_grad():
                    test_loss = self.trainer.process_epoch(self.test_loader, training=False)
                writer.add_scalar("Loss/train_loss", train_loss, epoch)
                writer.add_scalar("Loss/test_loss", test_loss, epoch)

                # early stop
                save_ckpt, _ = early_stop(test_loss)

                if save_ckpt:
                    self.trainer.save(epoch, [train_loss, test_loss], save_name)

                # print process bar
                pbar_epoch.set_postfix(
                    OrderedDict(train_loss=train_loss, test_loss=test_loss)
                )
                pbar_epoch.update()

if __name__ == "__main__":
    train = TrainSarnn()
    train.run()
