# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Modular trainer for VIPlanner.
Separated from the original monolithic implementation.
"""

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision.transforms as transforms
import tqdm

from viplanner.losses import TrajectoryLoss
from viplanner.models import AutoEncoder, DualAutoEncoder
from viplanner.traj_cost import TrajCost, TrajOpt
from viplanner.utils.logging import WandbLogger
from viplanner.utils.optimizer import EarlyStopScheduler


class ViPlannerTrainer:
    """
    Modular trainer for VIPlanner model.
    """

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        
        # Setup paths
        os.makedirs(self._cfg.curr_model_dir, exist_ok=True)
        self.model_path = os.path.join(self._cfg.curr_model_dir, "model.pt")

        # Image transforms
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((self._cfg.img_input_size), antialias=True),
            ]
        )

        # Initialize components
        self.net: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[EarlyStopScheduler] = None
        self.loss_fn: Optional[TrajectoryLoss] = None
        self.logger: Optional[WandbLogger] = None
        self.traj_cost: Optional[TrajCost] = None   # set via set_traj_cost()
        self._traj_opt = TrajOpt()
        
        self.best_loss = float("inf")
        self.test_loss = float("inf")

        print("[INFO] Trainer initialized")

    def set_traj_cost(self, traj_cost: TrajCost) -> None:
        """Inject a pre-configured TrajCost (with cost map loaded)."""
        self.traj_cost = traj_cost

    def setup(self):
        """Setup model, optimizer, loss, and logger"""
        self._setup_model()
        self._setup_optimizer()
        self._setup_loss()
        self._setup_logger()

    def _setup_model(self):
        """Initialize model"""
        if self._cfg.sem or self._cfg.rgb:
            self.net = DualAutoEncoder(
                in_channel=self._cfg.in_channel,
                knodes=self._cfg.knodes,
                decoder_small=self._cfg.decoder_small,
                use_rgb=self._cfg.rgb,
                pre_train_sem=self._cfg.pre_train_sem,
                pre_train_cfg=self._cfg.pre_train_cfg,
                pre_train_weights=self._cfg.pre_train_weights,
                pre_train_freeze=self._cfg.pre_train_freeze,
            )
        else:
            self.net = AutoEncoder(self._cfg.in_channel, self._cfg.knodes)

        assert torch.cuda.is_available(), "GPU required"
        self.net = self.net.cuda(self._cfg.gpu_id)
        
        if self._cfg.resume:
            self._load_checkpoint()

    def _setup_optimizer(self):
        """Initialize optimizer and scheduler"""
        if self._cfg.optimizer == "adam":
            self.optimizer = optim.Adam(
                self.net.parameters(),
                lr=self._cfg.lr,
                weight_decay=self._cfg.w_decay,
            )
        elif self._cfg.optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.net.parameters(),
                lr=self._cfg.lr,
                momentum=self._cfg.momentum,
                weight_decay=self._cfg.w_decay,
            )
        else:
            raise ValueError(f"Optimizer {self._cfg.optimizer} not supported")
        
        self.scheduler = EarlyStopScheduler(
            self.optimizer,
            factor=self._cfg.factor,
            verbose=True,
            min_lr=self._cfg.min_lr,
            patience=self._cfg.patience,
        )

    def _setup_loss(self):
        """Initialize loss function"""
        self.loss_fn = TrajectoryLoss(
            w_obs=self._cfg.w_obs,
            w_height=self._cfg.w_height,
            w_motion=self._cfg.w_motion,
            w_goal=self._cfg.w_goal,
            obstacle_thread=self._cfg.obstacle_thread,
            fear_ahead_dist=self._cfg.fear_ahead_dist,
        )

    def _setup_logger(self):
        """Initialize logger"""
        self.logger = WandbLogger(
            project=self._cfg.wb_project,
            entity=self._cfg.wb_entity,
            name=self._cfg.get_model_save(),
            config=vars(self._cfg),
            log_dir=self._cfg.log_dir,
        )

    def _load_checkpoint(self):
        """Load model checkpoint"""
        if os.path.exists(self.model_path):
            model_state_dict, self.best_loss = torch.load(self.model_path)
            self.net.load_state_dict(model_state_dict)
            print(f"[INFO] Resumed from {self.model_path} with loss {self.best_loss}")

    def train_epoch(self, train_loader: Data.DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.net.train()
        train_loss = 0.0
        batches = len(train_loader)

        pbar = tqdm.tqdm(
            enumerate(train_loader),
            total=batches,
            desc=f"Epoch {epoch:3d} [train]",
            leave=False,
            dynamic_ncols=True,
        )
        for batch_idx, inputs in pbar:
            odom = inputs[2].cuda(self._cfg.gpu_id)
            goal = inputs[3].cuda(self._cfg.gpu_id)
            self.optimizer.zero_grad()

            if self._cfg.sem or self._cfg.rgb:
                depth_image = inputs[0].cuda(self._cfg.gpu_id)
                sem_rgb_image = inputs[1].cuda(self._cfg.gpu_id)
                preds, fear = self.net(depth_image, sem_rgb_image, goal)
            else:
                image = inputs[0].cuda(self._cfg.gpu_id)
                preds, fear = self.net(image, goal)

            # Flip y axis for augmented samples
            preds_flip = torch.clone(preds)
            preds_flip[inputs[4], :, 1] = preds_flip[inputs[4], :, 1] * -1
            goal_flip = torch.clone(goal)
            goal_flip[inputs[4], 1] = goal_flip[inputs[4], 1] * -1

            # Generate waypoints and compute loss
            waypoints = self._traj_opt.TrajGeneratorFromPFreeRot(preds_flip, step=0.1)
            if self.traj_cost is not None:
                loss = self.traj_cost.cost_of_traj(
                    waypoints, odom, goal_flip, fear,
                    ahead_dist=self._cfg.fear_ahead_dist,
                )
            else:
                loss = torch.tensor(0.0, requires_grad=True, device=waypoints.device)

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

            avg = train_loss / (batch_idx + 1)
            pbar.set_postfix(loss=f"{avg:.4f}", refresh=False)

        return train_loss / batches

    def validate_epoch(self, val_loader: Data.DataLoader, epoch: int) -> float:
        """Validate for one epoch"""
        self.net.eval()
        val_loss = 0.0

        pbar = tqdm.tqdm(
            enumerate(val_loader),
            total=len(val_loader),
            desc=f"Epoch {epoch:3d} [ val ]",
            leave=False,
            dynamic_ncols=True,
        )
        with torch.no_grad():
            for batch_idx, inputs in pbar:
                odom = inputs[2].cuda(self._cfg.gpu_id)
                goal = inputs[3].cuda(self._cfg.gpu_id)

                if self._cfg.sem or self._cfg.rgb:
                    depth_image = inputs[0].cuda(self._cfg.gpu_id)
                    sem_rgb_image = inputs[1].cuda(self._cfg.gpu_id)
                    preds, fear = self.net(depth_image, sem_rgb_image, goal)
                else:
                    image = inputs[0].cuda(self._cfg.gpu_id)
                    preds, fear = self.net(image, goal)

                # flip augmented
                preds[inputs[4], :, 1] *= -1
                goal[inputs[4], 1] *= -1

                waypoints = self._traj_opt.TrajGeneratorFromPFreeRot(preds, step=0.1)
                if self.traj_cost is not None:
                    loss = self.traj_cost.cost_of_traj(
                        waypoints, odom, goal, fear,
                        ahead_dist=self._cfg.fear_ahead_dist,
                    )
                else:
                    loss = torch.tensor(0.0, device=waypoints.device)
                val_loss += loss.item()

                avg = val_loss / (batch_idx + 1)
                pbar.set_postfix(loss=f"{avg:.4f}", refresh=False)

        return val_loss / len(val_loader)

    def train(self, train_loader, val_loader):
        """Main training loop"""
        epoch_pbar = tqdm.tqdm(
            range(self._cfg.epochs),
            desc="Training",
            dynamic_ncols=True,
        )
        for epoch in epoch_pbar:
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate_epoch(val_loader, epoch)

            epoch_pbar.set_postfix(
                train=f"{train_loss:.4f}",
                val=f"{val_loss:.4f}",
                best=f"{self.best_loss:.4f}",
            )

            self.logger.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch,
            })

            if val_loss < self.best_loss:
                tqdm.tqdm.write(f"[epoch {epoch:3d}] val loss improved "
                                f"{self.best_loss:.4f} → {val_loss:.4f}, saving model")
                torch.save((self.net.state_dict(), val_loss), self.model_path)
                self.best_loss = val_loss

            if self.scheduler.step(val_loss):
                tqdm.tqdm.write("[INFO] Early stopping!")
                break

        torch.cuda.empty_cache()

    def test(self, test_loader):
        """Test the model"""
        self.net.eval()
        self.test_loss = self.validate_epoch(test_loader, epoch=0)
        print(f"[INFO] Test loss: {self.test_loss:.4f}")

    def save_config(self):
        """Save configuration"""
        import yaml
        path, _ = os.path.splitext(self.model_path)
        yaml_path = path + ".yaml"
        
        save_dict = {
            "config": vars(self._cfg),
            "loss": {"val_loss": self.best_loss, "test_loss": self.test_loss}
        }
        
        with open(yaml_path, "w+") as file:
            yaml.dump(save_dict, file, allow_unicode=True, default_flow_style=False)
        
        self.logger.finish()
