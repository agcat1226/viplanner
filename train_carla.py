#!/usr/bin/env python3
# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Training script for VIPlanner on the Carla dataset.

Usage:
    python train_carla.py

Dataset expected at ~/Carla/carla/ with structure:
    depth/          - depth images (*.npy)
    semantics/      - semantic RGB images
    img_warp/       - pre-warped semantic images (aligned to depth frame)
    camera_extrinsic.txt
    intrinsics.txt
    maps/
"""

import os

import torch
import torch.utils.data as Data
import torchvision.transforms as transforms

from viplanner.configs import CarlaTrainConfig
from viplanner.datasets import CarlaDataGenerator
from viplanner.datasets.planner_dataset import PlannerData
from viplanner.traj_cost import TrajCost
from viplanner.trainers import ViPlannerTrainer

torch.set_default_dtype(torch.float32)


def build_dataloaders(cfg: CarlaTrainConfig):
    """Build train and validation DataLoaders from the Carla dataset."""

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(cfg.img_input_size, antialias=True),
    ])

    data_cfg = cfg.data_cfg

    print(f"[INFO] Loading Carla dataset from: {cfg.carla_data_path}")
    generator = CarlaDataGenerator(
        root=cfg.carla_data_path,
        semantics=cfg.sem,
        rgb=cfg.rgb,
        max_depth=data_cfg.max_depth,
        max_goal_distance=data_cfg.max_goal_distance,
        min_goal_distance=data_cfg.min_goal_distance,
        pairs_per_image=data_cfg.pairs_per_image,
        ratio=data_cfg.ratio,
        max_train_pairs=data_cfg.max_train_pairs,
    )

    train_data = PlannerData(
        transform=transform,
        semantics=cfg.sem,
        rgb=cfg.rgb,
        max_depth=data_cfg.max_depth,
    )
    val_data = PlannerData(
        transform=transform,
        semantics=cfg.sem,
        rgb=cfg.rgb,
        max_depth=data_cfg.max_depth,
    )

    generator.split_samples(train_data, val_data)

    if cfg.load_in_ram:
        print("[INFO] Loading data into RAM...")
        train_data.load_data_in_memory()
        val_data.load_data_in_memory()

    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )
    val_loader = Data.DataLoader(
        dataset=val_data,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )

    return train_loader, val_loader, generator.alpha_fov


def main():
    cfg = CarlaTrainConfig()

    # Override data path from environment variable if set
    if os.getenv("CARLA_DATA_PATH"):
        cfg.carla_data_path = os.getenv("CARLA_DATA_PATH")

    print(f"[INFO] Config: sem={cfg.sem}, epochs={cfg.epochs}, "
          f"batch_size={cfg.batch_size}, lr={cfg.lr}")

    # Build data loaders
    train_loader, val_loader, fov_angle = build_dataloaders(cfg)
    print(f"[INFO] Train batches: {len(train_loader)}, "
          f"Val batches: {len(val_loader)}, FOV: {fov_angle:.3f} rad")

    # Setup trainer
    trainer = ViPlannerTrainer(cfg)
    trainer.setup()

    # Load cost map and inject TrajCost
    print(f"[INFO] Loading cost map: {cfg.cost_map_name}")
    traj_cost = TrajCost(
        gpu_id=cfg.gpu_id,
        w_obs=cfg.w_obs,
        w_height=cfg.w_height,
        w_motion=cfg.w_motion,
        w_goal=cfg.w_goal,
        obstacle_thread=cfg.obstacle_thread,
    )
    traj_cost.set_map(cfg.carla_data_path, cfg.cost_map_name)
    trainer.set_traj_cost(traj_cost)

    # Train
    trainer.train(train_loader, val_loader)

    # Validate final model
    trainer.test(val_loader)

    # Save config
    trainer.save_config()

    torch.cuda.empty_cache()
    print("[INFO] Training complete.")


if __name__ == "__main__":
    main()
