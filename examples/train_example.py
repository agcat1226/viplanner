#!/usr/bin/env python3
# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example training script showing how to use the modular VIPlanner framework.
"""

import torch

from viplanner.configs import DataConfig, TrainConfig
from viplanner.trainers import ViPlannerTrainer


def train_with_custom_config():
    """Example: Training with custom configuration"""
    
    # Define data configuration
    data_cfg = DataConfig(
        max_depth=15.0,
        max_goal_distance=10.0,
        min_goal_distance=0.5,
        distance_scheme={1: 0.2, 3: 0.35, 5: 0.25, 7.5: 0.15, 10: 0.05},
        ratio=0.9,  # 90% train, 10% val
        # Optional augmentation
        depth_salt_pepper=0.01,
        depth_gaussian=0.001,
    )
    
    # Define training configuration
    train_cfg = TrainConfig(
        # Model settings
        sem=True,
        rgb=False,
        in_channel=16,
        knodes=5,
        decoder_small=False,
        img_input_size=(360, 640),
        
        # Training settings
        epochs=100,
        batch_size=64,
        lr=2e-3,
        optimizer="sgd",
        momentum=0.1,
        w_decay=1e-4,
        
        # Loss weights
        w_obs=0.25,
        w_height=1.0,
        w_motion=1.5,
        w_goal=4.0,
        
        # Data
        env_list=["2azQ1b91cZZ", "JeFG25nYj2p", "Vvot9Ly1tCj"],
        test_env_id=2,
        data_cfg=data_cfg,
        cost_map_name="cost_map_sem",
        
        # System
        gpu_id=0,
        num_workers=4,
        
        # Logging
        wb_project="viplanner",
        wb_entity="your_entity",
        file_name="custom_experiment",
    )
    
    # Create trainer
    trainer = ViPlannerTrainer(train_cfg)
    trainer.setup()
    
    print("[INFO] Trainer configured successfully")
    print(f"[INFO] Model will be saved to: {train_cfg.curr_model_dir}")
    
    # TODO: Load data and start training
    # train_loader, val_loader = load_data(train_cfg)
    # trainer.train(train_loader, val_loader)
    # test_loader = load_test_data(train_cfg)
    # trainer.test(test_loader)
    # trainer.save_config()


def train_minimal():
    """Example: Minimal training configuration"""
    
    cfg = TrainConfig(
        sem=True,
        epochs=50,
        batch_size=32,
        env_list=["2azQ1b91cZZ"],
        test_env_id=0,
    )
    
    trainer = ViPlannerTrainer(cfg)
    trainer.setup()
    
    print("[INFO] Minimal trainer ready")


def train_with_rgb():
    """Example: Training with RGB instead of semantics"""
    
    cfg = TrainConfig(
        sem=False,
        rgb=True,
        pre_train_sem=True,  # Use pre-trained RGB encoder
        pre_train_cfg="path/to/config.yaml",
        pre_train_weights="path/to/weights.pkl",
        epochs=100,
        env_list=["2azQ1b91cZZ", "JeFG25nYj2p"],
        test_env_id=1,
    )
    
    trainer = ViPlannerTrainer(cfg)
    trainer.setup()
    
    print("[INFO] RGB-based trainer ready")


if __name__ == "__main__":
    print("=" * 60)
    print("VIPlanner Training Examples")
    print("=" * 60)
    
    print("\n1. Custom Configuration")
    train_with_custom_config()
    
    print("\n2. Minimal Configuration")
    train_minimal()
    
    print("\n3. RGB-based Training")
    train_with_rgb()
    
    print("\n" + "=" * 60)
    print("Examples completed. Modify and run for actual training.")
    print("=" * 60)
