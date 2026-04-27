#!/usr/bin/env python3
# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Main training script for VIPlanner.
Modular implementation with separated components.
"""

import torch

from viplanner.configs import DataConfig, TrainConfig
from viplanner.trainers import ViPlannerTrainer

torch.set_default_dtype(torch.float32)


def main():
    # Example configuration
    env_list = [
        "2azQ1b91cZZ",
        "JeFG25nYj2p",
        "Vvot9Ly1tCj",
        "town01",
        "ur6pFq6Qu1A",
        "B6ByNegPMKs",
        "8WUmhLawc2A",
        "2n8kARJN3HM",
    ]
    
    # Create configuration
    cfg = TrainConfig(
        sem=True,
        cost_map_name="cost_map_sem",
        env_list=env_list,
        test_env_id=7,
        file_name="modular_training",
        data_cfg=DataConfig(
            max_goal_distance=10.0,
        ),
        n_visualize=128,
        wb_project="viplanner",
    )
    
    # Create trainer
    trainer = ViPlannerTrainer(cfg)
    trainer.setup()
    
    # Note: Data loading needs to be implemented
    # trainer.train(train_loader, val_loader)
    # trainer.test(test_loader)
    # trainer.save_config()
    
    print("[INFO] Training script ready. Data loading needs to be implemented.")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
