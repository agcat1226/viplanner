# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Carla dataset training configuration.
"""

from dataclasses import dataclass, field
from typing import Optional

from viplanner.configs.train_config import DataConfig, TrainConfig


@dataclass
class CarlaDataConfig(DataConfig):
    """Configuration for Carla dataset loading and preprocessing."""

    # Carla depth images are in mm (float32 npy), scale to meters
    max_depth: float = 15.0

    # Goal sampling distances (meters)
    max_goal_distance: float = 10.0
    min_goal_distance: float = 0.5

    # Number of goal samples per depth image frame
    pairs_per_image: int = 4

    # Train/val split ratio
    ratio: float = 0.9

    # Optional cap on total training pairs
    max_train_pairs: Optional[int] = None


@dataclass
class CarlaTrainConfig(TrainConfig):
    """Training configuration for the Carla dataset."""

    # Use semantic images (img_warp pre-warped semantics)
    sem: bool = True
    rgb: bool = False

    # Cost map name (must match maps/params/config_cost_map_sem.yaml)
    cost_map_name: str = "cost_map_sem"

    # Path to the Carla dataset root
    carla_data_path: str = "~/Carla/carla"

    # Training hyperparameters (tuned for Carla urban scenes)
    epochs: int = 100
    batch_size: int = 4
    lr: float = 2e-3
    optimizer: str = "sgd"

    # Loss weights
    w_obs: float = 0.25
    w_height: float = 1.0
    w_motion: float = 1.5
    w_goal: float = 4.0
    obstacle_thread: float = 1.2
    fear_ahead_dist: float = 2.5

    # Data config
    data_cfg: CarlaDataConfig = field(default_factory=CarlaDataConfig)

    # Logging
    file_name: str = "carla"
    wb_project: str = "viplanner"
