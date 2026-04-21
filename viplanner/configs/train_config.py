# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Training configuration classes.
Simplified and modularized from original implementation.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing"""
    
    # Data paths
    max_depth: float = 15.0
    max_goal_distance: float = 15.0
    min_goal_distance: float = 0.5
    
    # Sample distribution
    distance_scheme: dict = field(
        default_factory=lambda: {1: 0.2, 3: 0.35, 5: 0.25, 7.5: 0.15, 10: 0.05}
    )
    ratio_fov_samples: float = 1.0
    ratio_front_samples: float = 0.0
    ratio_back_samples: float = 0.0
    
    # Train/val split
    ratio: float = 0.9
    max_train_pairs: Optional[int] = None
    pairs_per_image: int = 4
    
    # Augmentation
    depth_salt_pepper: Optional[float] = None
    depth_gaussian: Optional[float] = None
    depth_random_polygons_nb: Optional[int] = None
    depth_random_polygon_size: int = 10
    
    sem_rgb_pepper: Optional[float] = None
    sem_rgb_black_img: Optional[float] = None
    sem_rgb_random_polygons_nb: Optional[int] = None
    sem_rgb_random_polygon_size: int = 20


@dataclass
class TrainConfig:
    """Configuration for model training"""
    
    # Model settings
    sem: bool = True
    rgb: bool = False
    in_channel: int = 16
    knodes: int = 5
    decoder_small: bool = False
    img_input_size: Tuple[int, int] = field(default_factory=lambda: (360, 640))
    
    # Pre-training
    pre_train_sem: bool = True
    pre_train_cfg: Optional[str] = None
    pre_train_weights: Optional[str] = None
    pre_train_freeze: bool = True
    
    # Training settings
    epochs: int = 100
    batch_size: int = 64
    lr: float = 2e-3
    optimizer: str = "sgd"
    momentum: float = 0.1
    w_decay: float = 1e-4
    
    # Scheduler
    factor: float = 0.5
    min_lr: float = 1e-5
    patience: int = 3
    
    # Loss weights
    w_obs: float = 0.25
    w_height: float = 1.0
    w_motion: float = 1.5
    w_goal: float = 4.0
    obstacle_thread: float = 1.2
    fear_ahead_dist: float = 2.5
    
    # Data
    env_list: List[str] = field(default_factory=list)
    test_env_id: int = 0
    data_cfg: DataConfig = field(default_factory=DataConfig)
    cost_map_name: str = "cost_map_sem"
    
    # System
    gpu_id: int = 0
    seed: int = 0
    num_workers: int = 4
    load_in_ram: bool = False
    
    # Paths
    file_path: str = "${USER_PATH_TO_MODEL_DATA}"
    file_name: Optional[str] = None
    
    # Logging
    wb_project: str = "viplanner"
    wb_entity: str = "viplanner"
    wb_api_key: str = ""
    n_visualize: int = 15
    
    # Hierarchical training
    hierarchical: bool = False
    hierarchical_step: int = 50
    hierarchical_front_step_ratio: float = 0.02
    hierarchical_back_step_ratio: float = 0.01
    
    # Resume
    resume: bool = False
    
    def get_model_save(self, epoch: Optional[int] = None) -> str:
        """Generate model save name"""
        input_domain = "DepSem" if self.sem else "Dep"
        cost_name = "Geom" if self.cost_map_name == "cost_map_geom" else "Sem"
        optim = "SGD" if self.optimizer == "sgd" else "Adam"
        name = f"_{self.file_name}" if self.file_name else ""
        epoch = epoch if epoch is not None else self.epochs
        hierarch = "_hierarch" if self.hierarchical else ""
        env = self.env_list[0] if self.env_list else "default"
        return f"plannernet_env{env}_ep{epoch}_input{input_domain}_cost{cost_name}_optim{optim}{hierarch}{name}"

    @property
    def all_model_dir(self):
        return os.path.join(os.getenv("EXPERIMENT_DIRECTORY", self.file_path), "models")

    @property
    def curr_model_dir(self):
        return os.path.join(self.all_model_dir, self.get_model_save())

    @property
    def data_dir(self):
        return os.path.join(os.getenv("EXPERIMENT_DIRECTORY", self.file_path), "data")

    @property
    def log_dir(self):
        return os.path.join(os.getenv("EXPERIMENT_DIRECTORY", self.file_path), "logs")
