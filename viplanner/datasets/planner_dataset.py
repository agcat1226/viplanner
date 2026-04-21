# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Dataset classes for VIPlanner training.
Separated from the original monolithic implementation for better modularity.
"""

import os
from typing import List, Optional

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from .preprocessing import ImageAugmentation


class PlannerData(Dataset):
    """
    PyTorch Dataset for planner training data.
    """

    def __init__(
        self,
        transform,
        semantics: bool = False,
        rgb: bool = False,
        pixel_mean: Optional[np.ndarray] = None,
        pixel_std: Optional[np.ndarray] = None,
        max_depth: float = 15.0,
        augmentation_cfg: Optional[dict] = None,
    ) -> None:
        self.transform = transform
        self.semantics = semantics
        self.rgb = rgb
        assert not (semantics and rgb), "Semantics and RGB cannot be used at the same time"
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.max_depth = max_depth

        # Image augmentation
        self.augmentor = ImageAugmentation(augmentation_cfg or {})
        self.flip_transform = transforms.RandomHorizontalFlip(p=1.0)

        # Data buffers
        self.depth_filename: List[str] = []
        self.sem_rgb_filename: List[str] = []
        self.depth_imgs: List[torch.Tensor] = []
        self.sem_imgs: List[torch.Tensor] = []
        self.odom: torch.Tensor = None
        self.goal: torch.Tensor = None
        self.pair_augment: np.ndarray = None
        self.fov_angle: float = 0.0
        self.load_ram: bool = False

    def update_buffers(
        self,
        depth_filename: List[str],
        sem_rgb_filename: List[str],
        odom: torch.Tensor,
        goal: torch.Tensor,
        pair_augment: np.ndarray,
    ) -> None:
        self.depth_filename = depth_filename
        self.sem_rgb_filename = sem_rgb_filename
        self.odom = odom
        self.goal = goal
        self.pair_augment = pair_augment

    def set_fov(self, fov_angle):
        self.fov_angle = fov_angle

    def load_data_in_memory(self) -> None:
        """Load all data into RAM for faster training"""
        from tqdm import tqdm
        for idx in tqdm(range(len(self.depth_filename)), desc="Loading images into RAM"):
            self.depth_imgs.append(self._load_depth_img(idx))
            if self.semantics or self.rgb:
                self.sem_imgs.append(self._load_sem_rgb_img(idx))
        self.load_ram = True

    def _load_depth_img(self, idx) -> torch.Tensor:
        if self.depth_filename[idx].endswith(".png"):
            depth_image = Image.open(self.depth_filename[idx])
            depth_image = np.array(depth_image)
        else:
            depth_image = np.load(self.depth_filename[idx])
        
        depth_image[~np.isfinite(depth_image)] = 0.0
        depth_image = (depth_image / 1000.0).astype("float32")
        depth_image[depth_image > self.max_depth] = 0.0

        # Apply augmentation
        depth_image = self.augmentor.augment_depth(depth_image)

        # Transform
        depth_image = self.transform(depth_image).type(torch.float32)
        if self.pair_augment[idx]:
            depth_image = self.flip_transform.forward(depth_image)

        return depth_image

    def _load_sem_rgb_img(self, idx) -> torch.Tensor:
        image = Image.open(self.sem_rgb_filename[idx])
        image = np.array(image)
        
        # Normalize
        if self.pixel_mean is not None and self.pixel_std is not None:
            image = (image - self.pixel_mean) / self.pixel_std

        # Apply augmentation
        image = self.augmentor.augment_sem_rgb(image)

        # Transform
        image = self.transform(image).type(torch.float32)
        if self.pair_augment[idx]:
            image = self.flip_transform.forward(image)

        return image

    def __len__(self):
        return len(self.depth_filename)

    def __getitem__(self, idx):
        if self.load_ram:
            depth_image = self.depth_imgs[idx]
            sem_rgb_image = self.sem_imgs[idx] if (self.semantics or self.rgb) else 0
        else:
            depth_image = self._load_depth_img(idx)
            sem_rgb_image = self._load_sem_rgb_img(idx) if (self.semantics or self.rgb) else 0

        return (
            depth_image,
            sem_rgb_image,
            self.odom[idx],
            self.goal[idx],
            self.pair_augment[idx],
        )


class PlannerDataGenerator:
    """
    Data generator for creating training/validation splits.
    Simplified version focusing on core functionality.
    """

    def __init__(
        self,
        root: str,
        semantics: bool = False,
        rgb: bool = False,
        cost_map=None,
    ) -> None:
        self.root = root
        self.semantics = semantics
        self.rgb = rgb
        self.cost_map = cost_map

    def split_samples(
        self,
        test_dataset: PlannerData,
        train_dataset: Optional[PlannerData] = None,
        generate_split: bool = False,
        ratio: float = 0.9,
    ) -> None:
        """
        Split data into train/test sets.
        Placeholder for actual implementation.
        """
        pass

    def cleanup(self):
        """Cleanup temporary files"""
        pass
