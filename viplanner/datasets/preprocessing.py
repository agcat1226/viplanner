# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Image preprocessing and augmentation utilities.
"""

import random
from typing import Optional

import cv2
import numpy as np
from skimage.util import random_noise


class ImageAugmentation:
    """
    Image augmentation for depth and semantic/RGB images.
    """

    def __init__(self, cfg: dict):
        self.depth_salt_pepper = cfg.get("depth_salt_pepper", None)
        self.depth_gaussian = cfg.get("depth_gaussian", None)
        self.depth_random_polygons_nb = cfg.get("depth_random_polygons_nb", None)
        self.depth_random_polygon_size = cfg.get("depth_random_polygon_size", 10)
        
        self.sem_rgb_pepper = cfg.get("sem_rgb_pepper", None)
        self.sem_rgb_black_img = cfg.get("sem_rgb_black_img", None)
        self.sem_rgb_random_polygons_nb = cfg.get("sem_rgb_random_polygons_nb", None)
        self.sem_rgb_random_polygon_size = cfg.get("sem_rgb_random_polygon_size", 20)

    def _add_random_polygons(self, image, nb_polygons, max_size):
        """Add random black polygons to image"""
        for i in range(nb_polygons):
            num_corners = random.randint(10, 20)
            polygon_points = np.random.randint(0, max_size, size=(num_corners, 2))
            x_offset = np.random.randint(0, image.shape[0])
            y_offset = np.random.randint(0, image.shape[1])
            polygon_points[:, 0] += x_offset
            polygon_points[:, 1] += y_offset

            hull = cv2.convexHull(polygon_points)
            cv2.fillPoly(image, [hull], (0, 0, 0))
        return image

    def augment_depth(self, depth_image: np.ndarray) -> np.ndarray:
        """Apply augmentation to depth image"""
        if self.depth_salt_pepper or self.depth_gaussian:
            depth_norm = (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image) + 1e-8)
            if self.depth_salt_pepper:
                depth_norm = random_noise(
                    depth_norm,
                    mode="s&p",
                    amount=self.depth_salt_pepper,
                    clip=False,
                )
            if self.depth_gaussian:
                depth_norm = random_noise(
                    depth_norm,
                    mode="gaussian",
                    mean=0,
                    var=self.depth_gaussian,
                    clip=False,
                )
            depth_image = depth_norm * (np.max(depth_image) - np.min(depth_image)) + np.min(depth_image)
        
        if self.depth_random_polygons_nb and self.depth_random_polygons_nb > 0:
            depth_image = self._add_random_polygons(
                depth_image,
                self.depth_random_polygons_nb,
                self.depth_random_polygon_size,
            )

        return depth_image

    def augment_sem_rgb(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentation to semantic/RGB image"""
        if self.sem_rgb_black_img:
            if random.randint(0, 99) < self.sem_rgb_black_img * 100:
                image = np.zeros_like(image)
        
        if self.sem_rgb_pepper:
            image = random_noise(
                image,
                mode="pepper",
                amount=self.sem_rgb_pepper,
                clip=False,
            )
        
        if self.sem_rgb_random_polygons_nb and self.sem_rgb_random_polygons_nb > 0:
            image = self._add_random_polygons(
                image,
                self.sem_rgb_random_polygons_nb,
                self.sem_rgb_random_polygon_size,
            )

        return image
