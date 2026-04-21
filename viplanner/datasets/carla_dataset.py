# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Carla dataset loader for VIPlanner training.

Dataset structure (~/Carla/carla/):
  depth/          - 5260 depth images (*.npy float32 mm, *.png 16-bit)
  semantics/      - 5260 semantic RGB images (1280x720)
  img_warp/       - 2111 pre-warped semantic images aligned to depth frame (848x480)
  camera_extrinsic.txt  - 50000 camera poses [x,y,z,qx,qy,qz,qw], 5 per unique position
  viewpoints_seed1_samples50000.pkl - same 50000 poses [x,y,z,qw,qx,qy,qz]
  intrinsics.txt  - camera intrinsics (2 rows: depth K, semantic K as 3x4 P matrices)
  maps/           - cost map (cost_map_sem_map.txt = xyz cost point cloud)
  cloud.ply       - point cloud of the scene

Mapping:
  - 50000 samples grouped by position (5 orientations per position = 5 goal directions)
  - 10000 unique positions; first 5260 correspond to depth images 0000-5259
  - Each sample: odom = camera_extrinsic[i], goal = relative goal in camera frame
  - img_warp images are pre-computed semantic projections onto depth frame
"""

import math
import os
import pickle
from pathlib import Path
from random import sample
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from .planner_dataset import PlannerData


class CarlaDataGenerator:
    """
    Data generator for the Carla dataset.

    Loads camera poses and depth/semantic images, then generates
    (odom, goal) training pairs using the pre-sampled viewpoints.
    """

    def __init__(
        self,
        root: str,
        semantics: bool = True,
        rgb: bool = False,
        max_depth: float = 15.0,
        max_goal_distance: float = 10.0,
        min_goal_distance: float = 0.5,
        pairs_per_image: int = 4,
        ratio: float = 0.9,
        cost_map=None,
        max_train_pairs: Optional[int] = None,
    ) -> None:
        self.root = os.path.expanduser(root)
        self.semantics = semantics
        self.rgb = rgb
        assert not (semantics and rgb), "semantics and rgb cannot both be True"
        self.max_depth = max_depth
        self.max_goal_distance = max_goal_distance
        self.min_goal_distance = min_goal_distance
        self.pairs_per_image = pairs_per_image
        self.ratio = ratio
        self.cost_map = cost_map
        self.max_train_pairs = max_train_pairs

        # Load camera intrinsics and compute FOV
        self._load_intrinsics()

        # Load all 50000 camera poses
        self._load_poses()

        # Build depth/semantic image file lists (5260 frames)
        self._build_file_lists()

        # Generate (odom, goal) training pairs
        self._generate_pairs()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _load_intrinsics(self) -> None:
        """Load camera intrinsics from intrinsics.txt (2 rows, each a 3x4 P matrix)."""
        intr = np.loadtxt(os.path.join(self.root, "intrinsics.txt"), delimiter=",")
        # Row 0 = depth camera K (as 3x4 P matrix)
        self.K_depth = intr[0].reshape(3, 4)[:3, :3]
        # Row 1 = semantic camera K
        self.K_sem = intr[1].reshape(3, 4)[:3, :3]

        # Horizontal FOV from depth camera focal length and principal point
        # alpha_fov = 2 * atan(cx / fx)  (half-angle on each side)
        self.alpha_fov = 2.0 * math.atan(self.K_depth[0, 2] / self.K_depth[0, 0])

    def _load_poses(self) -> None:
        """
        Load 50000 camera poses from camera_extrinsic.txt.
        Format: [x, y, z, qx, qy, qz, qw] per row.
        Poses are grouped: every 5 consecutive rows share the same xyz position
        (5 different goal orientations from the same camera location).
        """
        poses_np = np.loadtxt(
            os.path.join(self.root, "camera_extrinsic.txt"), delimiter=","
        )
        self.all_poses = torch.from_numpy(poses_np).float()  # [50000, 7]

        # Build group index: each group of 5 rows = one unique camera position
        # Group i corresponds to depth image i (for i < 5260)
        self._build_pose_groups()

    def _build_pose_groups(self) -> None:
        """
        Group consecutive rows that share the same xyz position.
        Returns list of (start_idx, end_idx) for each group.
        """
        self.pose_groups: List[Tuple[int, int]] = []
        poses_np = self.all_poses[:, :3].numpy()

        i = 0
        while i < len(poses_np):
            j = i + 1
            while j < len(poses_np) and np.allclose(poses_np[j, :3], poses_np[i, :3], atol=1e-2):
                j += 1
            self.pose_groups.append((i, j))
            i = j

        # First 5260 groups correspond to depth images 0-5259
        self.n_frames = min(len(self.pose_groups), 5260)

    def _build_file_lists(self) -> None:
        """Build sorted file lists for depth and semantic images."""
        depth_dir = os.path.join(self.root, "depth")
        sem_dir = os.path.join(self.root, "semantics")
        warp_dir = os.path.join(self.root, "img_warp")

        # Prefer .npy over .png for depth (higher precision)
        npy_files = sorted(Path(depth_dir).glob("*.npy"), key=lambda p: int(p.stem))
        self.depth_files = [str(p) for p in npy_files]

        # Semantic files (full resolution, used for warping if img_warp not available)
        self.sem_files = sorted(
            [str(p) for p in Path(sem_dir).glob("*.png")],
            key=lambda p: int(Path(p).stem),
        )

        # Pre-warped semantic images (subset, already aligned to depth frame)
        warp_indices = set(int(Path(p).stem) for p in Path(warp_dir).glob("*.png"))
        self.warp_files: Dict[int, str] = {
            int(Path(p).stem): str(p)
            for p in Path(warp_dir).glob("*.png")
        }

        assert len(self.depth_files) == len(self.sem_files), (
            f"Depth ({len(self.depth_files)}) and semantic ({len(self.sem_files)}) "
            "file counts do not match"
        )
        self.n_depth_files = len(self.depth_files)

    # ------------------------------------------------------------------
    # Pair generation
    # ------------------------------------------------------------------

    def _generate_pairs(self) -> None:
        """
        Generate (odom, goal) training pairs.

        For each depth image frame i (0 to n_frames-1):
          - odom = the camera pose for that frame (first pose in group i)
          - goal candidates = poses from OTHER groups within [min_dist, max_dist]
          - goal in camera frame = transform world-frame goal into odom frame

        We use the pre-sampled viewpoints (pose groups) as goal candidates.
        """
        print(f"[CarlaDataGenerator] Generating pairs from {self.n_frames} frames...")

        odom_list: List[torch.Tensor] = []
        goal_list: List[torch.Tensor] = []
        depth_list: List[str] = []
        sem_list: List[str] = []
        augment_list: List[bool] = []

        # Get representative pose for each frame (first pose in group = camera position)
        frame_poses = torch.stack([
            self.all_poses[self.pose_groups[i][0]]
            for i in range(self.n_frames)
        ])  # [n_frames, 7]

        frame_xyz = frame_poses[:, :3].numpy()  # [n_frames, 3]

        # For each frame, find goal candidates within distance range
        from scipy.spatial import KDTree
        kdtree = KDTree(frame_xyz[:, :2])  # 2D distance (x, y)

        pairs_generated = 0
        max_pairs = self.max_train_pairs or (self.n_frames * self.pairs_per_image * 2)

        for frame_idx in range(self.n_frames):
            if pairs_generated >= max_pairs:
                break

            odom_pose = frame_poses[frame_idx]  # [7]: x,y,z,qx,qy,qz,qw
            odom_xyz = frame_xyz[frame_idx]

            # Find frames within goal distance range
            candidate_indices = kdtree.query_ball_point(
                odom_xyz[:2], r=self.max_goal_distance
            )
            candidate_indices = [
                idx for idx in candidate_indices
                if idx != frame_idx
                and np.linalg.norm(frame_xyz[idx, :2] - odom_xyz[:2]) >= self.min_goal_distance
            ]

            if len(candidate_indices) == 0:
                continue

            # Sample up to pairs_per_image goals
            n_sample = min(self.pairs_per_image, len(candidate_indices))
            selected = np.random.choice(candidate_indices, n_sample, replace=False)

            for goal_frame_idx in selected:
                goal_pose = frame_poses[goal_frame_idx]  # world frame

                # Transform goal position into odom (camera) frame
                goal_cam = self._world_to_camera_frame(odom_pose, goal_pose)

                # Check if goal is within FOV (optional filter)
                goal_angle = abs(math.atan2(goal_cam[1].item(), goal_cam[0].item()))
                if goal_angle > self.alpha_fov / 2 * 1.5:  # allow 1.5x FOV
                    continue

                # Get semantic image path
                if self.semantics or self.rgb:
                    if frame_idx in self.warp_files:
                        sem_path = self.warp_files[frame_idx]
                    else:
                        sem_path = self.sem_files[frame_idx]
                else:
                    sem_path = ""

                odom_list.append(odom_pose)
                goal_list.append(goal_cam)
                depth_list.append(self.depth_files[frame_idx])
                sem_list.append(sem_path)
                augment_list.append(False)
                pairs_generated += 1

                # Add horizontally flipped augmentation
                goal_cam_flip = goal_cam.clone()
                goal_cam_flip[1] = -goal_cam_flip[1]
                odom_list.append(odom_pose)
                goal_list.append(goal_cam_flip)
                depth_list.append(self.depth_files[frame_idx])
                sem_list.append(sem_path)
                augment_list.append(True)
                pairs_generated += 1

        self.odom = torch.stack(odom_list)          # [N, 7]
        self.goal = torch.stack(goal_list)           # [N, 3]
        self.depth_files_paired = depth_list
        self.sem_files_paired = sem_list
        self.augment = np.array(augment_list, dtype=bool)

        print(f"[CarlaDataGenerator] Generated {len(self.odom)} training pairs "
              f"from {self.n_frames} frames")

    @staticmethod
    def _world_to_camera_frame(
        odom: torch.Tensor, goal: torch.Tensor
    ) -> torch.Tensor:
        """
        Transform goal world position into the odom (camera) frame.

        Args:
            odom: [7] tensor [x, y, z, qx, qy, qz, qw] - camera pose in world
            goal: [7] tensor [x, y, z, qx, qy, qz, qw] - goal pose in world

        Returns:
            [3] tensor: goal position in camera frame [x_fwd, y_left, z_up]
        """
        # Extract positions
        odom_pos = odom[:3].numpy()
        goal_pos = goal[:3].numpy()

        # Extract quaternion [qx, qy, qz, qw]
        q = odom[3:].numpy()  # [qx, qy, qz, qw]

        # Build rotation matrix from quaternion (world -> camera)
        qx, qy, qz, qw = q
        R = np.array([
            [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
            [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
        ])

        # Transform goal into camera frame
        delta = goal_pos - odom_pos
        goal_cam = R.T @ delta  # R.T = R_inv for rotation matrix

        return torch.from_numpy(goal_cam.astype(np.float32))

    # ------------------------------------------------------------------
    # Train/val split
    # ------------------------------------------------------------------

    def split_samples(
        self,
        train_dataset: "PlannerData",
        val_dataset: "PlannerData",
    ) -> None:
        """Split generated pairs into train and validation sets."""
        n = len(self.odom)
        indices = list(range(n))
        train_indices = sample(indices, int(n * self.ratio))
        val_indices = list(set(indices) - set(train_indices))

        train_dataset.update_buffers(
            depth_filename=[self.depth_files_paired[i] for i in train_indices],
            sem_rgb_filename=[self.sem_files_paired[i] for i in train_indices],
            odom=self.odom[train_indices],
            goal=self.goal[train_indices],
            pair_augment=self.augment[train_indices],
        )
        train_dataset.set_fov(self.alpha_fov)

        val_dataset.update_buffers(
            depth_filename=[self.depth_files_paired[i] for i in val_indices],
            sem_rgb_filename=[self.sem_files_paired[i] for i in val_indices],
            odom=self.odom[val_indices],
            goal=self.goal[val_indices],
            pair_augment=self.augment[val_indices],
        )
        val_dataset.set_fov(self.alpha_fov)

        print(f"[CarlaDataGenerator] Train: {len(train_indices)}, Val: {len(val_indices)}")
