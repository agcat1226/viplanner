# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Trajectory cost computation (no pypose / open3d / wandb dependency).

Mirrors the logic of deprecated/viplanner/traj_cost_opt/traj_cost.py but uses
plain PyTorch SE3 transforms via rotation matrices.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cost_map import CostMap
from .traj_opt import TrajOpt


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion [qx, qy, qz, qw] to rotation matrix.

    Args:
        q: [..., 4]
    Returns:
        R: [..., 3, 3]
    """
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = torch.stack([
        1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw),
            2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw),
            2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return R


def _transform_points(odom: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Transform points from camera frame to world frame.

    Args:
        odom:   [B, 7]    [x, y, z, qx, qy, qz, qw]
        points: [B, N, 3] points in camera frame
    Returns:
        world_pts: [B, N, 3]
    """
    t = odom[:, :3].unsqueeze(1)          # [B, 1, 3]
    R = _quat_to_rotmat(odom[:, 3:])      # [B, 3, 3]
    # world = R @ p + t
    world_pts = (R.unsqueeze(1) @ points.unsqueeze(-1)).squeeze(-1) + t
    return world_pts


class TrajCost:
    """
    Computes the full trajectory loss given predicted waypoints and a cost map.

    Loss components (same as original):
      - obstacle loss  (w_obs)
      - height loss    (w_height)
      - goal loss      (w_goal)
      - motion loss    (w_motion)
      - fear BCE loss
    """

    def __init__(
        self,
        gpu_id: int = 0,
        w_obs: float = 0.25,
        w_height: float = 1.0,
        w_motion: float = 1.5,
        w_goal: float = 2.0,
        obstacle_thread: float = 0.75,
        robot_width: float = 0.6,
    ) -> None:
        self.gpu_id = gpu_id
        self.w_obs = w_obs
        self.w_height = w_height
        self.w_motion = w_motion
        self.w_goal = w_goal
        self.obstacle_thread = obstacle_thread
        self.robot_width = robot_width

        self.opt = TrajOpt()
        self.cost_map: Optional[CostMap] = None
        self._neg_reward: float = 0.0

    # ------------------------------------------------------------------
    def set_map(self, root_path: str, map_name: str) -> None:
        self.cost_map = CostMap.load(root_path, map_name, self.gpu_id)
        if self.cost_map.cfg.semantics:
            self._neg_reward = self.cost_map.cfg.sem_cost_map.negative_reward

    # ------------------------------------------------------------------
    def cost_of_traj(
        self,
        waypoints: torch.Tensor,   # [B, N, 3]  camera frame
        odom: torch.Tensor,        # [B, 7]     world frame pose
        goal: torch.Tensor,        # [B, 3]     camera frame goal
        fear: torch.Tensor,        # [B, 1]     predicted fear
        ahead_dist: float = 2.5,
    ) -> torch.Tensor:
        assert self.cost_map is not None, "Call set_map() before cost_of_traj()"

        B, N, _ = waypoints.shape

        # ---- transform waypoints to world frame ----
        world_ps = _transform_points(odom, waypoints)   # [B, N, 3]

        # ---- obstacle loss ----
        oloss_M = self._obstacle_loss(world_ps, B)      # [B*3, N-1] or [B, N]

        # ---- height loss ----
        hloss = self._height_loss(world_ps, odom, B)

        # ---- goal loss ----
        gloss = torch.mean(torch.log(
            torch.norm(goal[:, :3] - waypoints[:, -1, :], dim=1) + 1.0
        ))

        # ---- motion loss ----
        mloss = self._motion_loss(waypoints, goal, N)

        # ---- total trajectory loss ----
        traj_loss = (self.w_obs * torch.mean(torch.sum(oloss_M, dim=1))
                     + self.w_height * hloss
                     + self.w_goal * gloss
                     + self.w_motion * mloss)

        # ---- fear loss ----
        wp_ds = torch.norm(waypoints[:, 1:, :] - waypoints[:, :-1, :], dim=2)  # [B, N-1]
        goal_dists = torch.cumsum(wp_ds, dim=1)                                  # [B, N-1]

        # oloss_M shape: [3*B, N-1] (inflated) or [B, N-1]
        fear_oloss = oloss_M if oloss_M.shape[0] == B else oloss_M.reshape(3, B, -1).max(0).values
        fear_oloss = fear_oloss.clone()
        fear_oloss[goal_dists > ahead_dist] = 0.0
        fear_labels = (fear_oloss.max(dim=1).values > self.obstacle_thread + self._neg_reward).float().unsqueeze(1)
        fear_loss = nn.BCELoss()(fear, fear_labels)

        return traj_loss + fear_loss

    # ------------------------------------------------------------------
    def _obstacle_loss(self, world_ps: torch.Tensor, B: int) -> torch.Tensor:
        """Inflate robot footprint and sample cost map."""
        # tangent / normal vectors for footprint inflation
        tangent = world_ps[:, 1:, :2] - world_ps[:, :-1, :2]          # [B, N-1, 2]
        norm_t  = torch.norm(tangent, dim=2, keepdim=True).clamp(min=1e-6)
        tangent = tangent / norm_t
        normals = tangent[..., [1, 0]] * torch.tensor([-1, 1], device=world_ps.device)

        # three rows: right, center, left
        pts_center = world_ps[:, :-1, :]                                # [B, N-1, 3]
        pts_right  = pts_center.clone(); pts_right[..., :2]  += normals * self.robot_width / 2
        pts_left   = pts_center.clone(); pts_left[...,  :2]  -= normals * self.robot_width / 2
        pts_inflated = torch.cat([pts_right, pts_center, pts_left], dim=0)  # [3B, N-1, 3]

        norm_inds = self.cost_map.pos2ind_norm(pts_inflated)            # [3B, N-1, 2]
        cost_grid = self.cost_map.cost_array.T.unsqueeze(0).unsqueeze(0).expand(
            pts_inflated.shape[0], 1, -1, -1
        ).float()

        oloss = F.grid_sample(
            cost_grid,
            norm_inds.unsqueeze(1).float(),
            mode="bicubic",
            padding_mode="border",
            align_corners=False,
        ).squeeze(1).squeeze(1)                                         # [3B, N-1]

        return oloss.float()

    def _height_loss(self, world_ps: torch.Tensor, odom: torch.Tensor, B: int) -> torch.Tensor:
        norm_inds = self.cost_map.pos2ind_norm(world_ps)                # [B, N, 2]
        ground_grid = self.cost_map.ground_array.T.unsqueeze(0).unsqueeze(0).expand(
            B, 1, -1, -1
        ).float()

        h_pred = F.grid_sample(
            ground_grid,
            norm_inds.unsqueeze(1).float(),
            mode="bicubic",
            padding_mode="border",
            align_corners=False,
        ).squeeze(1).squeeze(1)                                         # [B, N]

        hloss = torch.abs(world_ps[:, :, 2] - odom[:, 2:3] - h_pred).sum(dim=1).mean()
        return hloss

    def _motion_loss(self, waypoints: torch.Tensor, goal: torch.Tensor, N: int) -> torch.Tensor:
        desired = self.opt.TrajGeneratorFromPFreeRot(
            goal[:, None, :3], step=1.0 / (N - 1)
        )
        desired_ds = torch.norm(desired[:, 1:N, :] - desired[:, :N-1, :], dim=2)
        wp_ds      = torch.norm(waypoints[:, 1:, :] - waypoints[:, :-1, :], dim=2)
        return torch.abs(desired_ds - wp_ds).sum(dim=1).mean()
