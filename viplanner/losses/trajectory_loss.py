# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Tuple

import torch
import torch.nn as nn


class TrajectoryLoss(nn.Module):
    """
    Trajectory loss that combines obstacle, height, motion, and goal costs.
    """

    def __init__(
        self,
        w_obs: float = 0.25,
        w_height: float = 1.0,
        w_motion: float = 1.5,
        w_goal: float = 4.0,
        obstacle_thread: float = 1.2,
        fear_ahead_dist: float = 2.5,
    ):
        super().__init__()
        self.w_obs = w_obs
        self.w_height = w_height
        self.w_motion = w_motion
        self.w_goal = w_goal
        self.obstacle_thread = obstacle_thread
        self.fear_ahead_dist = fear_ahead_dist

    def forward(
        self,
        preds: torch.Tensor,
        fear: torch.Tensor,
        waypoints: torch.Tensor,
        odom: torch.Tensor,
        goal: torch.Tensor,
        traj_cost_fn,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute trajectory loss.
        
        Args:
            preds: Predicted trajectory parameters
            fear: Fear prediction
            waypoints: Generated waypoints
            odom: Odometry
            goal: Goal position
            traj_cost_fn: Trajectory cost function object
            
        Returns:
            loss: Total loss
            loss_dict: Dictionary of individual loss components
        """
        loss = traj_cost_fn.CostofTraj(
            waypoints,
            odom,
            goal,
            fear,
            ahead_dist=self.fear_ahead_dist,
        )
        
        return loss, {}
