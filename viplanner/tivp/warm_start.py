# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Warm-start 初始化模块
"""

from typing import Optional
import torch


class WarmStarter:
    """
    Warm-start 轨迹初始化
    
    使用上一周期的预测轨迹作为当前周期的初始化，
    通过时间偏移和噪声混合提高轨迹连续性。
    """
    
    def __init__(
        self,
        noise_ratio: float = 0.3,
        time_shift: int = 5,
        horizon: int = 50,
        traj_dim: int = 2,
    ):
        """
        Args:
            noise_ratio: 噪声混合比例 [0, 1]
            time_shift: 时间偏移步数
            horizon: 轨迹长度 T
            traj_dim: 轨迹维度 D
        """
        self.noise_ratio = noise_ratio
        self.time_shift = time_shift
        self.horizon = horizon
        self.traj_dim = traj_dim
        
        self.prev_traj: Optional[torch.Tensor] = None
    
    def prepare(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        准备初始轨迹
        
        Args:
            batch_size: 批次大小 B
            device: 设备
            dtype: 数据类型
        
        Returns:
            x_init: 初始化轨迹。Shape: (B, T, D)
        """
        if self.prev_traj is None:
            # 第一次调用，返回纯随机噪声
            return torch.randn(
                batch_size, self.horizon, self.traj_dim,
                device=device, dtype=dtype
            )
        
        # 时间偏移：向前推进轨迹
        # (B, T, D) -> 取 [time_shift:] 部分
        shifted_traj = self.prev_traj[:, self.time_shift:, :]
        
        # 填充末尾（简单策略：重复最后一个点）
        last_point = self.prev_traj[:, -1:, :]  # (B, 1, D)
        padding = last_point.repeat(1, self.time_shift, 1)  # (B, time_shift, D)
        shifted_traj = torch.cat([shifted_traj, padding], dim=1)  # (B, T, D)
        
        # 确保在正确设备上
        shifted_traj = shifted_traj.to(device=device, dtype=dtype)
        
        # 生成随机噪声
        noise = torch.randn_like(shifted_traj)
        
        # 混合
        x_init = (1 - self.noise_ratio) * shifted_traj + self.noise_ratio * noise
        
        return x_init
    
    def update(self, traj: torch.Tensor):
        """
        更新上一周期轨迹
        
        Args:
            traj: 当前预测轨迹。Shape: (B, T, D)
        """
        # 保存副本，断开计算图
        self.prev_traj = traj.detach().clone()
    
    def reset(self):
        """重置状态"""
        self.prev_traj = None
