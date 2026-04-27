# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
历史帧缓存模块
"""

from typing import Optional, Tuple
import torch
import numpy as np


class HistoryBuffer:
    """
    管理历史观测帧的缓存
    
    维护固定大小的滑动窗口，存储：
    - RGB 图像历史
    - Depth 图像历史
    - 相机位姿历史
    """
    
    def __init__(
        self,
        window_size: int = 4,
        img_height: int = 64,
        img_width: int = 64,
        depth_channels: int = 1,
        rgb_channels: int = 3,
        device: str = "cuda",
    ):
        """
        Args:
            window_size: 历史窗口大小 K
            img_height: 图像高度 H
            img_width: 图像宽度 W
            depth_channels: 深度通道数
            rgb_channels: RGB/语义通道数
            device: 设备
        """
        self.window_size = window_size
        self.img_height = img_height
        self.img_width = img_width
        self.depth_channels = depth_channels
        self.rgb_channels = rgb_channels
        self.device = device
        
        # 初始化缓存
        self.reset()
    
    def reset(self):
        """重置缓存"""
        self.depth_history = []  # List of (C, H, W)
        self.rgb_history = []    # List of (C, H, W)
        self.pose_history = []   # List of (pos, quat)
        self.count = 0
    
    def update(
        self,
        depth: torch.Tensor,
        rgb: Optional[torch.Tensor] = None,
        cam_pos: Optional[np.ndarray] = None,
        cam_quat: Optional[np.ndarray] = None,
    ):
        """
        更新历史缓存
        
        Args:
            depth: 当前深度图。Shape: (C, H, W)
            rgb: 当前 RGB/语义图。Shape: (C, H, W)
            cam_pos: 相机位置。Shape: (3,)
            cam_quat: 相机四元数。Shape: (4,)
        """
        # 确保在正确设备上
        depth = depth.detach().to(self.device)
        if rgb is not None:
            rgb = rgb.detach().to(self.device)
        
        # 添加到历史
        self.depth_history.append(depth)
        if rgb is not None:
            self.rgb_history.append(rgb)
        if cam_pos is not None and cam_quat is not None:
            self.pose_history.append((cam_pos.copy(), cam_quat.copy()))
        
        # 维护窗口大小
        if len(self.depth_history) > self.window_size:
            self.depth_history.pop(0)
            if rgb is not None:
                self.rgb_history.pop(0)
            if cam_pos is not None:
                self.pose_history.pop(0)
        
        self.count += 1
    
    def get_history_tensor(
        self,
        use_rgb: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        获取历史张量
        
        Args:
            use_rgb: 是否返回 RGB 历史
        
        Returns:
            depth_seq: 深度历史。Shape: (K, C_depth, H, W)
            rgb_seq: RGB 历史（如果有）。Shape: (K, C_rgb, H, W) 或 None
        """
        if len(self.depth_history) == 0:
            # 返回零张量
            depth_seq = torch.zeros(
                self.window_size, self.depth_channels, 
                self.img_height, self.img_width,
                device=self.device
            )
            rgb_seq = None
            if use_rgb and len(self.rgb_history) > 0:
                rgb_seq = torch.zeros(
                    self.window_size, self.rgb_channels,
                    self.img_height, self.img_width,
                    device=self.device
                )
            return depth_seq, rgb_seq
        
        # 堆叠历史帧
        # 如果不足 window_size，用最早的帧填充
        k = len(self.depth_history)
        if k < self.window_size:
            # 重复第一帧
            padding = [self.depth_history[0]] * (self.window_size - k)
            depth_list = padding + self.depth_history
        else:
            depth_list = self.depth_history
        
        depth_seq = torch.stack(depth_list, dim=0)  # (K, C, H, W)
        
        rgb_seq = None
        if use_rgb and len(self.rgb_history) > 0:
            k_rgb = len(self.rgb_history)
            if k_rgb < self.window_size:
                padding = [self.rgb_history[0]] * (self.window_size - k_rgb)
                rgb_list = padding + self.rgb_history
            else:
                rgb_list = self.rgb_history
            rgb_seq = torch.stack(rgb_list, dim=0)  # (K, C, H, W)
        
        return depth_seq, rgb_seq
    
    def is_ready(self) -> bool:
        """检查是否已收集足够的历史帧"""
        return len(self.depth_history) >= self.window_size
    
    def __len__(self) -> int:
        """返回当前缓存的帧数"""
        return len(self.depth_history)
