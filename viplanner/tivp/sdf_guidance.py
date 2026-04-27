# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
SDF 引导模块
"""

from typing import Optional
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class SDFGuidance:
    """
    SDF 物理引导
    
    在 diffusion 去噪过程中，使用 SDF 梯度引导轨迹远离障碍物
    """
    
    def __init__(
        self,
        sdf_model: Optional[nn.Module] = None,
        guidance_scale: float = 10.0,
        safety_margin: float = 0.3,
        max_grad_norm: float = 10.0,
        sdf_truncation: float = 0.1,
    ):
        """
        Args:
            sdf_model: SDF 查询模型（可选）
            guidance_scale: 引导尺度
            safety_margin: 安全边距（米）
            max_grad_norm: 最大梯度范数
            sdf_truncation: SDF 截断距离
        """
        self.sdf_model = sdf_model
        self.guidance_scale = guidance_scale
        self.safety_margin = safety_margin
        self.max_grad_norm = max_grad_norm
        self.sdf_truncation = sdf_truncation
    
    def set_sdf_model(self, sdf_model: nn.Module):
        """设置 SDF 模型"""
        self.sdf_model = sdf_model
    
    @torch.enable_grad()
    def compute_guidance(
        self,
        x_prev: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 SDF 引导梯度
        
        Args:
            x_prev: 去噪后的轨迹。Shape: (B, T, D)
        
        Returns:
            gradient: 引导梯度。Shape: (B, T, D)
        """
        if self.sdf_model is None:
            logger.warning("SDF model not set, returning zero gradient")
            return torch.zeros_like(x_prev)
        
        # 确保需要梯度
        x_prev = x_prev.detach().requires_grad_(True)
        
        # 查询 SDF 值
        # 注意：这里假设 sdf_model 接受 (B, T, D) 输入
        # 实际实现需要根据具体 SDF 模型调整
        sdf_values = self.sdf_model(x_prev)  # (B, T) or (B, T, 1)
        
        if sdf_values.ndim == 3:
            sdf_values = sdf_values.squeeze(-1)  # (B, T)
        
        # 计算碰撞惩罚
        # 当 SDF < safety_margin 时施加惩罚
        collision_mask = sdf_values < self.safety_margin
        penalty = torch.where(
            collision_mask,
            (self.safety_margin - sdf_values) ** 2,
            torch.zeros_like(sdf_values)
        )
        
        loss = penalty.sum()
        
        # 计算梯度
        gradient = torch.autograd.grad(
            outputs=loss,
            inputs=x_prev,
            retain_graph=False,
            create_graph=False,
        )[0]
        
        # 检查有限性
        if not torch.isfinite(gradient).all():
            logger.warning("Detected NaN/Inf in SDF gradient, using zero gradient")
            gradient = torch.zeros_like(gradient)
        
        # 梯度裁剪
        grad_norm = gradient.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        gradient = gradient * (self.max_grad_norm / grad_norm).clamp(max=1.0)
        
        return gradient.detach()
    
    def apply_guidance(
        self,
        x_prev: torch.Tensor,
    ) -> torch.Tensor:
        """
        应用 SDF 引导
        
        Args:
            x_prev: 去噪后的轨迹。Shape: (B, T, D)
        
        Returns:
            x_guided: 引导后的轨迹。Shape: (B, T, D)
        """
        gradient = self.compute_guidance(x_prev)
        x_guided = x_prev - self.guidance_scale * gradient
        return x_guided.detach()


class DummySDFModel(nn.Module):
    """
    虚拟 SDF 模型（用于测试）
    
    返回基于距离原点的简单 SDF 值
    """
    
    def __init__(self, obstacle_center: tuple = (5.0, 5.0), obstacle_radius: float = 1.0):
        super().__init__()
        self.obstacle_center = torch.tensor(obstacle_center)
        self.obstacle_radius = obstacle_radius
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 轨迹点。Shape: (B, T, 2)
        
        Returns:
            sdf: SDF 值。Shape: (B, T)
        """
        # 计算到障碍物中心的距离
        center = self.obstacle_center.to(x.device)
        dist = torch.norm(x - center, dim=-1)  # (B, T)
        
        # SDF: 距离 - 半径
        sdf = dist - self.obstacle_radius
        
        return sdf
