# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Diffusion 采样器
"""

from typing import Optional
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class DDPMScheduler:
    """
    DDPM 噪声调度器
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 20,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
    ):
        """
        Args:
            num_train_timesteps: 训练时的总步数
            num_inference_steps: 推理时的采样步数
            beta_start: beta 起始值
            beta_end: beta 结束值
            beta_schedule: 调度类型
        """
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        
        # 生成 beta 序列
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "scaled_linear":
            betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")
        
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # 推理时间步（均匀采样）
        step_ratio = num_train_timesteps // num_inference_steps
        self.timesteps = torch.arange(0, num_train_timesteps, step_ratio).flip(0)
    
    def add_noise(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        添加噪声（前向过程）
        
        Args:
            x_0: 原始数据。Shape: (B, ...)
            noise: 噪声。Shape: (B, ...)
            timesteps: 时间步。Shape: (B,)
        
        Returns:
            x_t: 加噪后的数据。Shape: (B, ...)
        """
        # 确保所有张量在正确的设备上
        device = x_0.device
        if timesteps.device != device:
            timesteps = timesteps.to(device)
        if self.alphas_cumprod.device != device:
            self.alphas_cumprod = self.alphas_cumprod.to(device)
        
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        
        # 扩展维度以匹配 x_0
        while sqrt_alpha_prod.ndim < x_0.ndim:
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        x_t = sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise
        return x_t
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """
        单步去噪（反向过程）
        
        Args:
            model_output: 模型预测的噪声。Shape: (B, ...)
            timestep: 当前时间步
            sample: 当前样本。Shape: (B, ...)
        
        Returns:
            prev_sample: 去噪后的样本。Shape: (B, ...)
        """
        t = timestep
        
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t
        
        # 预测 x_0
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        
        # 计算 x_{t-1} 的均值
        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * self.betas[t]) / beta_prod_t
        current_sample_coeff = self.alphas[t] ** 0.5 * (1 - alpha_prod_t_prev) / beta_prod_t
        
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        
        return pred_prev_sample


class GuidedDiffusionSampler:
    """
    带引导的 Diffusion 采样器
    
    支持：
    - Warm-start 初始化
    - SDF 物理引导
    """
    
    def __init__(
        self,
        model: nn.Module,
        scheduler: DDPMScheduler,
        guidance_module: Optional[object] = None,
        guidance_start_step: int = 0,
    ):
        """
        Args:
            model: Diffusion 模型
            scheduler: 噪声调度器
            guidance_module: 引导模块（可选）
            guidance_start_step: 引导开始步数
        """
        self.model = model
        self.scheduler = scheduler
        self.guidance_module = guidance_module
        self.guidance_start_step = guidance_start_step
    
    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        x_init: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        horizon: int = 50,
        traj_dim: int = 2,
        enable_guidance: bool = False,
    ) -> torch.Tensor:
        """
        采样轨迹
        
        Args:
            context: 条件上下文。Shape: (B, D_ctx)
            x_init: 初始化轨迹（可选）。Shape: (B, T, D)
            batch_size: 批次大小
            horizon: 轨迹长度
            traj_dim: 轨迹维度
            enable_guidance: 是否启用引导
        
        Returns:
            x_0: 采样的轨迹。Shape: (B, T, D)
        """
        device = context.device
        dtype = context.dtype
        
        # 初始化
        if x_init is None:
            x_t = torch.randn(batch_size, horizon, traj_dim, device=device, dtype=dtype)
        else:
            x_t = x_init
        
        # 迭代去噪
        timesteps = self.scheduler.timesteps.to(device)
        
        for i, t in enumerate(timesteps):
            # 准备时间步张量
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # 模型预测噪声
            with torch.no_grad():
                eps_pred = self.model(x_t, t_tensor, context)
            
            # 去噪
            x_prev = self.scheduler.step(eps_pred, t.item(), x_t)
            
            # 应用引导（如果启用且达到开始步数）
            if enable_guidance and self.guidance_module is not None and i >= self.guidance_start_step:
                x_prev = self.guidance_module.apply_guidance(x_prev)
            
            x_t = x_prev
        
        return x_t
