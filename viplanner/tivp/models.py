# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
TiVP 神经网络模型
"""

import torch
import torch.nn as nn
import math


class TemporalEncoder(nn.Module):
    """
    时序特征编码器
    
    提取历史观测序列的时序上下文特征
    """
    
    def __init__(
        self,
        img_height: int = 64,
        img_width: int = 64,
        depth_channels: int = 1,
        rgb_channels: int = 3,
        hidden_dim: int = 256,
        context_dim: int = 512,
        use_rgb: bool = True,
    ):
        """
        Args:
            img_height: 图像高度 H
            img_width: 图像宽度 W
            depth_channels: 深度通道数
            rgb_channels: RGB/语义通道数
            hidden_dim: 隐藏层维度
            context_dim: 输出上下文维度 D_ctx
            use_rgb: 是否使用 RGB/语义
        """
        super().__init__()
        
        self.use_rgb = use_rgb
        
        # 深度编码器
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(depth_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        
        # RGB/语义编码器
        if use_rgb:
            self.rgb_encoder = nn.Sequential(
                nn.Conv2d(rgb_channels, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            feat_dim = 128 + 128
        else:
            feat_dim = 128
        
        # 时序聚合（简单平均池化，可替换为 LSTM/Transformer）
        self.temporal_fc = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, context_dim),
        )
    
    def forward(
        self,
        depth_seq: torch.Tensor,
        rgb_seq: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            depth_seq: 深度序列。Shape: (B, K, C_depth, H, W)
            rgb_seq: RGB 序列。Shape: (B, K, C_rgb, H, W) 或 None
        
        Returns:
            context: 上下文向量。Shape: (B, D_ctx)
        """
        b, k, c_d, h, w = depth_seq.shape
        
        # (B, K, C, H, W) -> (B*K, C, H, W)
        depth_flat = depth_seq.view(b * k, c_d, h, w)
        depth_feat = self.depth_encoder(depth_flat)  # (B*K, 128, 1, 1)
        depth_feat = depth_feat.view(b, k, 128)  # (B, K, 128)
        
        if self.use_rgb and rgb_seq is not None:
            c_rgb = rgb_seq.shape[2]
            rgb_flat = rgb_seq.view(b * k, c_rgb, h, w)
            rgb_feat = self.rgb_encoder(rgb_flat)  # (B*K, 128, 1, 1)
            rgb_feat = rgb_feat.view(b, k, 128)  # (B, K, 128)
            
            # 拼接特征
            feat = torch.cat([depth_feat, rgb_feat], dim=-1)  # (B, K, 256)
        else:
            feat = depth_feat  # (B, K, 128)
        
        # 时序聚合：简单平均
        feat_avg = feat.mean(dim=1)  # (B, 256 or 128)
        
        # 映射到上下文维度
        context = self.temporal_fc(feat_avg)  # (B, D_ctx)
        
        return context


class SinusoidalPositionEmbedding(nn.Module):
    """正弦位置编码"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: 时间步。Shape: (B,)
        
        Returns:
            emb: 位置编码。Shape: (B, dim)
        """
        device = timesteps.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class DiffusionUNet1D(nn.Module):
    """
    1D UNet for trajectory diffusion
    
    预测给定噪声轨迹和条件的噪声
    """
    
    def __init__(
        self,
        traj_dim: int = 2,
        horizon: int = 50,
        context_dim: int = 512,
        hidden_dim: int = 256,
        time_embed_dim: int = 128,
    ):
        """
        Args:
            traj_dim: 轨迹维度 D
            horizon: 轨迹长度 T
            context_dim: 条件上下文维度
            hidden_dim: 隐藏层维度
            time_embed_dim: 时间嵌入维度
        """
        super().__init__()
        
        self.traj_dim = traj_dim
        self.horizon = horizon
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )
        
        # 输入投影
        self.input_proj = nn.Linear(traj_dim, hidden_dim)
        
        # 条件投影
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        
        # 时间条件投影
        self.time_proj = nn.Linear(time_embed_dim, hidden_dim)
        
        # 简化的 UNet 结构（可扩展为更复杂的架构）
        self.down1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        
        self.down2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
        )
        
        self.mid = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
        )
        
        self.up2 = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.SiLU(),
        )
        
        self.up1 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
        )
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, traj_dim)
    
    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_t: 噪声轨迹。Shape: (B, T, D)
            timesteps: 扩散时间步。Shape: (B,)
            context: 条件上下文。Shape: (B, D_ctx)
        
        Returns:
            eps_pred: 预测噪声。Shape: (B, T, D)
        """
        b, t, d = x_t.shape
        
        # 时间嵌入
        t_emb = self.time_embed(timesteps)  # (B, time_embed_dim)
        t_feat = self.time_proj(t_emb)  # (B, hidden_dim)
        
        # 条件嵌入
        c_feat = self.context_proj(context)  # (B, hidden_dim)
        
        # 输入投影
        x = self.input_proj(x_t)  # (B, T, hidden_dim)
        
        # 添加时间和条件信息（广播）
        x = x + t_feat[:, None, :] + c_feat[:, None, :]  # (B, T, hidden_dim)
        
        # Encoder
        h1 = self.down1(x)  # (B, T, hidden_dim)
        h2 = self.down2(h1)  # (B, T, hidden_dim*2)
        
        # Middle
        h_mid = self.mid(h2)  # (B, T, hidden_dim*2)
        
        # Decoder with skip connections
        h = torch.cat([h_mid, h2], dim=-1)  # (B, T, hidden_dim*4)
        h = self.up2(h)  # (B, T, hidden_dim)
        
        h = torch.cat([h, h1], dim=-1)  # (B, T, hidden_dim*2)
        h = self.up1(h)  # (B, T, hidden_dim)
        
        # 输出
        eps_pred = self.output_proj(h)  # (B, T, D)
        
        return eps_pred
