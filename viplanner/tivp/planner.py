# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
TiVP 规划器包装器

兼容原版 VIPlanner 的 Isaac Sim 接口
"""

from typing import Optional, Tuple
import torch
import numpy as np
import logging

from .configs import TiVPConfig
from .models import TemporalEncoder, DiffusionUNet1D
from .samplers import GuidedDiffusionSampler, DDPMScheduler
from .history import HistoryBuffer
from .warm_start import WarmStarter
from .sdf_guidance import SDFGuidance

logger = logging.getLogger(__name__)


class TiVPAlgo:
    """
    TiVP 规划算法包装器
    
    提供与原版 VIPlannerAlgo 兼容的接口：
    - goal_transformer(...)
    - path_transformer(...)
    - plan(...)
    """
    
    def __init__(self, cfg: TiVPConfig):
        """
        Args:
            cfg: TiVP 配置
        """
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        # 初始化模块
        self._init_models()
        self._init_history()
        self._init_warm_start()
        self._init_guidance()
        self._init_sampler()
        
        logger.info(f"TiVPAlgo initialized for Phase {cfg.phase}")
    
    def _init_models(self):
        """初始化神经网络模型"""
        # 时序编码器
        self.encoder = TemporalEncoder(
            img_height=self.cfg.img_height,
            img_width=self.cfg.img_width,
            depth_channels=self.cfg.depth_channels,
            rgb_channels=self.cfg.rgb_channels if self.cfg.use_rgb else self.cfg.sem_channels,
            context_dim=self.cfg.diffusion.context_dim,
            use_rgb=self.cfg.use_semantics or self.cfg.use_rgb,
        ).to(self.device)
        
        # Diffusion 模型
        self.diffusion_model = DiffusionUNet1D(
            traj_dim=self.cfg.diffusion.traj_dim,
            horizon=self.cfg.diffusion.horizon,
            context_dim=self.cfg.diffusion.context_dim,
        ).to(self.device)
        
        # 加载权重
        if self.cfg.model_checkpoint is not None:
            self.load_checkpoint(self.cfg.model_checkpoint)
        
        # 设置为评估模式
        self.encoder.eval()
        self.diffusion_model.eval()
    
    def _init_history(self):
        """初始化历史缓存"""
        if self.cfg.history.enabled:
            self.history_buffer = HistoryBuffer(
                window_size=self.cfg.history.window_size,
                img_height=self.cfg.img_height,
                img_width=self.cfg.img_width,
                depth_channels=self.cfg.depth_channels,
                rgb_channels=self.cfg.rgb_channels if self.cfg.use_rgb else self.cfg.sem_channels,
                device=self.cfg.device,
            )
        else:
            self.history_buffer = None
    
    def _init_warm_start(self):
        """初始化 warm-start"""
        if self.cfg.warm_start.enabled:
            self.warm_starter = WarmStarter(
                noise_ratio=self.cfg.warm_start.noise_ratio,
                time_shift=self.cfg.warm_start.time_shift,
                horizon=self.cfg.diffusion.horizon,
                traj_dim=self.cfg.diffusion.traj_dim,
            )
        else:
            self.warm_starter = None
    
    def _init_guidance(self):
        """初始化 SDF 引导"""
        if self.cfg.guidance.enabled:
            # 这里需要加载实际的 SDF 模型
            # 暂时使用 None，后续可以通过 set_sdf_model 设置
            self.sdf_guidance = SDFGuidance(
                sdf_model=None,
                guidance_scale=self.cfg.guidance.guidance_scale,
                safety_margin=self.cfg.guidance.safety_margin,
                max_grad_norm=self.cfg.guidance.max_grad_norm,
                sdf_truncation=self.cfg.guidance.sdf_truncation,
            )
        else:
            self.sdf_guidance = None
    
    def _init_sampler(self):
        """初始化采样器"""
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            num_inference_steps=self.cfg.diffusion.num_inference_steps,
            beta_start=self.cfg.diffusion.beta_start,
            beta_end=self.cfg.diffusion.beta_end,
            beta_schedule=self.cfg.diffusion.beta_schedule,
        )
        
        self.sampler = GuidedDiffusionSampler(
            model=self.diffusion_model,
            scheduler=scheduler,
            guidance_module=self.sdf_guidance,
            guidance_start_step=self.cfg.guidance.guidance_start_step,
        )
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载模型权重"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if "encoder" in checkpoint:
            self.encoder.load_state_dict(checkpoint["encoder"])
        if "diffusion_model" in checkpoint:
            self.diffusion_model.load_state_dict(checkpoint["diffusion_model"])
        
        logger.info("Checkpoint loaded successfully")
    
    def set_sdf_model(self, sdf_model):
        """设置 SDF 模型"""
        if self.sdf_guidance is not None:
            self.sdf_guidance.set_sdf_model(sdf_model)
            logger.info("SDF model set for guidance")
    
    def goal_transformer(
        self,
        goal_world: np.ndarray,
        cam_pos: np.ndarray,
        cam_quat: np.ndarray,
    ) -> torch.Tensor:
        """
        将世界坐标系目标转换到局部坐标系
        
        Args:
            goal_world: 世界坐标系目标点。Shape: (3,) or (B, 3)
            cam_pos: 相机世界位置。Shape: (3,)
            cam_quat: 相机世界四元数 (w, x, y, z)。Shape: (4,)
        
        Returns:
            goal_local: 局部坐标系目标。Shape: (B, 2) or (2,)
        """
        # 简化实现：只考虑 x-y 平面
        # 实际实现需要完整的旋转变换
        
        if goal_world.ndim == 1:
            goal_world = goal_world[np.newaxis, :]  # (1, 3)
        
        # 相对位置
        rel_pos = goal_world - cam_pos  # (B, 3)
        
        # 提取 x-y 分量（简化，实际需要旋转）
        goal_local = rel_pos[:, :2]  # (B, 2)
        
        return torch.from_numpy(goal_local).float().to(self.device)
    
    def path_transformer(
        self,
        traj_local: torch.Tensor,
        cam_pos: np.ndarray,
        cam_quat: np.ndarray,
    ) -> np.ndarray:
        """
        将局部坐标系轨迹转换到世界坐标系
        
        Args:
            traj_local: 局部轨迹。Shape: (B, T, 2)
            cam_pos: 相机世界位置。Shape: (3,)
            cam_quat: 相机世界四元数。Shape: (4,)
        
        Returns:
            traj_world: 世界坐标系轨迹。Shape: (B, T, 3)
        """
        # 简化实现
        traj_np = traj_local.detach().cpu().numpy()
        b, t, _ = traj_np.shape
        
        # 添加 z=0
        traj_world = np.zeros((b, t, 3))
        traj_world[:, :, :2] = traj_np
        
        # 平移到世界坐标系
        traj_world += cam_pos
        
        return traj_world
    
    @torch.no_grad()
    def plan(
        self,
        depth: torch.Tensor,
        rgb: Optional[torch.Tensor],
        goal_local: torch.Tensor,
        cam_pos: Optional[np.ndarray] = None,
        cam_quat: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        规划轨迹
        
        Args:
            depth: 当前深度图。Shape: (B, C, H, W) or (C, H, W)
            rgb: 当前 RGB/语义图。Shape: (B, C, H, W) or (C, H, W) or None
            goal_local: 局部目标点。Shape: (B, 2) or (2,)
            cam_pos: 相机位置（用于历史）
            cam_quat: 相机姿态（用于历史）
        
        Returns:
            traj_local: 局部轨迹。Shape: (B, T, D)
            info: 辅助信息字典
        """
        # 处理输入维度
        if depth.ndim == 3:
            depth = depth.unsqueeze(0)  # (1, C, H, W)
        if rgb is not None and rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)
        if goal_local.ndim == 1:
            goal_local = goal_local.unsqueeze(0)  # (1, 2)
        
        batch_size = depth.shape[0]
        
        # 更新历史
        if self.history_buffer is not None:
            self.history_buffer.update(
                depth=depth[0],  # 只保存第一个样本
                rgb=rgb[0] if rgb is not None else None,
                cam_pos=cam_pos,
                cam_quat=cam_quat,
            )
            
            # 获取历史序列
            depth_seq, rgb_seq = self.history_buffer.get_history_tensor(
                use_rgb=self.cfg.use_semantics or self.cfg.use_rgb
            )
            
            # 添加 batch 维度
            depth_seq = depth_seq.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)  # (B, K, C, H, W)
            if rgb_seq is not None:
                rgb_seq = rgb_seq.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        else:
            # Phase 1: 不使用历史，只用当前帧
            depth_seq = depth.unsqueeze(1)  # (B, 1, C, H, W)
            rgb_seq = rgb.unsqueeze(1) if rgb is not None else None
        
        # 编码上下文
        context = self.encoder(depth_seq, rgb_seq)  # (B, D_ctx)
        
        # TODO: 将 goal 信息融入 context
        # 这里简化处理，实际需要设计 goal 编码方式
        
        # 准备初始轨迹
        if self.warm_starter is not None:
            x_init = self.warm_starter.prepare(
                batch_size=batch_size,
                device=self.device,
            )
        else:
            x_init = None
        
        # 采样轨迹
        traj_local = self.sampler.sample(
            context=context,
            x_init=x_init,
            batch_size=batch_size,
            horizon=self.cfg.diffusion.horizon,
            traj_dim=self.cfg.diffusion.traj_dim,
            enable_guidance=self.cfg.guidance.enabled,
        )
        
        # 更新 warm-start
        if self.warm_starter is not None:
            self.warm_starter.update(traj_local)
        
        # 收集信息
        info = {
            "phase": self.cfg.phase,
            "history_enabled": self.cfg.history.enabled,
            "warm_start_enabled": self.cfg.warm_start.enabled,
            "guidance_enabled": self.cfg.guidance.enabled,
        }
        
        return traj_local, info
    
    def reset(self):
        """重置状态"""
        if self.history_buffer is not None:
            self.history_buffer.reset()
        if self.warm_starter is not None:
            self.warm_starter.reset()
        logger.info("TiVPAlgo reset")
