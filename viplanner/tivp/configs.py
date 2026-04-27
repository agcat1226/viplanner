# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
TiVP 配置类
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DiffusionConfig:
    """Diffusion 采样配置"""
    
    # 采样步数
    num_inference_steps: int = 20
    
    # 噪声调度
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"  # "linear", "scaled_linear", "squaredcos_cap_v2"
    
    # 轨迹参数
    horizon: int = 50  # 轨迹长度
    traj_dim: int = 2  # 轨迹维度 (x, y)
    
    # 条件编码维度
    context_dim: int = 512


@dataclass
class GuidanceConfig:
    """SDF 引导配置"""
    
    # 是否启用引导
    enabled: bool = False
    
    # 引导尺度
    guidance_scale: float = 10.0
    
    # 安全边距 (米)
    safety_margin: float = 0.3
    
    # SDF 截断距离
    sdf_truncation: float = 0.1
    
    # 梯度裁剪
    max_grad_norm: float = 10.0
    
    # 引导开始步数（从第几步开始引导）
    guidance_start_step: int = 0


@dataclass
class HistoryConfig:
    """历史帧配置"""
    
    # 历史窗口大小
    window_size: int = 4
    
    # 是否启用历史
    enabled: bool = False


@dataclass
class WarmStartConfig:
    """Warm-start 配置"""
    
    # 是否启用 warm-start
    enabled: bool = False
    
    # 噪声混合比例 [0, 1]，0=完全使用上次轨迹，1=完全随机
    noise_ratio: float = 0.3
    
    # 轨迹时间偏移（向前推进几步）
    time_shift: int = 5


@dataclass
class TiVPConfig:
    """TiVP 总配置"""
    
    # 子配置
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    warm_start: WarmStartConfig = field(default_factory=WarmStartConfig)
    
    # 模型路径
    model_checkpoint: Optional[str] = None
    sdf_checkpoint: Optional[str] = None
    
    # 设备
    device: str = "cuda"
    
    # 图像输入尺寸
    img_height: int = 64
    img_width: int = 64
    
    # 输入通道
    depth_channels: int = 1
    rgb_channels: int = 3
    sem_channels: int = 3
    
    # 是否使用语义
    use_semantics: bool = True
    
    # 是否使用 RGB
    use_rgb: bool = False
    
    # 最大深度值
    max_depth: float = 15.0
    
    # Phase 控制
    phase: int = 1  # 1: 最小闭环, 2: 时序增强, 3: SDF引导
    
    def __post_init__(self):
        """根据 phase 自动配置子模块"""
        if self.phase == 1:
            # Phase 1: 最小闭环
            self.history.enabled = False
            self.warm_start.enabled = False
            self.guidance.enabled = False
        elif self.phase == 2:
            # Phase 2: 时序增强
            self.history.enabled = True
            self.warm_start.enabled = True
            self.guidance.enabled = False
        elif self.phase == 3:
            # Phase 3: SDF 引导
            self.history.enabled = True
            self.warm_start.enabled = True
            self.guidance.enabled = True
        else:
            raise ValueError(f"Invalid phase: {self.phase}, must be 1, 2, or 3")
