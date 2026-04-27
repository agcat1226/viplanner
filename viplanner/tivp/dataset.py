# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
TiVP 数据集

用于训练 TiVP 模型的数据集类
"""

from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np


class TiVPDataset(Dataset):
    """
    TiVP 训练数据集
    
    数据格式：
    - depth_seq: 深度序列 (K, C, H, W)
    - rgb_seq: RGB/语义序列 (K, C, H, W)
    - traj_gt: 真实轨迹 (T, D)
    - goal: 目标点 (2,)
    """
    
    def __init__(
        self,
        data_root: str,
        window_size: int = 4,
        horizon: int = 50,
        phase: int = 1,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            data_root: 数据根目录
            window_size: 历史窗口大小 K
            horizon: 轨迹长度 T
            phase: 训练阶段（1, 2, 3）
            max_samples: 最大样本数（用于调试）
        """
        self.data_root = data_root
        self.window_size = window_size
        self.horizon = horizon
        self.phase = phase
        
        # TODO: 实现数据加载逻辑
        # 这里提供一个示例结构
        self.samples = self._load_samples(max_samples)
    
    def _load_samples(self, max_samples: Optional[int]) -> list:
        """
        加载样本列表
        
        Returns:
            samples: 样本列表，每个元素是一个字典，包含：
                - depth_paths: 深度图路径列表 (K,)
                - rgb_paths: RGB 图路径列表 (K,)
                - traj_path: 轨迹文件路径
                - goal: 目标点
        """
        # TODO: 实现实际的数据加载
        # 示例：从文件系统扫描数据
        samples = []
        
        # 这里应该扫描 data_root 目录，构建样本列表
        # 例如：
        # for episode_dir in Path(data_root).glob("episode_*"):
        #     for i in range(len(frames) - window_size - horizon):
        #         sample = {
        #             "depth_paths": [...],
        #             "rgb_paths": [...],
        #             "traj_path": ...,
        #             "goal": ...,
        #         }
        #         samples.append(sample)
        
        if max_samples is not None:
            samples = samples[:max_samples]
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        """
        获取一个样本
        
        Returns:
            batch: 包含以下键的字典
                - depth_seq: (K, C, H, W)
                - rgb_seq: (K, C, H, W) 或 None
                - traj_gt: (T, D)
                - goal: (2,)
        """
        sample = self.samples[idx]
        
        # TODO: 实现实际的数据读取
        # 这里提供示例代码
        
        # 加载深度序列
        depth_seq = self._load_depth_sequence(sample["depth_paths"])
        
        # 加载 RGB 序列
        rgb_seq = None
        if "rgb_paths" in sample:
            rgb_seq = self._load_rgb_sequence(sample["rgb_paths"])
        
        # 加载真实轨迹
        traj_gt = self._load_trajectory(sample["traj_path"])
        
        # 加载目标
        goal = torch.tensor(sample["goal"], dtype=torch.float32)
        
        batch = {
            "depth_seq": depth_seq,
            "rgb_seq": rgb_seq,
            "traj_gt": traj_gt,
            "goal": goal,
        }
        
        return batch
    
    def _load_depth_sequence(self, paths: list) -> torch.Tensor:
        """
        加载深度序列
        
        Args:
            paths: 深度图路径列表
        
        Returns:
            depth_seq: (K, C, H, W)
        """
        # TODO: 实现实际的图像加载
        # 示例：
        # depth_list = []
        # for path in paths:
        #     depth = load_depth_image(path)
        #     depth_list.append(depth)
        # depth_seq = torch.stack(depth_list, dim=0)
        
        # 临时返回随机数据
        depth_seq = torch.rand(self.window_size, 1, 64, 64)
        return depth_seq
    
    def _load_rgb_sequence(self, paths: list) -> torch.Tensor:
        """
        加载 RGB 序列
        
        Args:
            paths: RGB 图路径列表
        
        Returns:
            rgb_seq: (K, C, H, W)
        """
        # TODO: 实现实际的图像加载
        rgb_seq = torch.rand(self.window_size, 3, 64, 64)
        return rgb_seq
    
    def _load_trajectory(self, path: str) -> torch.Tensor:
        """
        加载轨迹
        
        Args:
            path: 轨迹文件路径
        
        Returns:
            traj: (T, D)
        """
        # TODO: 实现实际的轨迹加载
        # 示例：
        # traj_data = np.load(path)
        # traj = torch.from_numpy(traj_data).float()
        
        # 临时返回随机数据
        traj = torch.rand(self.horizon, 2)
        return traj


class DummyTiVPDataset(Dataset):
    """
    虚拟数据集（用于测试和调试）
    
    生成随机数据，用于快速验证训练流程
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        window_size: int = 4,
        horizon: int = 50,
        img_height: int = 64,
        img_width: int = 64,
    ):
        self.num_samples = num_samples
        self.window_size = window_size
        self.horizon = horizon
        self.img_height = img_height
        self.img_width = img_width
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> dict:
        """生成随机样本"""
        depth_seq = torch.rand(self.window_size, 1, self.img_height, self.img_width)
        rgb_seq = torch.rand(self.window_size, 3, self.img_height, self.img_width)
        
        # 生成简单的直线轨迹
        t = torch.linspace(0, 1, self.horizon).unsqueeze(1)
        goal = torch.rand(2) * 10
        traj_gt = t * goal
        
        return {
            "depth_seq": depth_seq,
            "rgb_seq": rgb_seq,
            "traj_gt": traj_gt,
            "goal": goal,
        }


def create_dataloader(
    cfg,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    use_dummy: bool = False,
    data_root: str = "processed_data",
):
    """
    创建数据加载器
    
    Args:
        cfg: TiVPConfig 配置
        batch_size: 批次大小
        num_workers: 工作线程数
        shuffle: 是否打乱
        use_dummy: 是否使用虚拟数据集
        data_root: 处理后数据的根目录
    
    Returns:
        dataloader: PyTorch DataLoader
    """
    from torch.utils.data import DataLoader
    
    if use_dummy:
        dataset = DummyTiVPDataset(
            num_samples=1000,
            window_size=cfg.history.window_size,
            horizon=cfg.diffusion.horizon,
            img_height=cfg.img_height,
            img_width=cfg.img_width,
        )
    else:
        # 使用处理好的真实数据集
        from viplanner.datasets.processed_databet_dataset import ProcessedDatabetDataset
        
        dataset = ProcessedDatabetDataset(
            data_root=data_root,
            split="train",
            train_ratio=0.8,
            normalize_images=True,
        )
        
        # 创建适配器包装器
        dataset = TiVPDatasetAdapter(dataset, cfg)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataloader


class TiVPDatasetAdapter(Dataset):
    """
    适配器：将 ProcessedDatabetDataset 转换为 TiVP 训练格式
    """
    
    def __init__(self, base_dataset, cfg):
        self.base_dataset = base_dataset
        self.cfg = cfg
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        """
        将 ProcessedDatabetDataset 格式转换为 TiVP 格式
        
        输入格式 (ProcessedDatabetDataset):
            - images: (K, C, H, W), float32, [0, 1]
            - depths: (K, 1, H, W), float32
            - trajectory: (L, 3), float32
        
        输出格式 (TiVP):
            - depth_seq: (K, C, H, W)
            - rgb_seq: (K, C, H, W)
            - traj_gt: (T, D), 只取 x, y 坐标
            - goal: (2,), 轨迹终点
        """
        sample = self.base_dataset[idx]
        
        depth_seq = sample["depths"]  # (K, 1, H, W)
        rgb_seq = sample["images"]  # (K, 3, H, W)
        trajectory = sample["trajectory"]  # (L, 3)
        
        # 只取 x, y 坐标（忽略 z）
        traj_gt = trajectory[:, :2]  # (L, 2)
        
        # 目标点是轨迹的最后一个点
        goal = traj_gt[-1]  # (2,)
        
        return {
            "depth_seq": depth_seq,
            "rgb_seq": rgb_seq,  # 总是返回 RGB，即使不使用
            "traj_gt": traj_gt,
            "goal": goal,
        }
