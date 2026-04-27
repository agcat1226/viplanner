"""
处理后的 Databet 数据集加载器

用于加载 process_databet.py 处理后的数据，供训练和推理使用。
符合 VIPlanner 代码规范和开发文档要求。
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ProcessedDatabetDataset(Dataset):
    """
    处理后的 Databet 数据集
    
    加载 process_databet.py 生成的 npz 文件，返回训练所需的张量格式。
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        train_ratio: float = 0.8,
        normalize_images: bool = True,
        augment: bool = False,
    ):
        """
        初始化数据集
        
        Args:
            data_root: 处理后数据的根目录
            split: 数据集划分，"train" 或 "val"
            train_ratio: 训练集比例
            normalize_images: 是否归一化图像到 [0, 1]
            augment: 是否进行数据增强（暂未实现）
        """
        self.data_root = Path(data_root)
        self.split = split
        self.normalize_images = normalize_images
        self.augment = augment
        
        # 加载所有样本路径
        self.sample_paths = self._collect_sample_paths()
        
        # 划分训练/验证集
        self._split_dataset(train_ratio)
        
        logger.info(
            f"加载 {split} 数据集: {len(self.sample_paths)} 个样本 "
            f"来自 {data_root}"
        )
    
    def _collect_sample_paths(self) -> List[Path]:
        """
        收集所有样本文件路径
        
        Returns:
            sample_paths: 样本文件路径列表
        """
        sample_paths = []
        
        # 遍历所有序列目录
        for seq_dir in sorted(self.data_root.iterdir()):
            if not seq_dir.is_dir():
                continue
            
            # 收集该序列的所有样本
            seq_samples = sorted(seq_dir.glob("sample_*.npz"))
            sample_paths.extend(seq_samples)
        
        logger.info(f"共发现 {len(sample_paths)} 个样本")
        return sample_paths
    
    def _split_dataset(self, train_ratio: float) -> None:
        """
        划分训练/验证集
        
        Args:
            train_ratio: 训练集比例
        """
        total_samples = len(self.sample_paths)
        train_size = int(total_samples * train_ratio)
        
        if self.split == "train":
            self.sample_paths = self.sample_paths[:train_size]
        elif self.split == "val":
            self.sample_paths = self.sample_paths[train_size:]
        else:
            raise ValueError(f"未知的 split: {self.split}")
        
        logger.info(f"{self.split} 集样本数: {len(self.sample_paths)}")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.sample_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            sample: 包含以下字段的字典
                - images: Tensor[K, C, H, W], float32, 范围 [0, 1] 或 [0, 255]
                - depths: Tensor[K, 1, H, W], float32, 单位：米
                - trajectory: Tensor[L, 3], float32, 局部坐标系轨迹
                - current_position: Tensor[3], float32
                - current_rotation: Tensor[3], float32
                - valid_mask: Tensor[K], bool, 全为 True（预留）
        """
        sample_path = self.sample_paths[idx]
        
        # 加载 npz 文件
        data = np.load(sample_path)
        
        # 提取数据
        images = data["images"]  # (K, H, W, 3), uint8
        depths = data["depths"]  # (K, H, W), float32
        trajectory = data["trajectory"]  # (L, 3), float32
        current_position = data["current_position"]  # (3,), float32
        current_rotation = data["current_rotation"]  # (3,), float32
        
        # 转换为 Tensor
        images = torch.from_numpy(images).float()  # (K, H, W, 3)
        depths = torch.from_numpy(depths).float()  # (K, H, W)
        trajectory = torch.from_numpy(trajectory).float()  # (L, 3)
        current_position = torch.from_numpy(current_position).float()  # (3,)
        current_rotation = torch.from_numpy(current_rotation).float()  # (3,)
        
        # 图像归一化
        if self.normalize_images:
            images = images / 255.0  # [0, 255] -> [0, 1]
        
        # 调整维度顺序：(K, H, W, C) -> (K, C, H, W)
        images = images.permute(0, 3, 1, 2)  # (K, C, H, W)
        
        # 深度图增加通道维度：(K, H, W) -> (K, 1, H, W)
        depths = depths.unsqueeze(1)  # (K, 1, H, W)
        
        # 创建有效性掩码（当前全为 True，预留用于处理缺帧情况）
        K = images.shape[0]
        valid_mask = torch.ones(K, dtype=torch.bool)
        
        # 数据增强（可选，暂未实现）
        if self.augment and self.split == "train":
            images, depths, trajectory = self._augment_sample(
                images, depths, trajectory
            )
        
        # 构建返回字典
        sample = {
            "images": images,  # (K, C, H, W), float32
            "depths": depths,  # (K, 1, H, W), float32
            "trajectory": trajectory,  # (L, 3), float32
            "current_position": current_position,  # (3,), float32
            "current_rotation": current_rotation,  # (3,), float32
            "valid_mask": valid_mask,  # (K,), bool
        }
        
        # 数据有效性断言（符合代码规范）
        self._validate_sample(sample, idx)
        
        return sample
    
    def _validate_sample(self, sample: Dict[str, torch.Tensor], idx: int) -> None:
        """
        验证样本数据有效性
        
        Args:
            sample: 样本字典
            idx: 样本索引
        """
        # 检查维度
        images = sample["images"]
        depths = sample["depths"]
        trajectory = sample["trajectory"]
        
        assert images.ndim == 4, (
            f"images 维度错误: {images.shape}, 应为 (K, C, H, W)"
        )
        assert depths.ndim == 4, (
            f"depths 维度错误: {depths.shape}, 应为 (K, 1, H, W)"
        )
        assert trajectory.ndim == 2, (
            f"trajectory 维度错误: {trajectory.shape}, 应为 (L, 3)"
        )
        
        # 检查通道数
        assert images.shape[1] == 3, (
            f"images 通道数错误: {images.shape[1]}, 应为 3"
        )
        assert depths.shape[1] == 1, (
            f"depths 通道数错误: {depths.shape[1]}, 应为 1"
        )
        
        # 检查轨迹维度
        assert trajectory.shape[1] == 3, (
            f"trajectory 维度错误: {trajectory.shape[1]}, 应为 3"
        )
        
        # 检查有限值
        if not torch.isfinite(images).all():
            raise ValueError(f"样本 {idx} 的 images 包含 NaN/Inf")
        if not torch.isfinite(depths).all():
            raise ValueError(f"样本 {idx} 的 depths 包含 NaN/Inf")
        if not torch.isfinite(trajectory).all():
            raise ValueError(f"样本 {idx} 的 trajectory 包含 NaN/Inf")
    
    def _augment_sample(
        self,
        images: torch.Tensor,
        depths: torch.Tensor,
        trajectory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        数据增强（预留接口）
        
        Args:
            images: (K, C, H, W)
            depths: (K, 1, H, W)
            trajectory: (L, 3)
            
        Returns:
            增强后的 images, depths, trajectory
        """
        # TODO: 实现数据增强
        # 可能的增强方式：
        # - 随机水平翻转（需同步翻转轨迹）
        # - 随机亮度/对比度调整
        # - 随机添加噪声
        return images, depths, trajectory
    
    def get_statistics(self) -> Dict[str, float]:
        """
        计算数据集统计信息
        
        Returns:
            stats: 统计信息字典
        """
        logger.info("计算数据集统计信息...")
        
        traj_lengths = []
        traj_max_speeds = []
        
        for idx in range(len(self)):
            sample = self[idx]
            traj = sample["trajectory"]  # (L, 3)
            
            # 轨迹长度
            diffs = torch.diff(traj, dim=0)  # (L-1, 3)
            lengths = torch.norm(diffs, dim=1)  # (L-1,)
            traj_lengths.append(lengths.sum().item())
            
            # 最大速度（假设时间步长为 0.1s）
            dt = 0.1
            speeds = lengths / dt  # (L-1,)
            traj_max_speeds.append(speeds.max().item())
        
        stats = {
            "num_samples": len(self),
            "mean_traj_length": np.mean(traj_lengths),
            "std_traj_length": np.std(traj_lengths),
            "mean_max_speed": np.mean(traj_max_speeds),
            "std_max_speed": np.std(traj_max_speeds),
        }
        
        return stats


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    自定义 collate 函数，用于 DataLoader
    
    Args:
        batch: 样本列表，每个样本是 __getitem__ 返回的字典
        
    Returns:
        batched: 批量化后的字典，所有张量增加 batch 维度
            - images: Tensor[B, K, C, H, W]
            - depths: Tensor[B, K, 1, H, W]
            - trajectory: Tensor[B, L, 3]
            - current_position: Tensor[B, 3]
            - current_rotation: Tensor[B, 3]
            - valid_mask: Tensor[B, K]
    """
    # 检查 batch 是否为空
    if len(batch) == 0:
        raise ValueError("batch 为空")
    
    # 堆叠所有样本
    batched = {
        "images": torch.stack([s["images"] for s in batch], dim=0),  # (B, K, C, H, W)
        "depths": torch.stack([s["depths"] for s in batch], dim=0),  # (B, K, 1, H, W)
        "trajectory": torch.stack([s["trajectory"] for s in batch], dim=0),  # (B, L, 3)
        "current_position": torch.stack([s["current_position"] for s in batch], dim=0),  # (B, 3)
        "current_rotation": torch.stack([s["current_rotation"] for s in batch], dim=0),  # (B, 3)
        "valid_mask": torch.stack([s["valid_mask"] for s in batch], dim=0),  # (B, K)
    }
    
    return batched


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 创建数据集
    dataset = ProcessedDatabetDataset(
        data_root="processed_data",
        split="train",
        train_ratio=0.8,
    )
    
    # 测试单个样本
    sample = dataset[0]
    print("\n样本字段:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        else:
            print(f"  {key}: {value}")
    
    # 测试 DataLoader
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    
    batch = next(iter(loader))
    print("\nBatch 字段:")
    for key, value in batch.items():
        print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
    
    # 计算统计信息
    stats = dataset.get_statistics()
    print("\n数据集统计:")
    for key, value in stats.items():
        print(f"  {key}: {value:.3f}")
