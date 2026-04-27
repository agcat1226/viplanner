#!/usr/bin/env python3
"""
处理后数据可视化脚本

用于可视化 process_databet.py 生成的数据，帮助验证数据质量。
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def visualize_sample(sample_idx: int = 0, data_root: str = "processed_data"):
    """
    可视化单个样本
    
    Args:
        sample_idx: 样本索引
        data_root: 数据根目录
    """
    from viplanner.datasets.processed_databet_dataset import ProcessedDatabetDataset
    
    # 加载数据集
    dataset = ProcessedDatabetDataset(
        data_root=data_root,
        split="train",
        normalize_images=True,
    )
    
    if len(dataset) == 0:
        logger.error("数据集为空")
        return
    
    if sample_idx >= len(dataset):
        logger.warning(f"样本索引 {sample_idx} 超出范围，使用索引 0")
        sample_idx = 0
    
    # 加载样本
    sample = dataset[sample_idx]
    images = sample["images"]  # (K, C, H, W)
    depths = sample["depths"]  # (K, 1, H, W)
    trajectory = sample["trajectory"]  # (L, 3)
    
    K = images.shape[0]
    L = trajectory.shape[0]
    
    logger.info(f"可视化样本 {sample_idx}")
    logger.info(f"  历史窗口长度 K={K}")
    logger.info(f"  轨迹长度 L={L}")
    
    # 创建图形
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 可视化历史 RGB 图像
    for i in range(K):
        ax = plt.subplot(3, K, i + 1)
        # (C, H, W) -> (H, W, C)
        img = images[i].permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(f"RGB t-{K-1-i}", fontsize=10)
        ax.axis("off")
    
    # 2. 可视化历史深度图
    for i in range(K):
        ax = plt.subplot(3, K, K + i + 1)
        # (1, H, W) -> (H, W)
        depth = depths[i, 0].numpy()
        im = ax.imshow(depth, cmap="viridis")
        ax.set_title(f"Depth t-{K-1-i}", fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 3. 可视化轨迹（俯视图）
    ax = plt.subplot(3, 2, 5)
    traj_np = trajectory.numpy()
    
    # 绘制轨迹
    ax.plot(traj_np[:, 0], traj_np[:, 1], 'b-o', linewidth=2, markersize=4, label="Trajectory")
    
    # 标记起点和终点
    ax.plot(0, 0, 'r*', markersize=20, label="Current Position")
    ax.plot(traj_np[-1, 0], traj_np[-1, 1], 'g*', markersize=15, label="Goal")
    
    # 添加方向箭头
    for i in range(0, L-1, max(1, L//5)):
        dx = traj_np[i+1, 0] - traj_np[i, 0]
        dy = traj_np[i+1, 1] - traj_np[i, 1]
        ax.arrow(
            traj_np[i, 0], traj_np[i, 1], dx, dy,
            head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.5
        )
    
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title("Local Trajectory (Top View)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    
    # 4. 可视化轨迹统计
    ax = plt.subplot(3, 2, 6)
    
    # 计算统计信息
    diffs = np.diff(traj_np, axis=0)  # (L-1, 3)
    step_lengths = np.linalg.norm(diffs, axis=1)  # (L-1,)
    cumulative_length = np.cumsum(step_lengths)
    
    # 绘制累积距离
    ax.plot(range(1, L), cumulative_length, 'b-o', linewidth=2, markersize=4)
    ax.set_xlabel("Waypoint Index", fontsize=12)
    ax.set_ylabel("Cumulative Distance (m)", fontsize=12)
    ax.set_title("Trajectory Length", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # 添加统计信息文本
    total_length = cumulative_length[-1] if len(cumulative_length) > 0 else 0
    mean_step = np.mean(step_lengths) if len(step_lengths) > 0 else 0
    max_step = np.max(step_lengths) if len(step_lengths) > 0 else 0
    
    stats_text = (
        f"Total Length: {total_length:.2f} m\n"
        f"Mean Step: {mean_step:.3f} m\n"
        f"Max Step: {max_step:.3f} m\n"
        f"Num Points: {L}"
    )
    ax.text(
        0.05, 0.95, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    # 保存图像
    output_path = Path("visualizations")
    output_path.mkdir(exist_ok=True)
    output_file = output_path / f"sample_{sample_idx:06d}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"保存可视化结果到: {output_file}")
    
    plt.show()


def visualize_dataset_statistics(data_root: str = "processed_data"):
    """
    可视化数据集统计信息
    
    Args:
        data_root: 数据根目录
    """
    from viplanner.datasets.processed_databet_dataset import ProcessedDatabetDataset
    
    # 加载数据集
    dataset = ProcessedDatabetDataset(
        data_root=data_root,
        split="train",
    )
    
    if len(dataset) == 0:
        logger.error("数据集为空")
        return
    
    logger.info(f"分析数据集: {len(dataset)} 个样本")
    
    # 收集统计信息
    traj_lengths = []
    traj_max_speeds = []
    traj_curvatures = []
    
    for idx in range(min(len(dataset), 1000)):  # 最多分析 1000 个样本
        sample = dataset[idx]
        traj = sample["trajectory"]  # (L, 3)
        
        # 轨迹长度
        diffs = torch.diff(traj, dim=0)  # (L-1, 3)
        lengths = torch.norm(diffs, dim=1)  # (L-1,)
        traj_lengths.append(lengths.sum().item())
        
        # 最大速度（假设时间步长为 0.1s）
        dt = 0.1
        speeds = lengths / dt  # (L-1,)
        traj_max_speeds.append(speeds.max().item())
        
        # 曲率（简化计算）
        if len(diffs) > 1:
            angles = []
            for i in range(len(diffs) - 1):
                v1 = diffs[i].numpy()
                v2 = diffs[i+1].numpy()
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                angles.append(angle)
            if len(angles) > 0:
                traj_curvatures.append(np.mean(angles))
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 轨迹长度分布
    ax = axes[0, 0]
    ax.hist(traj_lengths, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel("Trajectory Length (m)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Trajectory Length Distribution", fontsize=14)
    ax.axvline(np.mean(traj_lengths), color='r', linestyle='--', label=f'Mean: {np.mean(traj_lengths):.2f}m')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 最大速度分布
    ax = axes[0, 1]
    ax.hist(traj_max_speeds, bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax.set_xlabel("Max Speed (m/s)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Max Speed Distribution", fontsize=14)
    ax.axvline(np.mean(traj_max_speeds), color='r', linestyle='--', label=f'Mean: {np.mean(traj_max_speeds):.2f}m/s')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 曲率分布
    ax = axes[1, 0]
    ax.hist(traj_curvatures, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax.set_xlabel("Mean Curvature (rad)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Trajectory Curvature Distribution", fontsize=14)
    ax.axvline(np.mean(traj_curvatures), color='r', linestyle='--', label=f'Mean: {np.mean(traj_curvatures):.3f}rad')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 统计摘要
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = f"""
    数据集统计摘要
    {'='*40}
    
    样本数量: {len(dataset)}
    分析样本: {min(len(dataset), 1000)}
    
    轨迹长度:
      均值: {np.mean(traj_lengths):.3f} m
      标准差: {np.std(traj_lengths):.3f} m
      最小值: {np.min(traj_lengths):.3f} m
      最大值: {np.max(traj_lengths):.3f} m
    
    最大速度:
      均值: {np.mean(traj_max_speeds):.3f} m/s
      标准差: {np.std(traj_max_speeds):.3f} m/s
      最小值: {np.min(traj_max_speeds):.3f} m/s
      最大值: {np.max(traj_max_speeds):.3f} m/s
    
    平均曲率:
      均值: {np.mean(traj_curvatures):.3f} rad
      标准差: {np.std(traj_curvatures):.3f} rad
    """
    
    ax.text(
        0.1, 0.9, stats_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    )
    
    plt.tight_layout()
    
    # 保存图像
    output_path = Path("visualizations")
    output_path.mkdir(exist_ok=True)
    output_file = output_path / "dataset_statistics.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"保存统计结果到: {output_file}")
    
    plt.show()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="可视化处理后的 databet 数据")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sample", "stats"],
        default="sample",
        help="可视化模式: sample (单个样本) 或 stats (数据集统计)"
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="样本索引（仅在 sample 模式下有效）"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="processed_data",
        help="处理后数据的根目录"
    )
    
    args = parser.parse_args()
    
    if args.mode == "sample":
        visualize_sample(args.sample_idx, args.data_root)
    elif args.mode == "stats":
        visualize_dataset_statistics(args.data_root)


if __name__ == "__main__":
    main()
