#!/usr/bin/env python3
"""
数据处理测试脚本

用于验证 process_databet.py 和 ProcessedDatabetDataset 的正确性。
"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_data_processing():
    """测试数据处理流程"""
    logger.info("=" * 60)
    logger.info("测试 1: 数据处理脚本")
    logger.info("=" * 60)
    
    # 检查输入数据是否存在
    input_path = Path("databet")
    if not input_path.exists():
        logger.error(f"输入目录不存在: {input_path}")
        logger.info("请确保 databet 目录存在并包含数据")
        return False
    
    # 检查是否有序列
    seq_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    if len(seq_dirs) == 0:
        logger.error("databet 目录下没有序列数据")
        return False
    
    logger.info(f"✓ 发现 {len(seq_dirs)} 个序列")
    
    # 检查第一个序列的结构
    first_seq = seq_dirs[0]
    frame_dirs = [d for d in first_seq.iterdir() if d.is_dir()]
    if len(frame_dirs) == 0:
        logger.error(f"序列 {first_seq.name} 没有帧数据")
        return False
    
    logger.info(f"✓ 序列 {first_seq.name} 包含 {len(frame_dirs)} 帧")
    
    # 检查第一帧的文件
    first_frame = frame_dirs[0]
    required_files = ["camera.png", "depth.png", "data.json"]
    for fname in required_files:
        if not (first_frame / fname).exists():
            logger.error(f"帧 {first_frame.name} 缺少文件: {fname}")
            return False
    
    logger.info(f"✓ 帧数据结构正确")
    
    # 运行数据处理
    logger.info("\n开始处理数据...")
    try:
        from process_databet import ProcessConfig, DatabetProcessor
        
        config = ProcessConfig(
            input_root="databet",
            output_root="processed_data_test",
            hist_len=4,  # 使用较小的值加快测试
            pred_len=10,
            stride=10,
            min_sequence_len=20,
        )
        
        processor = DatabetProcessor(config)
        processor.process_all_sequences()
        
        logger.info("✓ 数据处理完成")
        
    except Exception as e:
        logger.exception(f"✗ 数据处理失败: {e}")
        return False
    
    return True


def test_dataset_loading():
    """测试数据集加载"""
    logger.info("\n" + "=" * 60)
    logger.info("测试 2: 数据集加载")
    logger.info("=" * 60)
    
    # 检查处理后的数据是否存在
    output_path = Path("processed_data_test")
    if not output_path.exists():
        logger.error(f"处理后的数据目录不存在: {output_path}")
        return False
    
    try:
        from viplanner.datasets.processed_databet_dataset import (
            ProcessedDatabetDataset,
            collate_fn,
        )
        from torch.utils.data import DataLoader
        
        # 创建数据集
        dataset = ProcessedDatabetDataset(
            data_root="processed_data_test",
            split="train",
            train_ratio=0.8,
        )
        
        if len(dataset) == 0:
            logger.error("数据集为空")
            return False
        
        logger.info(f"✓ 数据集包含 {len(dataset)} 个样本")
        
        # 测试单个样本
        sample = dataset[0]
        logger.info("\n样本字段:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        
        # 验证维度
        images = sample["images"]
        depths = sample["depths"]
        trajectory = sample["trajectory"]
        
        assert images.ndim == 4, f"images 维度错误: {images.shape}"
        assert images.shape[1] == 3, f"images 通道数错误: {images.shape[1]}"
        assert depths.ndim == 4, f"depths 维度错误: {depths.shape}"
        assert depths.shape[1] == 1, f"depths 通道数错误: {depths.shape[1]}"
        assert trajectory.ndim == 2, f"trajectory 维度错误: {trajectory.shape}"
        assert trajectory.shape[1] == 3, f"trajectory 维度错误: {trajectory.shape[1]}"
        
        logger.info("✓ 样本维度正确")
        
        # 验证数值范围
        assert torch.isfinite(images).all(), "images 包含 NaN/Inf"
        assert torch.isfinite(depths).all(), "depths 包含 NaN/Inf"
        assert torch.isfinite(trajectory).all(), "trajectory 包含 NaN/Inf"
        
        assert images.min() >= 0 and images.max() <= 1, "images 范围错误"
        assert depths.min() >= 0, "depths 包含负值"
        
        logger.info("✓ 样本数值有效")
        
        # 测试 DataLoader
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
        
        batch = next(iter(loader))
        logger.info("\nBatch 字段:")
        for key, value in batch.items():
            logger.info(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        
        # 验证 batch 维度
        B = batch["images"].shape[0]
        assert B == 2, f"batch size 错误: {B}"
        
        logger.info("✓ DataLoader 工作正常")
        
        # 计算统计信息
        logger.info("\n计算数据集统计信息...")
        stats = dataset.get_statistics()
        logger.info("数据集统计:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value:.3f}")
        
        logger.info("✓ 统计信息计算成功")
        
    except Exception as e:
        logger.exception(f"✗ 数据集加载失败: {e}")
        return False
    
    return True


def test_coordinate_transform():
    """测试坐标变换"""
    logger.info("\n" + "=" * 60)
    logger.info("测试 3: 坐标变换")
    logger.info("=" * 60)
    
    try:
        from process_databet import DatabetProcessor, ProcessConfig
        
        config = ProcessConfig()
        processor = DatabetProcessor(config)
        
        # 测试简单的坐标变换
        # 世界坐标系中的点
        points_world = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ], dtype=np.float32)
        
        # 局部坐标系原点在世界坐标系中的位置和旋转
        origin_pos = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        origin_rot = np.array([0.0, 90.0, 0.0], dtype=np.float32)  # 绕Z轴旋转90度
        
        # 转换到局部坐标系
        points_local = processor._world_to_local(
            points_world,
            origin_pos,
            origin_rot
        )
        
        logger.info("世界坐标系点:")
        logger.info(f"  {points_world}")
        logger.info("局部坐标系点:")
        logger.info(f"  {points_local}")
        
        # 验证转换结果
        assert np.isfinite(points_local).all(), "转换结果包含 NaN/Inf"
        
        logger.info("✓ 坐标变换正常")
        
    except Exception as e:
        logger.exception(f"✗ 坐标变换测试失败: {e}")
        return False
    
    return True


def main():
    """主函数"""
    logger.info("开始数据处理测试")
    logger.info("=" * 60)
    
    # 运行所有测试
    tests = [
        ("数据处理", test_data_processing),
        ("数据集加载", test_dataset_loading),
        ("坐标变换", test_coordinate_transform),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.exception(f"测试 {test_name} 异常: {e}")
            results.append((test_name, False))
    
    # 输出测试结果
    logger.info("\n" + "=" * 60)
    logger.info("测试结果汇总")
    logger.info("=" * 60)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"{test_name}: {status}")
    
    # 判断总体结果
    all_passed = all(result for _, result in results)
    
    if all_passed:
        logger.info("\n🎉 所有测试通过！")
        logger.info("\n下一步:")
        logger.info("1. 查看处理后的数据: processed_data_test/")
        logger.info("2. 使用 ProcessedDatabetDataset 进行训练")
        logger.info("3. 参考 DATABET_PROCESSING_GUIDE.md 了解更多")
        return 0
    else:
        logger.error("\n❌ 部分测试失败，请检查错误信息")
        return 1


if __name__ == "__main__":
    sys.exit(main())
