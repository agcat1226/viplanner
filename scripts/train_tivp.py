#!/usr/bin/env python3
# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
TiVP 训练脚本

支持三个阶段的训练：
- Phase 1: 最小闭环（无历史、无 warm-start、无 SDF）
- Phase 2: 时序增强（加历史、warm-start）
- Phase 3: SDF 引导（加 SDF guidance）
"""

import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

from viplanner.tivp import (
    TiVPConfig, 
    TemporalEncoder, 
    DiffusionUNet1D,
    create_dataloader,
)
from viplanner.tivp.samplers import DDPMScheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TiVPTrainer:
    """TiVP 训练器"""
    
    def __init__(self, cfg: TiVPConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        # 初始化模型
        self.encoder = TemporalEncoder(
            img_height=cfg.img_height,
            img_width=cfg.img_width,
            depth_channels=cfg.depth_channels,
            rgb_channels=cfg.rgb_channels if cfg.use_rgb else cfg.sem_channels,
            context_dim=cfg.diffusion.context_dim,
            use_rgb=cfg.use_semantics or cfg.use_rgb,
        ).to(self.device)
        
        self.diffusion_model = DiffusionUNet1D(
            traj_dim=cfg.diffusion.traj_dim,
            horizon=cfg.diffusion.horizon,
            context_dim=cfg.diffusion.context_dim,
        ).to(self.device)
        
        # 初始化调度器
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=cfg.diffusion.beta_start,
            beta_end=cfg.diffusion.beta_end,
            beta_schedule=cfg.diffusion.beta_schedule,
        )
        
        # 优化器
        self.optimizer = optim.AdamW(
            list(self.encoder.parameters()) + list(self.diffusion_model.parameters()),
            lr=1e-4,
        )
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        logger.info(f"TiVPTrainer initialized for Phase {cfg.phase}")
    
    def train_step(self, batch):
        """
        单步训练
        
        Args:
            batch: 包含 depth_seq, rgb_seq, traj_gt 的字典
        
        Returns:
            loss: 损失值
        """
        self.encoder.train()
        self.diffusion_model.train()
        
        # 解包数据
        depth_seq = batch["depth_seq"].to(self.device)  # (B, K, C, H, W)
        rgb_seq = batch.get("rgb_seq")
        if rgb_seq is not None:
            rgb_seq = rgb_seq.to(self.device)
        traj_gt = batch["traj_gt"].to(self.device)  # (B, T, D)
        
        batch_size = depth_seq.shape[0]
        
        # 编码上下文
        context = self.encoder(depth_seq, rgb_seq)  # (B, D_ctx)
        
        # 随机采样时间步
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps,
            (batch_size,), device=self.device
        )
        
        # 添加噪声
        noise = torch.randn_like(traj_gt)
        x_t = self.scheduler.add_noise(traj_gt, noise, timesteps)
        
        # 预测噪声
        eps_pred = self.diffusion_model(x_t, timesteps, context)
        
        # 计算损失
        loss = self.criterion(eps_pred, noise)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, train_loader: DataLoader, num_epochs: int = 100):
        """训练循环"""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch in train_loader:
                loss = self.train_step(batch)
                total_loss += loss
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
            
            # 保存检查点
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint = {
            "encoder": self.encoder.state_dict(),
            "diffusion_model": self.diffusion_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.cfg,
        }
        
        save_path = Path("checkpoints") / filename
        save_path.parent.mkdir(exist_ok=True)
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train TiVP model")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3],
                        help="Training phase (1: minimal, 2: temporal, 3: SDF)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--use-dummy", action="store_true",
                        help="Use dummy dataset for testing")
    parser.add_argument("--data-root", type=str, default="processed_data",
                        help="Root directory of processed data")
    
    args = parser.parse_args()
    
    # 创建配置（参数匹配 processed_data 实际维度）
    cfg = TiVPConfig(
        phase=args.phase,
        device=args.device,
        img_height=224,
        img_width=224,
        use_rgb=True,
    )
    cfg.diffusion.horizon = 20    # 数据 pred_len=20，非默认 50
    cfg.history.window_size = 8   # 数据 hist_len=8，非默认 4
    
    # 创建训练器
    trainer = TiVPTrainer(cfg)
    
    logger.info("=" * 60)
    logger.info(f"Training TiVP - Phase {args.phase}")
    logger.info("=" * 60)
    logger.info(f"Phase {args.phase} configuration:")
    logger.info(f"  - History: {cfg.history.enabled}")
    logger.info(f"  - Warm-start: {cfg.warm_start.enabled}")
    logger.info(f"  - SDF Guidance: {cfg.guidance.enabled}")
    logger.info(f"Data dimensions:")
    logger.info(f"  - Image size: {cfg.img_height}x{cfg.img_width}")
    logger.info(f"  - Horizon: {cfg.diffusion.horizon}")
    logger.info(f"  - Window size: {cfg.history.window_size}")
    logger.info(f"  - Use RGB: {cfg.use_rgb}")
    logger.info("=" * 60)
    
    if args.use_dummy:
        # 使用虚拟数据集进行测试
        logger.info("Using dummy dataset for testing...")
        train_loader = create_dataloader(cfg, args.batch_size, use_dummy=True)
        
        # 训练
        trainer.train(train_loader, num_epochs=args.epochs)
        
        logger.info("Training completed with dummy dataset!")
    else:
        # 使用真实数据集
        from pathlib import Path
        data_root = Path(args.data_root)
        
        if not data_root.exists():
            logger.error(f"Data root not found: {data_root}")
            logger.error("Please run process_databet_simple.py first!")
            return
        
        # 统计样本数
        num_samples = len(list(data_root.glob("*/sample_*.npz")))
        logger.info(f"Found {num_samples} samples in {data_root}")
        
        if num_samples == 0:
            logger.error("No samples found! Please process data first.")
            logger.error("Run: python process_databet_simple.py")
            return
        
        logger.info(f"Using real dataset from: {args.data_root}")
        train_loader = create_dataloader(
            cfg, 
            args.batch_size, 
            use_dummy=False,
            data_root=args.data_root,
        )
        
        # 训练
        trainer.train(train_loader, num_epochs=args.epochs)
        
        logger.info("Training completed with real dataset!")


if __name__ == "__main__":
    main()
