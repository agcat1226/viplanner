#!/usr/bin/env python3
# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
TiVP 推理测试脚本

测试三个阶段的推理功能
"""

import argparse
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from viplanner.tivp import TiVPConfig, TiVPAlgo
from viplanner.tivp.sdf_guidance import DummySDFModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dummy_observation(
    batch_size: int = 1,
    img_height: int = 64,
    img_width: int = 64,
    device: str = "cuda",
):
    """创建虚拟观测数据"""
    depth = torch.rand(batch_size, 1, img_height, img_width, device=device)
    rgb = torch.rand(batch_size, 3, img_height, img_width, device=device)
    goal_local = torch.tensor([[5.0, 0.0]], device=device)  # 5米前方
    
    cam_pos = np.array([0.0, 0.0, 1.5])
    cam_quat = np.array([1.0, 0.0, 0.0, 0.0])
    
    return depth, rgb, goal_local, cam_pos, cam_quat


def visualize_trajectory(traj: torch.Tensor, save_path: str = None):
    """可视化轨迹"""
    traj_np = traj[0].cpu().numpy()  # (T, 2)
    
    plt.figure(figsize=(10, 8))
    plt.plot(traj_np[:, 0], traj_np[:, 1], 'b-', linewidth=2, label='Trajectory')
    plt.plot(traj_np[0, 0], traj_np[0, 1], 'go', markersize=10, label='Start')
    plt.plot(traj_np[-1, 0], traj_np[-1, 1], 'ro', markersize=10, label='End')
    
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('TiVP Predicted Trajectory')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Trajectory visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def test_phase_1(device: str = "cuda"):
    """测试 Phase 1: 最小闭环"""
    logger.info("=" * 60)
    logger.info("Testing Phase 1: Minimal Closed-Loop")
    logger.info("=" * 60)
    
    cfg = TiVPConfig(phase=1, device=device)
    planner = TiVPAlgo(cfg)
    
    # 创建虚拟观测
    depth, rgb, goal_local, cam_pos, cam_quat = create_dummy_observation(device=device)
    
    # 规划
    logger.info("Planning trajectory...")
    traj_local, info = planner.plan(depth, rgb, goal_local, cam_pos, cam_quat)
    
    logger.info(f"Trajectory shape: {traj_local.shape}")
    logger.info(f"Info: {info}")
    
    # 可视化
    visualize_trajectory(traj_local, "phase1_trajectory.png")
    
    logger.info("Phase 1 test completed successfully!")
    return planner, traj_local


def test_phase_2(device: str = "cuda"):
    """测试 Phase 2: 时序增强"""
    logger.info("=" * 60)
    logger.info("Testing Phase 2: Temporal Enhancement")
    logger.info("=" * 60)
    
    cfg = TiVPConfig(phase=2, device=device)
    planner = TiVPAlgo(cfg)
    
    # 模拟多帧推理
    trajectories = []
    
    for i in range(5):
        logger.info(f"Frame {i+1}/5")
        
        depth, rgb, goal_local, cam_pos, cam_quat = create_dummy_observation(device=device)
        
        # 模拟相机移动
        cam_pos[0] += i * 0.5
        
        traj_local, info = planner.plan(depth, rgb, goal_local, cam_pos, cam_quat)
        trajectories.append(traj_local)
        
        logger.info(f"  History buffer size: {len(planner.history_buffer)}")
        logger.info(f"  Warm-start used: {info['warm_start_enabled']}")
    
    # 可视化最后一帧
    visualize_trajectory(trajectories[-1], "phase2_trajectory.png")
    
    logger.info("Phase 2 test completed successfully!")
    return planner, trajectories


def test_phase_3(device: str = "cuda"):
    """测试 Phase 3: SDF 引导"""
    logger.info("=" * 60)
    logger.info("Testing Phase 3: SDF Guidance")
    logger.info("=" * 60)
    
    cfg = TiVPConfig(phase=3, device=device)
    planner = TiVPAlgo(cfg)
    
    # 设置虚拟 SDF 模型（障碍物在 (5, 5) 位置）
    dummy_sdf = DummySDFModel(obstacle_center=(5.0, 5.0), obstacle_radius=1.0)
    planner.set_sdf_model(dummy_sdf)
    
    # 创建虚拟观测
    depth, rgb, goal_local, cam_pos, cam_quat = create_dummy_observation(device=device)
    
    # 规划
    logger.info("Planning trajectory with SDF guidance...")
    traj_local, info = planner.plan(depth, rgb, goal_local, cam_pos, cam_quat)
    
    logger.info(f"Trajectory shape: {traj_local.shape}")
    logger.info(f"Info: {info}")
    logger.info(f"Guidance enabled: {info['guidance_enabled']}")
    
    # 可视化（包含障碍物）
    traj_np = traj_local[0].cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    
    # 绘制障碍物
    circle = plt.Circle((5.0, 5.0), 1.0, color='r', alpha=0.3, label='Obstacle')
    plt.gca().add_patch(circle)
    
    # 绘制轨迹
    plt.plot(traj_np[:, 0], traj_np[:, 1], 'b-', linewidth=2, label='Trajectory')
    plt.plot(traj_np[0, 0], traj_np[0, 1], 'go', markersize=10, label='Start')
    plt.plot(traj_np[-1, 0], traj_np[-1, 1], 'ro', markersize=10, label='End')
    
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('TiVP Trajectory with SDF Guidance')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig("phase3_trajectory.png")
    logger.info("Trajectory visualization saved to phase3_trajectory.png")
    plt.close()
    
    logger.info("Phase 3 test completed successfully!")
    return planner, traj_local


def main():
    parser = argparse.ArgumentParser(description="Test TiVP inference")
    parser.add_argument("--phase", type=int, default=None, choices=[1, 2, 3],
                        help="Test specific phase (default: test all)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    
    args = parser.parse_args()
    
    # 检查设备
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"
    
    # 创建输出目录
    Path("test_outputs").mkdir(exist_ok=True)
    
    try:
        if args.phase is None:
            # 测试所有阶段
            logger.info("Testing all phases...")
            test_phase_1(args.device)
            print("\n")
            test_phase_2(args.device)
            print("\n")
            test_phase_3(args.device)
        elif args.phase == 1:
            test_phase_1(args.device)
        elif args.phase == 2:
            test_phase_2(args.device)
        elif args.phase == 3:
            test_phase_3(args.device)
        
        logger.info("=" * 60)
        logger.info("All tests completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
