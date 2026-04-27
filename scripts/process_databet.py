#!/usr/bin/env python3
"""
Databet 数据处理脚本

功能：
1. 读取 databet 目录下的原始数据
2. 提取 RGB、Depth、Pose 信息
3. 转换坐标系到局部坐标系
4. 生成时序滑动窗口样本
5. 保存为训练/推理可用的格式

符合规范：
- 显式维度标注
- 坐标系统一转换
- 配置化参数管理
- 完整日志记录
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessConfig:
    """数据处理配置"""
    # 输入输出路径
    input_root: str = "databet"
    output_root: str = "processed_data"
    
    # 时序窗口参数
    hist_len: int = 8  # K: 历史窗口长度
    pred_len: int = 20  # L: 预测轨迹长度
    stride: int = 1  # 滑动窗口步长
    
    # 图像参数
    target_height: int = 224
    target_width: int = 224
    depth_scale: float = 1000.0  # 深度图单位转换（毫米->米）
    
    # 坐标系参数
    use_local_frame: bool = True  # 是否转换到局部坐标系
    
    # 数据过滤
    min_sequence_len: int = 30  # 最小序列长度
    max_translation: float = 0.5  # 相邻帧最大平移（米）
    max_rotation_deg: float = 30.0  # 相邻帧最大旋转（度）


class DatabetProcessor:
    """Databet 数据处理器"""
    
    def __init__(self, config: ProcessConfig):
        """
        初始化处理器
        
        Args:
            config: 处理配置
        """
        self.cfg = config
        self.input_path = Path(config.input_root)
        self.output_path = Path(config.output_root)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"初始化 DatabetProcessor")
        logger.info(f"输入路径: {self.input_path}")
        logger.info(f"输出路径: {self.output_path}")
        logger.info(f"历史窗口长度 K={config.hist_len}, 预测长度 L={config.pred_len}")
    
    def process_all_sequences(self) -> None:
        """处理所有序列"""
        # 获取所有时间戳目录
        sequence_dirs = sorted([d for d in self.input_path.iterdir() if d.is_dir()])
        
        logger.info(f"发现 {len(sequence_dirs)} 个序列")
        
        for seq_dir in sequence_dirs:
            try:
                self.process_sequence(seq_dir)
            except Exception as e:
                logger.exception(f"处理序列 {seq_dir.name} 失败: {e}")
                continue
    
    def process_sequence(self, seq_dir: Path) -> None:
        """
        处理单个序列
        
        Args:
            seq_dir: 序列目录路径
        """
        seq_name = seq_dir.name
        logger.info(f"开始处理序列: {seq_name}")
        
        # 1. 加载所有帧数据
        frames_data = self._load_sequence_frames(seq_dir)
        
        if len(frames_data) < self.cfg.min_sequence_len:
            logger.warning(
                f"序列 {seq_name} 长度不足 ({len(frames_data)} < {self.cfg.min_sequence_len})，跳过"
            )
            return
        
        # 2. 数据质量检查
        valid_frames = self._filter_valid_frames(frames_data)
        
        if len(valid_frames) < self.cfg.min_sequence_len:
            logger.warning(f"序列 {seq_name} 有效帧不足，跳过")
            return
        
        # 3. 生成滑动窗口样本
        samples = self._generate_sliding_window_samples(valid_frames, seq_name)
        
        # 4. 保存处理后的数据
        self._save_samples(samples, seq_name)
        
        logger.info(f"序列 {seq_name} 处理完成，生成 {len(samples)} 个样本")
    
    def _load_sequence_frames(self, seq_dir: Path) -> List[Dict]:
        """
        加载序列中的所有帧
        
        Args:
            seq_dir: 序列目录
            
        Returns:
            frames_data: 帧数据列表，每个元素包含 rgb, depth, pose, timestamp
        """
        frame_dirs = sorted([d for d in seq_dir.iterdir() if d.is_dir()])
        frames_data = []
        
        for frame_dir in tqdm(frame_dirs, desc=f"加载 {seq_dir.name}"):
            try:
                frame_data = self._load_single_frame(frame_dir)
                if frame_data is not None:
                    frames_data.append(frame_data)
            except Exception as e:
                logger.warning(f"加载帧 {frame_dir.name} 失败: {e}")
                continue
        
        return frames_data
    
    def _load_single_frame(self, frame_dir: Path) -> Optional[Dict]:
        """
        加载单帧数据
        
        Args:
            frame_dir: 帧目录路径
            
        Returns:
            frame_data: 包含以下字段的字典
                - rgb: np.ndarray, shape (H, W, 3), dtype uint8
                - depth: np.ndarray, shape (H, W), dtype float32, 单位：米
                - position: np.ndarray, shape (3,), 世界坐标系位置
                - rotation: np.ndarray, shape (3,), 欧拉角（度）
                - frame_id: str
        """
        # 检查必要文件
        rgb_path = frame_dir / "camera.png"
        depth_path = frame_dir / "depth.png"
        data_path = frame_dir / "data.json"
        
        if not all([rgb_path.exists(), depth_path.exists(), data_path.exists()]):
            return None
        
        # 加载 RGB
        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            return None
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # 加载 Depth
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            return None
        # 转换单位：毫米 -> 米
        depth = depth.astype(np.float32) / self.cfg.depth_scale
        
        # 加载位姿
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # 解析位置和旋转
        position = self._parse_vector(data.get("car_position", "(0,0,0)"))
        rotation = self._parse_vector(data.get("car_rotation", "(0,0,0)"))
        
        # Resize 图像
        rgb_resized = cv2.resize(
            rgb, 
            (self.cfg.target_width, self.cfg.target_height),
            interpolation=cv2.INTER_LINEAR
        )
        depth_resized = cv2.resize(
            depth,
            (self.cfg.target_width, self.cfg.target_height),
            interpolation=cv2.INTER_NEAREST
        )
        
        return {
            "rgb": rgb_resized,  # (H, W, 3)
            "depth": depth_resized,  # (H, W)
            "position": position,  # (3,)
            "rotation": rotation,  # (3,)
            "frame_id": frame_dir.name,
        }
    
    @staticmethod
    def _parse_vector(vec_str: str) -> np.ndarray:
        """
        解析向量字符串
        
        Args:
            vec_str: 格式如 "(x, y, z)"
            
        Returns:
            vec: shape (3,)
        """
        vec_str = vec_str.strip("()").replace(" ", "")
        values = [float(v) for v in vec_str.split(",")]
        return np.array(values, dtype=np.float32)
    
    def _filter_valid_frames(self, frames_data: List[Dict]) -> List[Dict]:
        """
        过滤无效帧
        
        Args:
            frames_data: 原始帧数据列表
            
        Returns:
            valid_frames: 过滤后的帧数据列表
        """
        if len(frames_data) == 0:
            return []
        
        valid_frames = [frames_data[0]]
        
        for i in range(1, len(frames_data)):
            prev_frame = valid_frames[-1]
            curr_frame = frames_data[i]
            
            # 计算位移
            translation = np.linalg.norm(
                curr_frame["position"] - prev_frame["position"]
            )
            
            # 计算旋转差
            rotation_diff = np.abs(curr_frame["rotation"] - prev_frame["rotation"])
            max_rot_diff = np.max(rotation_diff)
            
            # 检查是否超过阈值
            if translation > self.cfg.max_translation:
                logger.debug(f"帧 {curr_frame['frame_id']} 位移过大: {translation:.3f}m")
                continue
            
            if max_rot_diff > self.cfg.max_rotation_deg:
                logger.debug(f"帧 {curr_frame['frame_id']} 旋转过大: {max_rot_diff:.1f}°")
                continue
            
            valid_frames.append(curr_frame)
        
        logger.info(f"过滤后保留 {len(valid_frames)}/{len(frames_data)} 帧")
        return valid_frames

    
    def _generate_sliding_window_samples(
        self, 
        frames_data: List[Dict], 
        seq_name: str
    ) -> List[Dict]:
        """
        生成滑动窗口样本
        
        Args:
            frames_data: 帧数据列表
            seq_name: 序列名称
            
        Returns:
            samples: 样本列表，每个样本包含：
                - images: np.ndarray, shape (K, H, W, 3)
                - depths: np.ndarray, shape (K, H, W)
                - trajectory: np.ndarray, shape (L, 3), 局部坐标系轨迹
                - poses: np.ndarray, shape (K, 7), [x, y, z, qw, qx, qy, qz]
                - seq_name: str
                - sample_id: int
        """
        samples = []
        total_frames = len(frames_data)
        
        # 计算有效的中心时刻范围
        # 需要满足：有足够历史 (t >= K-1) 且有足够未来 (t + L <= total_frames)
        min_t = self.cfg.hist_len - 1
        max_t = total_frames - self.cfg.pred_len
        
        if max_t <= min_t:
            logger.warning(
                f"序列 {seq_name} 长度不足以生成样本 "
                f"(需要至少 {self.cfg.hist_len + self.cfg.pred_len} 帧)"
            )
            return []
        
        # 滑动窗口采样
        for t in range(min_t, max_t, self.cfg.stride):
            try:
                sample = self._create_sample(frames_data, t, seq_name, len(samples))
                if sample is not None:
                    samples.append(sample)
            except Exception as e:
                logger.warning(f"生成样本 t={t} 失败: {e}")
                continue
        
        return samples
    
    def _create_sample(
        self,
        frames_data: List[Dict],
        center_t: int,
        seq_name: str,
        sample_id: int,
    ) -> Optional[Dict]:
        """
        创建单个样本
        
        Args:
            frames_data: 帧数据列表
            center_t: 中心时刻索引
            seq_name: 序列名称
            sample_id: 样本ID
            
        Returns:
            sample: 样本字典
        """
        K = self.cfg.hist_len
        L = self.cfg.pred_len
        
        # 提取历史窗口 [t-K+1, ..., t]
        hist_start = center_t - K + 1
        hist_end = center_t + 1
        hist_frames = frames_data[hist_start:hist_end]
        
        # 提取未来轨迹 [t+1, ..., t+L]
        traj_start = center_t + 1
        traj_end = center_t + 1 + L
        traj_frames = frames_data[traj_start:traj_end]
        
        assert len(hist_frames) == K, f"历史窗口长度错误: {len(hist_frames)} != {K}"
        assert len(traj_frames) == L, f"轨迹长度错误: {len(traj_frames)} != {L}"
        
        # 当前时刻（历史窗口最后一帧）作为局部坐标系原点
        current_frame = hist_frames[-1]
        current_pos = current_frame["position"]  # (3,)
        current_rot = current_frame["rotation"]  # (3,) 欧拉角
        
        # 收集历史图像和深度
        images = np.stack([f["rgb"] for f in hist_frames], axis=0)  # (K, H, W, 3)
        depths = np.stack([f["depth"] for f in hist_frames], axis=0)  # (K, H, W)
        
        # 收集历史位姿（世界坐标系）
        hist_positions = np.stack([f["position"] for f in hist_frames], axis=0)  # (K, 3)
        hist_rotations = np.stack([f["rotation"] for f in hist_frames], axis=0)  # (K, 3)
        
        # 收集未来轨迹位置（世界坐标系）
        traj_positions_world = np.stack(
            [f["position"] for f in traj_frames], axis=0
        )  # (L, 3)
        
        # 转换到局部坐标系
        if self.cfg.use_local_frame:
            traj_positions_local = self._world_to_local(
                traj_positions_world,
                current_pos,
                current_rot
            )  # (L, 3)
        else:
            traj_positions_local = traj_positions_world
        
        # 构建样本
        sample = {
            "images": images,  # (K, H, W, 3), uint8
            "depths": depths,  # (K, H, W), float32
            "trajectory": traj_positions_local,  # (L, 3), float32
            "current_position": current_pos,  # (3,), float32
            "current_rotation": current_rot,  # (3,), float32
            "hist_positions": hist_positions,  # (K, 3), float32
            "hist_rotations": hist_rotations,  # (K, 3), float32
            "seq_name": seq_name,
            "sample_id": sample_id,
            "center_frame_id": current_frame["frame_id"],
        }
        
        # 数据有效性检查
        if not self._validate_sample(sample):
            return None
        
        return sample
    
    def _world_to_local(
        self,
        points_world: np.ndarray,
        origin_pos: np.ndarray,
        origin_rot: np.ndarray,
    ) -> np.ndarray:
        """
        将世界坐标系点转换到局部坐标系
        
        Args:
            points_world: 世界坐标系点, shape (N, 3)
            origin_pos: 局部坐标系原点在世界坐标系中的位置, shape (3,)
            origin_rot: 局部坐标系原点的旋转（欧拉角，度）, shape (3,)
            
        Returns:
            points_local: 局部坐标系点, shape (N, 3)
        """
        # 简化版本：只考虑平移和绕Z轴旋转（yaw）
        # 完整版本应使用完整的旋转矩阵
        
        # 平移
        points_translated = points_world - origin_pos  # (N, 3)
        
        # 绕Z轴旋转（假设 origin_rot[1] 是 yaw）
        yaw_deg = origin_rot[1]
        yaw_rad = np.deg2rad(yaw_deg)
        
        # 构建2D旋转矩阵（逆旋转）
        cos_yaw = np.cos(-yaw_rad)
        sin_yaw = np.sin(-yaw_rad)
        
        # 只旋转 x, y 坐标
        x_local = cos_yaw * points_translated[:, 0] - sin_yaw * points_translated[:, 1]
        y_local = sin_yaw * points_translated[:, 0] + cos_yaw * points_translated[:, 1]
        z_local = points_translated[:, 2]
        
        points_local = np.stack([x_local, y_local, z_local], axis=1)  # (N, 3)
        
        return points_local
    
    def _validate_sample(self, sample: Dict) -> bool:
        """
        验证样本数据有效性
        
        Args:
            sample: 样本字典
            
        Returns:
            is_valid: 是否有效
        """
        # 检查是否包含 NaN 或 Inf
        for key in ["images", "depths", "trajectory"]:
            data = sample[key]
            if not np.isfinite(data).all():
                logger.warning(f"样本 {sample['sample_id']} 的 {key} 包含 NaN/Inf")
                return False
        
        # 检查轨迹是否合理（不能有过大的跳变）
        traj = sample["trajectory"]  # (L, 3)
        if len(traj) > 1:
            diffs = np.linalg.norm(np.diff(traj, axis=0), axis=1)  # (L-1,)
            max_diff = np.max(diffs)
            if max_diff > 2.0:  # 单步最大位移 2 米
                logger.warning(
                    f"样本 {sample['sample_id']} 轨迹跳变过大: {max_diff:.3f}m"
                )
                return False
        
        return True
    
    def _save_samples(self, samples: List[Dict], seq_name: str) -> None:
        """
        保存样本到磁盘
        
        Args:
            samples: 样本列表
            seq_name: 序列名称
        """
        if len(samples) == 0:
            return
        
        # 创建序列输出目录
        seq_output_dir = self.output_path / seq_name
        seq_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存每个样本
        for sample in tqdm(samples, desc=f"保存 {seq_name}"):
            sample_id = sample["sample_id"]
            sample_file = seq_output_dir / f"sample_{sample_id:06d}.npz"
            
            # 保存为 npz 格式
            np.savez_compressed(
                sample_file,
                images=sample["images"],
                depths=sample["depths"],
                trajectory=sample["trajectory"],
                current_position=sample["current_position"],
                current_rotation=sample["current_rotation"],
                hist_positions=sample["hist_positions"],
                hist_rotations=sample["hist_rotations"],
                seq_name=sample["seq_name"],
                sample_id=sample["sample_id"],
                center_frame_id=sample["center_frame_id"],
            )
        
        # 保存序列元数据
        metadata = {
            "seq_name": seq_name,
            "num_samples": len(samples),
            "hist_len": self.cfg.hist_len,
            "pred_len": self.cfg.pred_len,
            "image_size": [self.cfg.target_height, self.cfg.target_width],
        }
        
        metadata_file = seq_output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"序列 {seq_name} 保存完成: {len(samples)} 个样本")


def main():
    """主函数"""
    # 创建配置
    config = ProcessConfig(
        input_root="databet",
        output_root="processed_data",
        hist_len=8,
        pred_len=20,
        stride=5,
        target_height=224,
        target_width=224,
        min_sequence_len=30,
    )
    
    # 创建处理器
    processor = DatabetProcessor(config)
    
    # 处理所有序列
    processor.process_all_sequences()
    
    logger.info("所有序列处理完成！")
    logger.info(f"输出目录: {config.output_root}")


if __name__ == "__main__":
    main()
