#!/usr/bin/env python3
"""
简化版 Databet 数据处理脚本
只使用标准库和 numpy，不依赖 torch/cv2
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessConfig:
    """数据处理配置"""
    input_root: str = "databet"
    output_root: str = "processed_data"
    hist_len: int = 8
    pred_len: int = 20
    stride: int = 5
    target_height: int = 224
    target_width: int = 224
    depth_scale: float = 1000.0
    use_local_frame: bool = True
    min_sequence_len: int = 30
    max_translation: float = 5.0  # 放宽到 5 米
    max_rotation_deg: float = 180.0  # 放宽到 180 度


class DatabetProcessor:
    """Databet 数据处理器"""
    
    def __init__(self, config: ProcessConfig):
        self.cfg = config
        self.input_path = Path(config.input_root)
        self.output_path = Path(config.output_root)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"初始化 DatabetProcessor")
        logger.info(f"输入路径: {self.input_path}")
        logger.info(f"输出路径: {self.output_path}")
        logger.info(f"历史窗口 K={config.hist_len}, 预测长度 L={config.pred_len}")
    
    def process_all_sequences(self) -> None:
        """处理所有序列"""
        sequence_dirs = sorted([d for d in self.input_path.iterdir() if d.is_dir()])
        logger.info(f"发现 {len(sequence_dirs)} 个序列")
        
        for seq_dir in sequence_dirs:
            try:
                self.process_sequence(seq_dir)
            except Exception as e:
                logger.exception(f"处理序列 {seq_dir.name} 失败: {e}")
                continue
    
    def process_sequence(self, seq_dir: Path) -> None:
        """处理单个序列"""
        seq_name = seq_dir.name
        logger.info(f"开始处理序列: {seq_name}")
        
        # 加载所有帧
        frames_data = self._load_sequence_frames(seq_dir)
        
        if len(frames_data) < self.cfg.min_sequence_len:
            logger.warning(f"序列 {seq_name} 长度不足 ({len(frames_data)} < {self.cfg.min_sequence_len})")
            return
        
        # 过滤有效帧
        valid_frames = self._filter_valid_frames(frames_data)
        
        if len(valid_frames) < self.cfg.min_sequence_len:
            logger.warning(f"序列 {seq_name} 有效帧不足")
            return
        
        # 生成样本
        samples = self._generate_sliding_window_samples(valid_frames, seq_name)
        
        # 保存样本
        self._save_samples(samples, seq_name)
        
        logger.info(f"序列 {seq_name} 完成，生成 {len(samples)} 个样本")
    
    def _load_sequence_frames(self, seq_dir: Path) -> List[Dict]:
        """加载序列中的所有帧"""
        frame_dirs = sorted([d for d in seq_dir.iterdir() if d.is_dir()])
        frames_data = []
        
        total = len(frame_dirs)
        for i, frame_dir in enumerate(frame_dirs):
            if i % 50 == 0:
                logger.info(f"  加载进度: {i}/{total}")
            
            try:
                frame_data = self._load_single_frame(frame_dir)
                if frame_data is not None:
                    frames_data.append(frame_data)
            except Exception as e:
                logger.warning(f"加载帧 {frame_dir.name} 失败: {e}")
                continue
        
        logger.info(f"  成功加载 {len(frames_data)} 帧")
        return frames_data
    
    def _load_single_frame(self, frame_dir: Path) -> Optional[Dict]:
        """加载单帧数据"""
        rgb_path = frame_dir / "camera.png"
        depth_path = frame_dir / "depth.png"
        data_path = frame_dir / "data.json"
        
        if not all([rgb_path.exists(), depth_path.exists(), data_path.exists()]):
            return None
        
        # 加载 RGB
        rgb = Image.open(rgb_path).convert('RGB')
        rgb = rgb.resize((self.cfg.target_width, self.cfg.target_height), Image.BILINEAR)
        rgb = np.array(rgb, dtype=np.uint8)  # (H, W, 3)
        
        # 加载 Depth
        depth = Image.open(depth_path)
        depth = depth.resize((self.cfg.target_width, self.cfg.target_height), Image.NEAREST)
        depth = np.array(depth, dtype=np.float32)
        
        # 如果是 RGB 格式，只取第一个通道
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        
        depth = depth / self.cfg.depth_scale  # 毫米 -> 米
        
        # 加载位姿
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        position = self._parse_vector(data.get("car_position", "(0,0,0)"))
        rotation = self._parse_vector(data.get("car_rotation", "(0,0,0)"))
        
        return {
            "rgb": rgb,
            "depth": depth,
            "position": position,
            "rotation": rotation,
            "frame_id": frame_dir.name,
        }
    
    @staticmethod
    def _parse_vector(vec_str: str) -> np.ndarray:
        """解析向量字符串"""
        vec_str = vec_str.strip("()").replace(" ", "")
        values = [float(v) for v in vec_str.split(",")]
        return np.array(values, dtype=np.float32)
    
    def _filter_valid_frames(self, frames_data: List[Dict]) -> List[Dict]:
        """过滤无效帧"""
        if len(frames_data) == 0:
            return []
        
        valid_frames = [frames_data[0]]
        
        for i in range(1, len(frames_data)):
            prev_frame = valid_frames[-1]
            curr_frame = frames_data[i]
            
            translation = np.linalg.norm(curr_frame["position"] - prev_frame["position"])
            rotation_diff = np.abs(curr_frame["rotation"] - prev_frame["rotation"])
            max_rot_diff = np.max(rotation_diff)
            
            if translation > self.cfg.max_translation:
                continue
            if max_rot_diff > self.cfg.max_rotation_deg:
                continue
            
            valid_frames.append(curr_frame)
        
        logger.info(f"  过滤后保留 {len(valid_frames)}/{len(frames_data)} 帧")
        return valid_frames
    
    def _generate_sliding_window_samples(
        self, frames_data: List[Dict], seq_name: str
    ) -> List[Dict]:
        """生成滑动窗口样本"""
        samples = []
        total_frames = len(frames_data)
        
        min_t = self.cfg.hist_len - 1
        max_t = total_frames - self.cfg.pred_len
        
        if max_t <= min_t:
            logger.warning(f"序列长度不足以生成样本")
            return []
        
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
        self, frames_data: List[Dict], center_t: int, seq_name: str, sample_id: int
    ) -> Optional[Dict]:
        """创建单个样本"""
        K = self.cfg.hist_len
        L = self.cfg.pred_len
        
        hist_start = center_t - K + 1
        hist_end = center_t + 1
        hist_frames = frames_data[hist_start:hist_end]
        
        traj_start = center_t + 1
        traj_end = center_t + 1 + L
        traj_frames = frames_data[traj_start:traj_end]
        
        current_frame = hist_frames[-1]
        current_pos = current_frame["position"]
        current_rot = current_frame["rotation"]
        
        images = np.stack([f["rgb"] for f in hist_frames], axis=0)
        depths = np.stack([f["depth"] for f in hist_frames], axis=0)
        
        traj_positions_world = np.stack([f["position"] for f in traj_frames], axis=0)
        
        if self.cfg.use_local_frame:
            traj_positions_local = self._world_to_local(
                traj_positions_world, current_pos, current_rot
            )
        else:
            traj_positions_local = traj_positions_world
        
        sample = {
            "images": images,
            "depths": depths,
            "trajectory": traj_positions_local,
            "current_position": current_pos,
            "current_rotation": current_rot,
            "seq_name": seq_name,
            "sample_id": sample_id,
        }
        
        if not self._validate_sample(sample):
            return None
        
        return sample
    
    def _world_to_local(
        self, points_world: np.ndarray, origin_pos: np.ndarray, origin_rot: np.ndarray
    ) -> np.ndarray:
        """世界坐标系转局部坐标系"""
        points_translated = points_world - origin_pos
        
        yaw_deg = origin_rot[1]
        yaw_rad = np.deg2rad(yaw_deg)
        
        cos_yaw = np.cos(-yaw_rad)
        sin_yaw = np.sin(-yaw_rad)
        
        x_local = cos_yaw * points_translated[:, 0] - sin_yaw * points_translated[:, 1]
        y_local = sin_yaw * points_translated[:, 0] + cos_yaw * points_translated[:, 1]
        z_local = points_translated[:, 2]
        
        points_local = np.stack([x_local, y_local, z_local], axis=1)
        return points_local
    
    def _validate_sample(self, sample: Dict) -> bool:
        """验证样本有效性"""
        for key in ["images", "depths", "trajectory"]:
            data = sample[key]
            if not np.isfinite(data).all():
                return False
        
        traj = sample["trajectory"]
        if len(traj) > 1:
            diffs = np.linalg.norm(np.diff(traj, axis=0), axis=1)
            if np.max(diffs) > 2.0:
                return False
        
        return True
    
    def _save_samples(self, samples: List[Dict], seq_name: str) -> None:
        """保存样本"""
        if len(samples) == 0:
            return
        
        seq_output_dir = self.output_path / seq_name
        seq_output_dir.mkdir(parents=True, exist_ok=True)
        
        total = len(samples)
        for i, sample in enumerate(samples):
            if i % 20 == 0:
                logger.info(f"  保存进度: {i}/{total}")
            
            sample_id = sample["sample_id"]
            sample_file = seq_output_dir / f"sample_{sample_id:06d}.npz"
            
            np.savez_compressed(
                sample_file,
                images=sample["images"],
                depths=sample["depths"],
                trajectory=sample["trajectory"],
                current_position=sample["current_position"],
                current_rotation=sample["current_rotation"],
                seq_name=sample["seq_name"],
                sample_id=sample["sample_id"],
            )
        
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
        
        logger.info(f"  保存完成: {len(samples)} 个样本")


def main():
    """主函数"""
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
    
    processor = DatabetProcessor(config)
    processor.process_all_sequences()
    
    logger.info("=" * 60)
    logger.info("所有序列处理完成！")
    logger.info(f"输出目录: {config.output_root}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
