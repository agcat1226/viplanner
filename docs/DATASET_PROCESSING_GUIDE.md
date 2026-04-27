# Databet 数据处理指南

本文档说明如何使用 `process_databet.py` 脚本处理原始 databet 数据，以及如何在训练中使用处理后的数据。

## 目录结构

```
.
├── databet/                          # 原始数据目录
│   ├── 2026-04-22T01-15-55/         # 序列1
│   │   ├── frame_0001/
│   │   │   ├── camera.png           # RGB 图像
│   │   │   ├── depth.png            # 深度图
│   │   │   └── data.json            # 位姿数据
│   │   ├── frame_0002/
│   │   └── ...
│   ├── 2026-04-22T01-25-26/         # 序列2
│   └── 2026-04-22T02-07-36/         # 序列3
│
├── processed_data/                   # 处理后数据目录（自动生成）
│   ├── 2026-04-22T01-15-55/
│   │   ├── sample_000000.npz
│   │   ├── sample_000001.npz
│   │   ├── ...
│   │   └── metadata.json
│   └── ...
│
├── process_databet.py                # 数据处理脚本
└── viplanner/datasets/
    └── processed_databet_dataset.py  # 数据集加载器
```

## 第一步：处理原始数据

### 1.1 基本使用

直接运行脚本，使用默认配置：

```bash
python process_databet.py
```

默认配置：
- 输入目录：`databet/`
- 输出目录：`processed_data/`
- 历史窗口长度 K=8
- 预测轨迹长度 L=20
- 滑动窗口步长=5
- 图像尺寸：224x224

### 1.2 自定义配置

修改 `process_databet.py` 中的 `ProcessConfig` 类：

```python
config = ProcessConfig(
    input_root="databet",
    output_root="processed_data",
    hist_len=12,              # 增加历史窗口长度
    pred_len=30,              # 增加预测长度
    stride=3,                 # 减小步长，生成更多样本
    target_height=256,        # 更大的图像尺寸
    target_width=256,
    min_sequence_len=50,      # 更严格的序列长度要求
)
```

### 1.3 处理流程说明

脚本会执行以下步骤：

1. **加载序列帧**
   - 读取每个 frame 目录下的 RGB、Depth、Pose
   - 将深度图从毫米转换为米
   - Resize 图像到目标尺寸

2. **数据质量过滤**
   - 过滤相邻帧位移过大的帧（默认 >0.5m）
   - 过滤相邻帧旋转过大的帧（默认 >30°）
   - 确保序列长度满足最小要求

3. **生成滑动窗口样本**
   - 对每个有效时刻 t，提取历史窗口 [t-K+1, ..., t]
   - 提取未来轨迹 [t+1, ..., t+L]
   - 将轨迹转换到当前时刻的局部坐标系

4. **保存处理后数据**
   - 每个样本保存为独立的 `.npz` 文件
   - 保存序列元数据到 `metadata.json`

### 1.4 输出数据格式

每个 `sample_XXXXXX.npz` 文件包含：

| 字段 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `images` | (K, H, W, 3) | uint8 | 历史 RGB 图像序列 |
| `depths` | (K, H, W) | float32 | 历史深度图序列（米） |
| `trajectory` | (L, 3) | float32 | 未来轨迹（局部坐标系） |
| `current_position` | (3,) | float32 | 当前位置（世界坐标系） |
| `current_rotation` | (3,) | float32 | 当前旋转（欧拉角，度） |
| `hist_positions` | (K, 3) | float32 | 历史位置序列 |
| `hist_rotations` | (K, 3) | float32 | 历史旋转序列 |

## 第二步：在训练中使用数据

### 2.1 基本使用

```python
from torch.utils.data import DataLoader
from viplanner.datasets.processed_databet_dataset import (
    ProcessedDatabetDataset,
    collate_fn,
)

# 创建训练集
train_dataset = ProcessedDatabetDataset(
    data_root="processed_data",
    split="train",
    train_ratio=0.8,
    normalize_images=True,
)

# 创建验证集
val_dataset = ProcessedDatabetDataset(
    data_root="processed_data",
    split="val",
    train_ratio=0.8,
    normalize_images=True,
)

# 创建 DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True,
)
```

### 2.2 数据格式说明

DataLoader 返回的 batch 格式：

| 字段 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `images` | (B, K, C, H, W) | float32 | RGB 图像，范围 [0, 1] |
| `depths` | (B, K, 1, H, W) | float32 | 深度图（米） |
| `trajectory` | (B, L, 3) | float32 | 目标轨迹（局部坐标系） |
| `current_position` | (B, 3) | float32 | 当前位置 |
| `current_rotation` | (B, 3) | float32 | 当前旋转 |
| `valid_mask` | (B, K) | bool | 有效性掩码 |

其中：
- B: batch size
- K: 历史窗口长度
- C: 通道数（RGB=3）
- H, W: 图像高度和宽度
- L: 轨迹预测长度

### 2.3 在训练循环中使用

```python
import torch
import torch.nn as nn

# 假设已有模型
model = YourTemporalDiffusionModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    model.train()
    
    for batch in train_loader:
        # 将数据移到 GPU（符合设备无关性规范）
        device = next(model.parameters()).device
        images = batch["images"].to(device)  # (B, K, C, H, W)
        trajectory = batch["trajectory"].to(device)  # (B, L, 3)
        
        # 前向传播
        pred_trajectory = model(images)
        
        # 计算损失
        loss = nn.MSELoss()(pred_trajectory, trajectory)
        
        # 反向传播
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
```

## 第三步：数据验证与统计

### 3.1 查看数据集统计信息

```python
from viplanner.datasets.processed_databet_dataset import ProcessedDatabetDataset

dataset = ProcessedDatabetDataset(
    data_root="processed_data",
    split="train",
)

# 计算统计信息
stats = dataset.get_statistics()
print("数据集统计:")
for key, value in stats.items():
    print(f"  {key}: {value:.3f}")
```

输出示例：
```
数据集统计:
  num_samples: 1234
  mean_traj_length: 5.234
  std_traj_length: 1.123
  mean_max_speed: 2.456
  std_max_speed: 0.789
```

### 3.2 可视化样本

```python
import matplotlib.pyplot as plt
import numpy as np

# 加载单个样本
sample = dataset[0]
images = sample["images"]  # (K, C, H, W)
trajectory = sample["trajectory"]  # (L, 3)

# 可视化历史图像
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i in range(8):
    ax = axes[i // 4, i % 4]
    # (C, H, W) -> (H, W, C)
    img = images[i].permute(1, 2, 0).numpy()
    ax.imshow(img)
    ax.set_title(f"Frame t-{7-i}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("history_frames.png")

# 可视化轨迹（俯视图）
plt.figure(figsize=(8, 8))
traj_np = trajectory.numpy()
plt.plot(traj_np[:, 0], traj_np[:, 1], 'b-o', label="Predicted Trajectory")
plt.plot(0, 0, 'r*', markersize=15, label="Current Position")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.title("Local Trajectory")
plt.savefig("trajectory.png")
```

## 常见问题

### Q1: 处理速度慢怎么办？

A: 可以调整以下参数：
- 增大 `stride`，减少样本数量
- 减小 `target_height` 和 `target_width`
- 使用更快的存储设备

### Q2: 内存不足怎么办？

A: 
- 减小 DataLoader 的 `batch_size`
- 减小 `num_workers`
- 使用更小的图像尺寸

### Q3: 如何处理多个数据集？

A: 可以分别处理，然后合并输出目录：

```bash
# 处理数据集1
python process_databet.py --input databet1 --output processed_data1

# 处理数据集2
python process_databet.py --input databet2 --output processed_data2

# 合并（手动或脚本）
cp -r processed_data1/* processed_data/
cp -r processed_data2/* processed_data/
```

### Q4: 如何调试数据处理问题？

A: 启用详细日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 符合规范说明

本数据处理脚本严格遵循《开发文档(1).md》和《代码规范(1).md》：

### 维度规范
- ✅ 所有函数都有明确的 shape 注释
- ✅ 使用语义化的维度命名（B, K, L, C, H, W）
- ✅ 显式的维度变换并附带注释

### 坐标系规范
- ✅ 统一转换到局部坐标系（Local Base Link）
- ✅ 明确的坐标变换函数 `_world_to_local`
- ✅ 保留原始世界坐标系位姿用于调试

### 数据有效性
- ✅ 对所有关键数据进行 NaN/Inf 检查
- ✅ 对轨迹跳变进行合理性检查
- ✅ 对相邻帧位移和旋转进行过滤

### 配置管理
- ✅ 使用 dataclass 管理所有配置参数
- ✅ 无硬编码魔法数字
- ✅ 配置可追溯和可复现

### 日志记录
- ✅ 完整的日志记录
- ✅ 明确的错误处理
- ✅ 进度条显示

## 下一步

处理完数据后，可以：

1. 使用 `viplanner/models/temporal_encoder.py` 训练时序编码器
2. 使用 `viplanner/models/diffusion_unet.py` 训练扩散模型
3. 参考 `viplanner/engine/diffusion_offline_trainer.py` 进行完整训练

详见开发文档的 Phase 1 训练流程。
