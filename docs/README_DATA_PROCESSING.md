# Databet 数据处理系统

根据《开发文档(1).md》和《代码规范(1).md》编写的完整数据处理解决方案。

## 📋 概述

本系统提供了从原始 databet 数据到训练就绪数据的完整处理流程，包括：

1. **数据处理脚本** (`process_databet.py`)
   - 读取原始 RGB-D 和位姿数据
   - 生成时序滑动窗口样本
   - 坐标系转换（世界坐标系 → 局部坐标系）
   - 数据质量过滤和验证

2. **数据集加载器** (`viplanner/datasets/processed_databet_dataset.py`)
   - PyTorch Dataset 实现
   - 自动训练/验证集划分
   - 完整的维度和数值验证
   - 统计信息计算

3. **测试脚本** (`test_data_processing.py`)
   - 端到端测试
   - 数据有效性验证
   - 坐标变换验证

4. **使用指南** (`DATABET_PROCESSING_GUIDE.md`)
   - 详细的使用说明
   - 常见问题解答
   - 可视化示例

## 🚀 快速开始

### 1. 处理数据

```bash
# 使用默认配置处理所有序列
python process_databet.py
```

这将：
- 读取 `databet/` 目录下的所有序列
- 生成处理后的数据到 `processed_data/`
- 每个样本包含 K=8 帧历史和 L=20 点轨迹

### 2. 测试处理结果

```bash
# 运行完整测试
python test_data_processing.py
```

### 3. 在训练中使用

```python
from torch.utils.data import DataLoader
from viplanner.datasets.processed_databet_dataset import (
    ProcessedDatabetDataset,
    collate_fn,
)

# 创建数据集
dataset = ProcessedDatabetDataset(
    data_root="processed_data",
    split="train",
    train_ratio=0.8,
)

# 创建 DataLoader
loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
)

# 训练循环
for batch in loader:
    images = batch["images"]  # (B, K, C, H, W)
    trajectory = batch["trajectory"]  # (B, L, 3)
    # ... 训练代码
```

## 📊 数据格式

### 输入数据结构

```
databet/
└── 2026-04-22T01-15-55/
    ├── frame_0001/
    │   ├── camera.png      # RGB 图像
    │   ├── depth.png       # 深度图（毫米）
    │   └── data.json       # 位姿数据
    ├── frame_0002/
    └── ...
```

### 输出数据结构

```
processed_data/
└── 2026-04-22T01-15-55/
    ├── sample_000000.npz   # 样本数据
    ├── sample_000001.npz
    ├── ...
    └── metadata.json       # 序列元数据
```

### 样本数据格式

每个 `.npz` 文件包含：

| 字段 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `images` | (K, H, W, 3) | uint8 | 历史 RGB 图像 |
| `depths` | (K, H, W) | float32 | 历史深度图（米） |
| `trajectory` | (L, 3) | float32 | 未来轨迹（局部坐标系） |
| `current_position` | (3,) | float32 | 当前位置（世界坐标系） |
| `current_rotation` | (3,) | float32 | 当前旋转（欧拉角） |

### DataLoader 输出格式

| 字段 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `images` | (B, K, C, H, W) | float32 | RGB，范围 [0, 1] |
| `depths` | (B, K, 1, H, W) | float32 | 深度（米） |
| `trajectory` | (B, L, 3) | float32 | 轨迹（局部坐标系） |
| `valid_mask` | (B, K) | bool | 有效性掩码 |

## ⚙️ 配置参数

在 `process_databet.py` 中修改 `ProcessConfig`：

```python
@dataclass
class ProcessConfig:
    # 路径配置
    input_root: str = "databet"
    output_root: str = "processed_data"
    
    # 时序窗口
    hist_len: int = 8        # K: 历史窗口长度
    pred_len: int = 20       # L: 预测轨迹长度
    stride: int = 1          # 滑动窗口步长
    
    # 图像参数
    target_height: int = 224
    target_width: int = 224
    depth_scale: float = 1000.0  # 深度单位转换
    
    # 数据过滤
    min_sequence_len: int = 30
    max_translation: float = 0.5    # 相邻帧最大平移（米）
    max_rotation_deg: float = 30.0  # 相邻帧最大旋转（度）
```

## ✅ 符合规范

本系统严格遵循项目规范：

### 维度规范 ✓
- 所有函数都有明确的 shape 注释
- 使用语义化维度命名（B, K, L, C, H, W）
- 显式维度变换并附带注释

### 坐标系规范 ✓
- 统一转换到局部坐标系（Local Base Link）
- 明确的坐标变换函数
- 保留原始世界坐标系位姿

### 数据有效性 ✓
- NaN/Inf 检查
- 轨迹跳变检查
- 相邻帧位移和旋转过滤

### 配置管理 ✓
- 使用 dataclass 管理配置
- 无硬编码魔法数字
- 配置可追溯

### 日志记录 ✓
- 完整的日志记录
- 明确的错误处理
- 进度条显示

### 类型注解 ✓
- 所有公开函数都有类型注解
- 明确的输入输出类型
- 完整的 Docstring

## 📈 性能优化建议

### 处理速度
- 增大 `stride` 减少样本数量
- 减小图像尺寸
- 使用 SSD 存储

### 内存占用
- 减小 DataLoader 的 `batch_size`
- 减小 `num_workers`
- 使用更小的图像尺寸

### 磁盘空间
- 使用 `np.savez_compressed` 压缩存储
- 定期清理中间数据
- 考虑使用 HDF5 格式

## 🔍 调试技巧

### 启用详细日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 检查单个样本

```python
from viplanner.datasets.processed_databet_dataset import ProcessedDatabetDataset

dataset = ProcessedDatabetDataset("processed_data", split="train")
sample = dataset[0]

# 检查维度
for key, value in sample.items():
    if isinstance(value, torch.Tensor):
        print(f"{key}: {value.shape}, {value.dtype}")

# 检查数值范围
print(f"images: [{sample['images'].min():.3f}, {sample['images'].max():.3f}]")
print(f"depths: [{sample['depths'].min():.3f}, {sample['depths'].max():.3f}]")
```

### 可视化数据

参考 `DATABET_PROCESSING_GUIDE.md` 中的可视化示例。

## 📚 相关文档

- **开发文档(1).md** - 系统架构和设计原则
- **代码规范(1).md** - 编码规范和最佳实践
- **DATABET_PROCESSING_GUIDE.md** - 详细使用指南
- **TIVP_QUICKSTART.md** - 项目快速入门

## 🐛 常见问题

### Q: 处理后的数据在哪里？
A: 默认在 `processed_data/` 目录下，每个序列一个子目录。

### Q: 如何修改历史窗口长度？
A: 修改 `ProcessConfig` 中的 `hist_len` 参数。

### Q: 如何处理多个数据集？
A: 可以多次运行脚本，指定不同的输入输出目录，然后合并。

### Q: 数据集太大怎么办？
A: 增大 `stride` 参数，减少样本数量；或者只处理部分序列。

### Q: 如何验证数据正确性？
A: 运行 `python test_data_processing.py` 进行完整测试。

## 🤝 贡献

如需修改或扩展功能，请遵循：
1. 保持代码规范一致性
2. 添加完整的类型注解和 Docstring
3. 更新相关文档
4. 添加单元测试

## 📝 更新日志

### v1.0.0 (2026-04-24)
- ✨ 初始版本
- ✅ 完整的数据处理流程
- ✅ PyTorch Dataset 实现
- ✅ 完整的测试和文档

## 📧 联系方式

如有问题或建议，请参考项目主 README 或提交 Issue。

---

**注意**: 本系统是 Temporal-iSDF-VIPlanner (TiVP) 项目的一部分，专门用于处理 databet 格式的数据。
