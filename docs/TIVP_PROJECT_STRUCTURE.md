# TiVP 项目结构说明

本文档说明 TiVP 模块在项目中的组织结构和文件职责。

## 完整项目结构

```
viplanner/                          # 主项目目录
│
├── viplanner/                      # 原版 VIPlanner 模块（保持不变）
│   ├── __init__.py
│   ├── models/                     # 原版模型
│   │   ├── plannernet.py
│   │   └── autoencoder.py
│   ├── losses/                     # 原版损失函数
│   │   └── trajectory_loss.py
│   ├── trainers/                   # 原版训练器
│   │   └── viplanner_trainer.py
│   ├── datasets/                   # 原版数据集
│   │   ├── planner_dataset.py
│   │   ├── carla_dataset.py
│   │   └── preprocessing.py
│   ├── configs/                    # 原版配置
│   │   ├── train_config.py
│   │   └── carla_config.py
│   ├── traj_cost/                  # 轨迹代价计算
│   │   ├── cost_map.py
│   │   ├── traj_cost.py
│   │   └── traj_opt.py
│   └── utils/                      # 工具函数
│       ├── optimizer.py
│       └── logging.py
│
├── viplanner/tivp/                 # 🆕 TiVP 新模块（新增）
│   ├── __init__.py                 # 模块导出
│   ├── README.md                   # 模块文档
│   ├── configs.py                  # TiVP 配置类
│   ├── models.py                   # TiVP 模型
│   │   ├── TemporalEncoder         # 时序编码器
│   │   └── DiffusionUNet1D         # Diffusion 模型
│   ├── samplers.py                 # 采样器
│   │   ├── DDPMScheduler           # DDPM 调度器
│   │   └── GuidedDiffusionSampler  # 引导采样器
│   ├── history.py                  # 历史帧缓存
│   ├── warm_start.py               # Warm-start 初始化
│   ├── sdf_guidance.py             # SDF 引导
│   ├── planner.py                  # TiVPAlgo 规划器
│   └── dataset.py                  # TiVP 数据集
│
├── train.py                        # 原版训练脚本（保持不变）
├── train_carla.py                  # 原版 Carla 训练（保持不变）
├── train_tivp.py                   # 🆕 TiVP 训练脚本
├── test_tivp_inference.py          # 🆕 TiVP 推理测试
│
├── deprecated/                     # 原版实现（参考）
│
├── README.md                       # 项目主文档
├── ARCHITECTURE.md                 # 原版架构文档
├── TIVP_USAGE.md                   # 🆕 TiVP 使用指南
├── TIVP_PROJECT_STRUCTURE.md       # 🆕 本文档
├── 推理开发文档.md                  # TiVP 开发文档
├── 代码规范(1).md                   # 代码规范
├── 推理代码规范.md                  # 推理代码规范
│
├── pyproject.toml                  # 项目配置
├── run_tests.sh                    # 测试脚本
│
├── checkpoints/                    # 🆕 模型检查点目录
│   ├── phase1_model.pt
│   ├── phase2_model.pt
│   └── phase3_model.pt
│
├── test_outputs/                   # 🆕 测试输出目录
│   ├── phase1_trajectory.png
│   ├── phase2_trajectory.png
│   └── phase3_trajectory.png
│
└── databet/                        # 数据集目录
```

## 模块职责划分

### 原版 VIPlanner 模块（不改动）

保持原有功能，继续支持：
- 传统编码器-解码器训练
- Carla 数据集训练
- 原有的轨迹优化方法

### TiVP 新模块（新增）

独立的模块，实现新架构：
- 时序信息处理
- Diffusion 轨迹采样
- SDF 引导机制
- 与原版兼容的接口

## 文件详细说明

### 核心模块文件

#### `viplanner/tivp/configs.py`
- `TiVPConfig`: 主配置类
- `DiffusionConfig`: Diffusion 配置
- `GuidanceConfig`: SDF 引导配置
- `HistoryConfig`: 历史帧配置
- `WarmStartConfig`: Warm-start 配置

#### `viplanner/tivp/models.py`
- `TemporalEncoder`: 时序特征编码器
  - 输入: (B, K, C, H, W)
  - 输出: (B, D_ctx)
- `DiffusionUNet1D`: 1D UNet for trajectory
  - 输入: x_t (B, T, D), timesteps (B,), context (B, D_ctx)
  - 输出: eps_pred (B, T, D)

#### `viplanner/tivp/samplers.py`
- `DDPMScheduler`: DDPM 噪声调度
- `GuidedDiffusionSampler`: 带引导的采样器

#### `viplanner/tivp/history.py`
- `HistoryBuffer`: 历史帧缓存管理
  - 维护固定大小滑动窗口
  - 存储 depth、RGB、pose 历史

#### `viplanner/tivp/warm_start.py`
- `WarmStarter`: Warm-start 初始化
  - 时间偏移
  - 噪声混合

#### `viplanner/tivp/sdf_guidance.py`
- `SDFGuidance`: SDF 物理引导
  - 计算碰撞惩罚
  - 梯度引导
- `DummySDFModel`: 虚拟 SDF（测试用）

#### `viplanner/tivp/planner.py`
- `TiVPAlgo`: 规划器包装器
  - 兼容原版接口
  - 集成所有 TiVP 组件

#### `viplanner/tivp/dataset.py`
- `TiVPDataset`: 训练数据集
- `DummyTiVPDataset`: 虚拟数据集（测试用）
- `create_dataloader`: 数据加载器工厂函数

### 脚本文件

#### `train_tivp.py`
TiVP 训练脚本，支持三个阶段：
```bash
python train_tivp.py --phase 1 --epochs 100
python train_tivp.py --phase 2 --epochs 100
python train_tivp.py --phase 3 --epochs 100
```

#### `test_tivp_inference.py`
TiVP 推理测试脚本：
```bash
python test_tivp_inference.py --phase 1
python test_tivp_inference.py --phase 2
python test_tivp_inference.py --phase 3
python test_tivp_inference.py  # 测试所有阶段
```

### 文档文件

#### `TIVP_USAGE.md`
完整的使用指南，包括：
- 环境准备
- 快速测试
- 训练流程
- 推理部署
- Isaac Sim 集成
- 性能调优
- 故障排查

#### `viplanner/tivp/README.md`
模块级文档，包括：
- 架构概览
- 快速开始
- 使用示例
- 配置说明
- 常见问题

## 开发工作流

### Phase 1 开发流程

1. **测试基础功能**
   ```bash
   python test_tivp_inference.py --phase 1
   ```

2. **准备训练数据**
   - 实现 `TiVPDataset._load_samples()`
   - 或使用 `DummyTiVPDataset` 测试

3. **训练模型**
   ```bash
   python train_tivp.py --phase 1 --epochs 100
   ```

4. **验证推理**
   ```bash
   python test_tivp_inference.py --phase 1
   ```

### Phase 2 开发流程

1. **在 Phase 1 基础上启用历史**
   ```python
   cfg = TiVPConfig(phase=2)
   ```

2. **测试时序功能**
   ```bash
   python test_tivp_inference.py --phase 2
   ```

3. **训练时序模型**
   ```bash
   python train_tivp.py --phase 2 --epochs 100
   ```

### Phase 3 开发流程

1. **准备 SDF 模型**
   - 训练或加载 iSDF 模型
   - 或使用 `DummySDFModel` 测试

2. **测试引导功能**
   ```bash
   python test_tivp_inference.py --phase 3
   ```

3. **训练引导模型**
   ```bash
   python train_tivp.py --phase 3 --epochs 100
   ```

## 与原版的集成

### 替换规划器

```python
# 原版代码
# from viplanner.algo import VIPlannerAlgo
# planner = VIPlannerAlgo(config)

# 替换为 TiVP
from viplanner.tivp import TiVPConfig, TiVPAlgo

cfg = TiVPConfig(phase=3, model_checkpoint="checkpoints/phase3_model.pt")
planner = TiVPAlgo(cfg)

# 接口完全兼容
goal_local = planner.goal_transformer(goal_world, cam_pos, cam_quat)
traj_local, info = planner.plan(depth, rgb, goal_local)
traj_world = planner.path_transformer(traj_local, cam_pos, cam_quat)
```

### 共存使用

原版和 TiVP 可以共存：

```python
# 同时导入
from viplanner.trainers import ViPlannerTrainer  # 原版
from viplanner.tivp import TiVPAlgo              # TiVP

# 根据需要选择
if use_tivp:
    planner = TiVPAlgo(tivp_config)
else:
    planner = ViPlannerTrainer(original_config)
```

## 目录权限和组织

### 新增目录

创建以下目录用于存储输出：

```bash
mkdir -p checkpoints
mkdir -p test_outputs
mkdir -p logs
```

### 检查点命名规范

```
checkpoints/
├── phase1_epoch_100.pt
├── phase2_epoch_100.pt
├── phase3_epoch_100.pt
├── phase3_best.pt
└── phase3_latest.pt
```

### 日志组织

```
logs/
├── phase1_train.log
├── phase2_train.log
├── phase3_train.log
└── inference_test.log
```

## 依赖关系图

```
TiVPAlgo
├── TemporalEncoder (models.py)
├── DiffusionUNet1D (models.py)
├── GuidedDiffusionSampler (samplers.py)
│   ├── DDPMScheduler (samplers.py)
│   └── SDFGuidance (sdf_guidance.py)
├── HistoryBuffer (history.py)
└── WarmStarter (warm_start.py)
```

## 测试覆盖

### 单元测试（待实现）

```
tests/tivp/
├── test_configs.py
├── test_models.py
├── test_samplers.py
├── test_history.py
├── test_warm_start.py
├── test_sdf_guidance.py
└── test_planner.py
```

### 集成测试

- `test_tivp_inference.py`: 端到端推理测试
- `train_tivp.py`: 训练流程测试

## 总结

TiVP 模块采用独立的文件夹结构，与原版 VIPlanner 完全解耦：

✅ **优点**:
- 不影响原有代码
- 清晰的模块边界
- 易于维护和扩展
- 可以独立测试

✅ **兼容性**:
- 提供兼容接口
- 可以无缝替换
- 支持共存使用

✅ **可扩展性**:
- 三阶段渐进开发
- 模块化设计
- 易于添加新功能
