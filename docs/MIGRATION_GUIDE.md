# Migration Guide: From Original to Modular Architecture

本指南帮助你从原始的 `deprecated/` 实现迁移到新的模块化架构。

## 主要变化

### 1. 文件结构

**原始结构**:
```
deprecated/viplanner/
├── train.py                    # 训练脚本
├── utils/
│   ├── trainer.py             # 训练器 (1300+ 行)
│   └── dataset.py             # 数据集 (1300+ 行)
├── plannernet/
│   ├── PlannerNet.py
│   └── autoencoder.py
└── config/
    └── learning_cfg.py
```

**新结构**:
```
viplanner/
├── models/                     # 模型定义
│   ├── plannernet.py
│   └── autoencoder.py
├── losses/                     # 损失函数
│   └── trajectory_loss.py
├── trainers/                   # 训练器
│   └── viplanner_trainer.py
├── datasets/                   # 数据集
│   ├── planner_dataset.py
│   └── preprocessing.py
├── configs/                    # 配置
│   └── train_config.py
└── utils/                      # 工具
    ├── optimizer.py
    └── logging.py
```

### 2. 导入变化

**原始代码**:
```python
from viplanner.config import TrainCfg, DataCfg
from viplanner.utils.trainer import Trainer
from viplanner.plannernet import AutoEncoder, DualAutoEncoder
```

**新代码**:
```python
from viplanner.configs import TrainConfig, DataConfig
from viplanner.trainers import ViPlannerTrainer
from viplanner.models import AutoEncoder, DualAutoEncoder
```

### 3. 配置类名称变化

| 原始 | 新 |
|------|-----|
| `TrainCfg` | `TrainConfig` |
| `DataCfg` | `DataConfig` |

### 4. 训练器使用变化

**原始代码**:
```python
cfg = TrainCfg(
    sem=True,
    env_list=["env1", "env2"],
    test_env_id=1,
)
trainer = Trainer(cfg)
trainer.train()
trainer.test()
trainer.save_config()
```

**新代码**:
```python
cfg = TrainConfig(
    sem=True,
    env_list=["env1", "env2"],
    test_env_id=1,
)
trainer = ViPlannerTrainer(cfg)
trainer.setup()  # 新增：显式设置
trainer.train(train_loader, val_loader)
trainer.test(test_loader)
trainer.save_config()
```

## 模块化的优势

### 1. 独立的模型定义

**好处**: 可以单独测试和使用模型

```python
# 只导入模型，不需要训练器
from viplanner.models import DualAutoEncoder

model = DualAutoEncoder(in_channel=16, knodes=5)
# 在推理或其他场景中使用
```

### 2. 可插拔的损失函数

**好处**: 轻松切换或组合不同的损失函数

```python
from viplanner.losses import TrajectoryLoss

# 自定义损失权重
loss_fn = TrajectoryLoss(
    w_obs=0.3,
    w_height=1.2,
    w_motion=1.5,
    w_goal=4.0,
)
```

### 3. 独立的数据处理

**好处**: 数据预处理可以独立开发和测试

```python
from viplanner.datasets import PlannerData
from viplanner.datasets.preprocessing import ImageAugmentation

# 自定义数据增强
augmentor = ImageAugmentation({
    "depth_salt_pepper": 0.02,
    "depth_gaussian": 0.001,
})
```

### 4. 灵活的训练器

**好处**: 可以继承和扩展训练器

```python
from viplanner.trainers import ViPlannerTrainer

class CustomTrainer(ViPlannerTrainer):
    def train_epoch(self, train_loader, epoch):
        # 自定义训练逻辑
        pass
```

## 迁移步骤

### 步骤 1: 更新导入

在你的代码中查找并替换：
```bash
# 配置
TrainCfg → TrainConfig
DataCfg → DataConfig

# 训练器
from viplanner.utils.trainer import Trainer
→ from viplanner.trainers import ViPlannerTrainer

# 模型
from viplanner.plannernet import ...
→ from viplanner.models import ...
```

### 步骤 2: 更新配置

配置参数基本保持不变，只需更新类名：

```python
# 原始
cfg = TrainCfg(
    sem=True,
    epochs=100,
    batch_size=64,
)

# 新
cfg = TrainConfig(
    sem=True,
    epochs=100,
    batch_size=64,
)
```

### 步骤 3: 更新训练代码

添加 `setup()` 调用：

```python
# 原始
trainer = Trainer(cfg)
trainer.train()

# 新
trainer = ViPlannerTrainer(cfg)
trainer.setup()  # 新增
trainer.train(train_loader, val_loader)
```

### 步骤 4: 测试

运行测试确保一切正常：

```bash
python tests/test_models.py
python tests/test_configs.py
```

## 常见问题

### Q: 原始代码还能用吗？

A: 可以。原始代码保留在 `deprecated/` 文件夹中作为参考。

### Q: 如何使用原始的完整数据加载功能？

A: 目前新架构中的数据加载是简化版本。完整的数据生成逻辑（包括图形构建、轨迹优化等）需要从 `deprecated/viplanner/utils/dataset.py` 迁移。这是下一步的工作。

### Q: 轨迹成本优化在哪里？

A: 轨迹成本优化模块（`traj_cost_opt/`）暂时保留在 `deprecated/` 中。需要集成到新的 `losses/` 模块中。

### Q: 如何添加新的模型架构？

A: 在 `viplanner/models/` 中创建新文件，继承 `nn.Module`，然后在 `__init__.py` 中导出。

### Q: 如何自定义训练循环？

A: 继承 `ViPlannerTrainer` 并重写 `train_epoch()` 或 `validate_epoch()` 方法。

## 下一步工作

为了完成迁移，还需要：

1. **完整的数据加载**: 将 `PlannerDataGenerator` 的完整实现从 `deprecated/` 迁移
2. **轨迹优化集成**: 将 `traj_cost_opt/` 集成到 `losses/` 模块
3. **可视化工具**: 迁移轨迹和图像可视化功能
4. **评估指标**: 添加评估指标计算
5. **单元测试**: 为所有模块添加完整的测试

## 示例对比

### 完整训练脚本对比

**原始 (`deprecated/viplanner/train.py`)**:
```python
from viplanner.config import DataCfg, TrainCfg
from viplanner.utils.trainer import Trainer

if __name__ == "__main__":
    env_list = ["2azQ1b91cZZ", "JeFG25nYj2p"]
    cfg = TrainCfg(
        sem=True,
        env_list=env_list,
        test_env_id=1,
        data_cfg=DataCfg(max_goal_distance=10.0),
    )
    trainer = Trainer(cfg)
    trainer.train()
    trainer.test()
    trainer.save_config()
```

**新 (`train.py`)**:
```python
from viplanner.configs import DataConfig, TrainConfig
from viplanner.trainers import ViPlannerTrainer

if __name__ == "__main__":
    env_list = ["2azQ1b91cZZ", "JeFG25nYj2p"]
    cfg = TrainConfig(
        sem=True,
        env_list=env_list,
        test_env_id=1,
        data_cfg=DataConfig(max_goal_distance=10.0),
    )
    trainer = ViPlannerTrainer(cfg)
    trainer.setup()
    # 注意：数据加载需要实现
    # trainer.train(train_loader, val_loader)
    # trainer.test(test_loader)
    trainer.save_config()
```

## 总结

新的模块化架构提供了：

✅ 更清晰的代码组织  
✅ 更好的可测试性  
✅ 更容易扩展  
✅ 更好的代码复用  
✅ 更容易维护  

虽然需要一些迁移工作，但长期来看会大大提高开发效率。
