# TiVP (Temporal + iSDF + VIPlanner) 模块

TiVP 是 VIPlanner 的扩展模块，引入了时序信息处理、Diffusion 轨迹采样和 SDF 引导机制。

## 架构概览

```
viplanner/tivp/
├── __init__.py           # 模块导出
├── configs.py            # 配置类
├── models.py             # 神经网络模型
│   ├── TemporalEncoder   # 时序特征编码器
│   └── DiffusionUNet1D   # Diffusion 模型
├── samplers.py           # 采样器
│   ├── DDPMScheduler     # DDPM 调度器
│   └── GuidedDiffusionSampler  # 引导采样器
├── history.py            # 历史帧缓存
├── warm_start.py         # Warm-start 初始化
├── sdf_guidance.py       # SDF 引导
└── planner.py            # TiVPAlgo 规划器包装器
```

## 三阶段开发

### Phase 1: 最小闭环
- **目标**: 验证基础 Diffusion 推理流程
- **特性**:
  - RGB-D 多模态输入
  - Diffusion 轨迹采样
  - 无历史帧
  - 无 warm-start
  - 无 SDF 引导

### Phase 2: 时序增强
- **目标**: 提高轨迹连续性和稳定性
- **特性**:
  - 历史帧缓存（K=4）
  - Warm-start 初始化
  - 时间偏移和噪声混合

### Phase 3: SDF 引导
- **目标**: 降低碰撞风险
- **特性**:
  - 静态 SDF 查询
  - Diffusion 去噪过程中的梯度引导
  - 碰撞惩罚和安全边距

## 快速开始

### 安装依赖

```bash
pip install torch torchvision numpy matplotlib
```

### 测试推理

```bash
# 测试所有阶段
python test_tivp_inference.py

# 测试特定阶段
python test_tivp_inference.py --phase 1
python test_tivp_inference.py --phase 2
python test_tivp_inference.py --phase 3
```

### 训练模型

```bash
# Phase 1 训练
python train_tivp.py --phase 1 --epochs 100

# Phase 2 训练
python train_tivp.py --phase 2 --epochs 100

# Phase 3 训练
python train_tivp.py --phase 3 --epochs 100
```

## 使用示例

### 基础推理

```python
from viplanner.tivp import TiVPConfig, TiVPAlgo
import torch

# 创建配置（Phase 1）
cfg = TiVPConfig(phase=1, device="cuda")

# 初始化规划器
planner = TiVPAlgo(cfg)

# 准备输入
depth = torch.rand(1, 1, 64, 64, device="cuda")
rgb = torch.rand(1, 3, 64, 64, device="cuda")
goal_local = torch.tensor([[5.0, 0.0]], device="cuda")

# 规划轨迹
traj_local, info = planner.plan(depth, rgb, goal_local)

print(f"Trajectory shape: {traj_local.shape}")  # (1, 50, 2)
```

### Phase 2: 时序增强

```python
cfg = TiVPConfig(phase=2, device="cuda")
planner = TiVPAlgo(cfg)

# 多帧推理
for i in range(10):
    depth, rgb, goal_local = get_observation()
    cam_pos, cam_quat = get_camera_pose()
    
    traj_local, info = planner.plan(
        depth, rgb, goal_local,
        cam_pos, cam_quat
    )
    
    print(f"Frame {i}: History size = {len(planner.history_buffer)}")
```

### Phase 3: SDF 引导

```python
from viplanner.tivp.sdf_guidance import DummySDFModel

cfg = TiVPConfig(phase=3, device="cuda")
planner = TiVPAlgo(cfg)

# 设置 SDF 模型
sdf_model = DummySDFModel(obstacle_center=(5.0, 5.0), obstacle_radius=1.0)
planner.set_sdf_model(sdf_model)

# 规划（带 SDF 引导）
traj_local, info = planner.plan(depth, rgb, goal_local)
print(f"Guidance enabled: {info['guidance_enabled']}")
```

## 配置说明

### TiVPConfig

主配置类，包含所有子配置：

```python
@dataclass
class TiVPConfig:
    # 子配置
    diffusion: DiffusionConfig
    guidance: GuidanceConfig
    history: HistoryConfig
    warm_start: WarmStartConfig
    
    # 模型路径
    model_checkpoint: Optional[str] = None
    sdf_checkpoint: Optional[str] = None
    
    # 设备
    device: str = "cuda"
    
    # 图像参数
    img_height: int = 64
    img_width: int = 64
    
    # Phase 控制（1, 2, 3）
    phase: int = 1
```

### DiffusionConfig

Diffusion 采样配置：

```python
@dataclass
class DiffusionConfig:
    num_inference_steps: int = 20  # 采样步数
    horizon: int = 50              # 轨迹长度
    traj_dim: int = 2              # 轨迹维度
    context_dim: int = 512         # 上下文维度
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"
```

### GuidanceConfig

SDF 引导配置：

```python
@dataclass
class GuidanceConfig:
    enabled: bool = False
    guidance_scale: float = 10.0
    safety_margin: float = 0.3
    max_grad_norm: float = 10.0
    guidance_start_step: int = 0
```

## 与原版 VIPlanner 的兼容性

TiVPAlgo 提供与原版 VIPlannerAlgo 兼容的接口：

```python
# 目标坐标转换
goal_local = planner.goal_transformer(goal_world, cam_pos, cam_quat)

# 规划
traj_local, info = planner.plan(depth, rgb, goal_local)

# 路径坐标转换
traj_world = planner.path_transformer(traj_local, cam_pos, cam_quat)
```

## 代码规范

本模块严格遵循项目代码规范：

1. **张量维度注释**: 所有函数标注输入输出 shape
2. **梯度管理**: 推理期严格控制 autograd 图
3. **设备无关性**: 不硬编码设备名
4. **配置化**: 所有超参数进入配置系统
5. **防御式编程**: 对关键值做有限性检查

详见 `代码规范(1).md` 和 `推理代码规范.md`。

## 性能优化建议

1. **显存优化**:
   - 使用 `torch.no_grad()` 包裹推理代码
   - 及时 `detach()` 中间结果
   - 避免保留不必要的计算图

2. **速度优化**:
   - 减少 `num_inference_steps`（20 → 10）
   - 使用更小的 `horizon`（50 → 30）
   - 批处理多个目标点

3. **稳定性优化**:
   - 增加 `history.window_size`（4 → 8）
   - 调整 `warm_start.noise_ratio`（0.3 → 0.2）
   - 启用梯度裁剪

## 常见问题

### Q: 如何加载预训练模型？

```python
cfg = TiVPConfig(
    phase=3,
    model_checkpoint="checkpoints/model.pt",
)
planner = TiVPAlgo(cfg)
```

### Q: 如何自定义 SDF 模型？

```python
import torch.nn as nn

class MySDFModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 2)
        # 返回 SDF 值: (B, T)
        ...

sdf_model = MySDFModel()
planner.set_sdf_model(sdf_model)
```

### Q: 如何调整轨迹长度？

```python
cfg = TiVPConfig(phase=1)
cfg.diffusion.horizon = 100  # 默认 50
planner = TiVPAlgo(cfg)
```

## 下一步开发

- [ ] 完整数据加载管道
- [ ] 在线 iSDF 建图集成
- [ ] 多线程并发支持
- [ ] Isaac Sim 环境集成
- [ ] 真机部署优化
- [ ] 单元测试覆盖

## 参考文档

- `推理开发文档.md`: 详细开发指南
- `代码规范(1).md`: 代码规范
- `ARCHITECTURE.md`: 原版架构文档
