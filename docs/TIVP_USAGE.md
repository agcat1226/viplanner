# TiVP 使用指南

本文档提供 TiVP 模块的详细使用说明，包括训练、推理和集成到 Isaac Sim 的完整流程。

## 目录

1. [环境准备](#环境准备)
2. [快速测试](#快速测试)
3. [训练流程](#训练流程)
4. [推理部署](#推理部署)
5. [Isaac Sim 集成](#isaac-sim-集成)
6. [性能调优](#性能调优)
7. [故障排查](#故障排查)

---

## 环境准备

### 基础依赖

```bash
# 安装 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install numpy opencv-python pillow scikit-image scipy tqdm pyyaml matplotlib
```

### 可选依赖

```bash
# Weights & Biases (日志记录)
pip install wandb

# Open3D (可视化)
pip install open3d
```

### 验证安装

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

from viplanner.tivp import TiVPConfig, TiVPAlgo
print("TiVP module imported successfully!")
```

---

## 快速测试

### 测试 Phase 1（最小闭环）

```bash
python test_tivp_inference.py --phase 1 --device cuda
```

预期输出：
```
Testing Phase 1: Minimal Closed-Loop
Planning trajectory...
Trajectory shape: torch.Size([1, 50, 2])
Info: {'phase': 1, 'history_enabled': False, ...}
Phase 1 test completed successfully!
```

### 测试所有阶段

```bash
python test_tivp_inference.py
```

这将依次测试 Phase 1、2、3，并生成可视化图像：
- `phase1_trajectory.png`
- `phase2_trajectory.png`
- `phase3_trajectory.png`

---

## 训练流程

### Phase 1: 基础训练

```bash
python train_tivp.py --phase 1 --epochs 100 --batch-size 32
```

### Phase 2: 时序训练

在 Phase 1 的基础上，启用历史帧：

```bash
python train_tivp.py --phase 2 --epochs 100 --batch-size 32
```

### Phase 3: 引导训练

在 Phase 2 的基础上，加入 SDF 引导：

```bash
python train_tivp.py --phase 3 --epochs 100 --batch-size 32
```

### 自定义训练脚本

```python
from viplanner.tivp import TiVPConfig
from train_tivp import TiVPTrainer

# 创建配置
cfg = TiVPConfig(
    phase=3,
    device="cuda",
)

# 自定义参数
cfg.diffusion.num_inference_steps = 30
cfg.guidance.guidance_scale = 15.0
cfg.history.window_size = 8

# 创建训练器
trainer = TiVPTrainer(cfg)

# 训练
# trainer.train(train_loader, num_epochs=100)
```

---

## 推理部署

### 基础推理

```python
import torch
from viplanner.tivp import TiVPConfig, TiVPAlgo

# 初始化
cfg = TiVPConfig(
    phase=3,
    model_checkpoint="checkpoints/model.pt",
    device="cuda",
)
planner = TiVPAlgo(cfg)

# 准备输入
depth = torch.rand(1, 1, 64, 64, device="cuda")
rgb = torch.rand(1, 3, 64, 64, device="cuda")
goal_local = torch.tensor([[5.0, 0.0]], device="cuda")

# 规划
traj_local, info = planner.plan(depth, rgb, goal_local)
```

### 批量推理

```python
# 批量处理多个目标
batch_size = 8
depth = torch.rand(batch_size, 1, 64, 64, device="cuda")
rgb = torch.rand(batch_size, 3, 64, 64, device="cuda")
goals = torch.rand(batch_size, 2, device="cuda") * 10  # 随机目标

traj_local, info = planner.plan(depth, rgb, goals)
print(f"Batch trajectory shape: {traj_local.shape}")  # (8, 50, 2)
```

### 在线推理循环

```python
import numpy as np

# 初始化
planner = TiVPAlgo(TiVPConfig(phase=2, device="cuda"))

# 推理循环
for step in range(100):
    # 获取观测
    depth, rgb = get_camera_observation()
    cam_pos, cam_quat = get_camera_pose()
    goal_world = get_navigation_goal()
    
    # 坐标转换
    goal_local = planner.goal_transformer(goal_world, cam_pos, cam_quat)
    
    # 规划
    traj_local, info = planner.plan(depth, rgb, goal_local, cam_pos, cam_quat)
    
    # 转换回世界坐标
    traj_world = planner.path_transformer(traj_local, cam_pos, cam_quat)
    
    # 执行第一段轨迹
    execute_trajectory(traj_world[0, :5, :])  # 执行前 5 个点
    
    # 每 10 步重置一次（可选）
    if step % 10 == 0:
        planner.reset()
```

---

## Isaac Sim 集成

### 替换原版 VIPlannerAlgo

```python
# 原版代码
# from viplanner.algo import VIPlannerAlgo
# planner = VIPlannerAlgo(config)

# 替换为 TiVP
from viplanner.tivp import TiVPConfig, TiVPAlgo

cfg = TiVPConfig(
    phase=3,
    model_checkpoint="path/to/checkpoint.pt",
    device="cuda",
)
planner = TiVPAlgo(cfg)

# 接口完全兼容
goal_local = planner.goal_transformer(goal_world, cam_pos, cam_quat)
traj_local, info = planner.plan(depth, rgb, goal_local)
traj_world = planner.path_transformer(traj_local, cam_pos, cam_quat)
```

### Isaac Sim 环境脚本示例

```python
import omni
from viplanner.tivp import TiVPConfig, TiVPAlgo

class TiVPIsaacEnv:
    def __init__(self):
        # 初始化 Isaac Sim 环境
        self.setup_isaac_sim()
        
        # 初始化 TiVP 规划器
        cfg = TiVPConfig(phase=3, device="cuda")
        self.planner = TiVPAlgo(cfg)
    
    def step(self):
        # 从 Isaac Sim 获取观测
        depth = self.get_depth_image()
        rgb = self.get_rgb_image()
        cam_pos, cam_quat = self.get_camera_pose()
        goal_world = self.get_goal()
        
        # 坐标转换
        goal_local = self.planner.goal_transformer(goal_world, cam_pos, cam_quat)
        
        # 规划
        traj_local, info = self.planner.plan(depth, rgb, goal_local, cam_pos, cam_quat)
        
        # 转换并执行
        traj_world = self.planner.path_transformer(traj_local, cam_pos, cam_quat)
        self.execute_path(traj_world)
        
        return traj_world, info
```

---

## 性能调优

### 显存优化

```python
# 1. 减少批次大小
cfg.batch_size = 16  # 默认 32

# 2. 减少采样步数
cfg.diffusion.num_inference_steps = 10  # 默认 20

# 3. 减少轨迹长度
cfg.diffusion.horizon = 30  # 默认 50

# 4. 减少历史窗口
cfg.history.window_size = 2  # 默认 4
```

### 速度优化

```python
# 1. 使用 TensorRT（需要额外配置）
# 2. 使用混合精度
with torch.cuda.amp.autocast():
    traj_local, info = planner.plan(depth, rgb, goal_local)

# 3. 预热模型
for _ in range(5):
    planner.plan(depth, rgb, goal_local)
```

### 质量优化

```python
# 1. 增加采样步数
cfg.diffusion.num_inference_steps = 50

# 2. 增加历史窗口
cfg.history.window_size = 8

# 3. 调整引导尺度
cfg.guidance.guidance_scale = 20.0

# 4. 减少 warm-start 噪声
cfg.warm_start.noise_ratio = 0.1
```

---

## 故障排查

### 问题 1: CUDA Out of Memory

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
```python
# 1. 减小批次大小
cfg.batch_size = 8

# 2. 清理缓存
torch.cuda.empty_cache()

# 3. 使用 CPU
cfg.device = "cpu"
```

### 问题 2: 轨迹不连续

**症状**: 相邻帧的轨迹跳变严重

**解决方案**:
```python
# 1. 启用 Phase 2
cfg.phase = 2

# 2. 增加历史窗口
cfg.history.window_size = 8

# 3. 减少 warm-start 噪声
cfg.warm_start.noise_ratio = 0.2
```

### 问题 3: 轨迹碰撞障碍物

**症状**: 规划的轨迹穿过障碍物

**解决方案**:
```python
# 1. 启用 Phase 3
cfg.phase = 3

# 2. 增加引导尺度
cfg.guidance.guidance_scale = 20.0

# 3. 增加安全边距
cfg.guidance.safety_margin = 0.5

# 4. 设置正确的 SDF 模型
planner.set_sdf_model(your_sdf_model)
```

### 问题 4: NaN/Inf 错误

**症状**: `RuntimeError: Function 'XXX' returned nan values`

**解决方案**:
```python
# 1. 检查输入数据
assert torch.isfinite(depth).all()
assert torch.isfinite(rgb).all()

# 2. 启用梯度裁剪
cfg.guidance.max_grad_norm = 5.0

# 3. 检查 SDF 模型输出
sdf_values = sdf_model(test_input)
assert torch.isfinite(sdf_values).all()
```

### 问题 5: 模型加载失败

**症状**: `KeyError: 'encoder'` 或 `RuntimeError: Error(s) in loading state_dict`

**解决方案**:
```python
# 检查检查点格式
checkpoint = torch.load("checkpoint.pt")
print(checkpoint.keys())

# 确保包含 'encoder' 和 'diffusion_model'
# 如果格式不匹配，手动加载：
planner.encoder.load_state_dict(checkpoint["encoder"])
planner.diffusion_model.load_state_dict(checkpoint["diffusion_model"])
```

---

## 高级用法

### 自定义 SDF 模型

```python
import torch.nn as nn

class CustomSDFModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 你的 SDF 网络架构
        self.mlp = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 轨迹点。Shape: (B, T, 2)
        Returns:
            sdf: SDF 值。Shape: (B, T)
        """
        b, t, d = x.shape
        x_flat = x.view(b * t, d)
        sdf_flat = self.mlp(x_flat).squeeze(-1)
        sdf = sdf_flat.view(b, t)
        return sdf

# 使用
sdf_model = CustomSDFModel()
planner.set_sdf_model(sdf_model)
```

### 多目标规划

```python
# 规划到多个候选目标
goals = torch.tensor([
    [5.0, 0.0],
    [5.0, 2.0],
    [5.0, -2.0],
], device="cuda")

# 批量规划
depth_batch = depth.repeat(3, 1, 1, 1)
rgb_batch = rgb.repeat(3, 1, 1, 1)

trajs, info = planner.plan(depth_batch, rgb_batch, goals)

# 选择最优轨迹（例如，最短路径）
costs = compute_trajectory_costs(trajs)
best_idx = costs.argmin()
best_traj = trajs[best_idx]
```

---

## 总结

TiVP 模块提供了灵活的三阶段开发流程，从最小闭环到完整的时序 SDF 引导系统。通过本文档，你应该能够：

1. ✅ 快速测试和验证 TiVP 功能
2. ✅ 训练自己的 TiVP 模型
3. ✅ 部署到推理环境
4. ✅ 集成到 Isaac Sim
5. ✅ 调优性能和质量
6. ✅ 排查常见问题

如有问题，请参考：
- `viplanner/tivp/README.md`: 模块文档
- `推理开发文档.md`: 详细开发指南
- `代码规范(1).md`: 代码规范
