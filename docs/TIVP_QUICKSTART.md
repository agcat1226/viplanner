# TiVP 快速开始指南

5 分钟快速上手 TiVP 模块。

## 1. 验证安装

```bash
# 检查 Python 和 PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 检查 TiVP 模块
python -c "from viplanner.tivp import TiVPAlgo; print('TiVP imported successfully!')"
```

## 2. 运行测试（3 分钟）

### 测试 Phase 1（最小闭环）

```bash
python test_tivp_inference.py --phase 1
```

预期输出：
```
Testing Phase 1: Minimal Closed-Loop
Planning trajectory...
Trajectory shape: torch.Size([1, 50, 2])
Phase 1 test completed successfully!
```

生成文件：`phase1_trajectory.png`

### 测试所有阶段

```bash
python test_tivp_inference.py
```

生成文件：
- `phase1_trajectory.png`
- `phase2_trajectory.png`
- `phase3_trajectory.png`

## 3. 基础推理示例

创建文件 `my_first_tivp.py`:

```python
import torch
from viplanner.tivp import TiVPConfig, TiVPAlgo

# 1. 创建配置
cfg = TiVPConfig(phase=1, device="cuda")

# 2. 初始化规划器
planner = TiVPAlgo(cfg)

# 3. 准备输入（模拟数据）
depth = torch.rand(1, 1, 64, 64, device="cuda")
rgb = torch.rand(1, 3, 64, 64, device="cuda")
goal = torch.tensor([[5.0, 0.0]], device="cuda")  # 5米前方

# 4. 规划轨迹
traj, info = planner.plan(depth, rgb, goal)

# 5. 查看结果
print(f"轨迹形状: {traj.shape}")  # (1, 50, 2)
print(f"起点: {traj[0, 0]}")
print(f"终点: {traj[0, -1]}")
print(f"信息: {info}")
```

运行：
```bash
python my_first_tivp.py
```

## 4. 训练测试（使用虚拟数据）

```bash
# Phase 1 训练（10 个 epoch，快速测试）
python train_tivp.py --phase 1 --epochs 10 --batch-size 16 --use-dummy

# 查看检查点
ls checkpoints/
```

## 5. 三个阶段对比

### Phase 1: 最小闭环
```python
cfg = TiVPConfig(phase=1)
# ✅ Diffusion 采样
# ❌ 无历史帧
# ❌ 无 warm-start
# ❌ 无 SDF 引导
```

### Phase 2: 时序增强
```python
cfg = TiVPConfig(phase=2)
# ✅ Diffusion 采样
# ✅ 历史帧缓存
# ✅ Warm-start
# ❌ 无 SDF 引导
```

### Phase 3: SDF 引导
```python
cfg = TiVPConfig(phase=3)
# ✅ Diffusion 采样
# ✅ 历史帧缓存
# ✅ Warm-start
# ✅ SDF 引导
```

## 6. 常用配置调整

### 调整轨迹长度
```python
cfg = TiVPConfig(phase=1)
cfg.diffusion.horizon = 100  # 默认 50
```

### 调整采样步数
```python
cfg.diffusion.num_inference_steps = 10  # 默认 20，减少可提速
```

### 调整历史窗口
```python
cfg.history.window_size = 8  # 默认 4
```

### 调整引导强度
```python
cfg.guidance.guidance_scale = 20.0  # 默认 10.0
```

## 7. 在线推理循环示例

```python
from viplanner.tivp import TiVPConfig, TiVPAlgo
import torch

# 初始化
cfg = TiVPConfig(phase=2, device="cuda")
planner = TiVPAlgo(cfg)

# 推理循环
for step in range(10):
    print(f"\n=== Step {step} ===")
    
    # 模拟获取观测
    depth = torch.rand(1, 1, 64, 64, device="cuda")
    rgb = torch.rand(1, 3, 64, 64, device="cuda")
    goal = torch.tensor([[5.0, 0.0]], device="cuda")
    
    # 规划
    traj, info = planner.plan(depth, rgb, goal)
    
    print(f"轨迹: {traj.shape}")
    print(f"历史帧数: {len(planner.history_buffer)}")
    
    # 模拟执行轨迹的前几步
    # execute_trajectory(traj[0, :5, :])
```

## 8. 可视化轨迹

```python
import matplotlib.pyplot as plt

# 规划轨迹
traj, info = planner.plan(depth, rgb, goal)
traj_np = traj[0].cpu().numpy()  # (50, 2)

# 绘制
plt.figure(figsize=(8, 8))
plt.plot(traj_np[:, 0], traj_np[:, 1], 'b-', linewidth=2)
plt.plot(traj_np[0, 0], traj_np[0, 1], 'go', markersize=10, label='Start')
plt.plot(traj_np[-1, 0], traj_np[-1, 1], 'ro', markersize=10, label='End')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.savefig('my_trajectory.png')
print("保存到 my_trajectory.png")
```

## 9. 故障排查

### CUDA 不可用
```bash
# 使用 CPU
python test_tivp_inference.py --device cpu
```

### 显存不足
```python
cfg = TiVPConfig(phase=1)
cfg.diffusion.num_inference_steps = 10  # 减少步数
cfg.diffusion.horizon = 30              # 减少轨迹长度
```

### 导入错误
```bash
# 确保在项目根目录
cd /path/to/viplanner
python test_tivp_inference.py
```

## 10. 下一步

- 📖 阅读完整文档：`TIVP_USAGE.md`
- 🏗️ 了解项目结构：`TIVP_PROJECT_STRUCTURE.md`
- 📝 查看开发文档：`推理开发文档.md`
- 🔧 学习代码规范：`代码规范(1).md`

## 常见问题

**Q: 如何切换到 CPU？**
```python
cfg = TiVPConfig(phase=1, device="cpu")
```

**Q: 如何加载预训练模型？**
```python
cfg = TiVPConfig(
    phase=3,
    model_checkpoint="checkpoints/model.pt"
)
```

**Q: 如何调整批次大小？**
```bash
python train_tivp.py --batch-size 8
```

**Q: 测试脚本在哪里？**
```bash
python test_tivp_inference.py --help
```

## 完整示例

```python
#!/usr/bin/env python3
"""完整的 TiVP 推理示例"""

import torch
from viplanner.tivp import TiVPConfig, TiVPAlgo
from viplanner.tivp.sdf_guidance import DummySDFModel

def main():
    # 1. 配置
    cfg = TiVPConfig(
        phase=3,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # 2. 初始化
    planner = TiVPAlgo(cfg)
    
    # 3. 设置 SDF（Phase 3）
    if cfg.phase == 3:
        sdf = DummySDFModel(obstacle_center=(5.0, 5.0), obstacle_radius=1.0)
        planner.set_sdf_model(sdf)
    
    # 4. 推理循环
    for i in range(5):
        print(f"\n=== Frame {i+1} ===")
        
        # 模拟观测
        depth = torch.rand(1, 1, 64, 64, device=cfg.device)
        rgb = torch.rand(1, 3, 64, 64, device=cfg.device)
        goal = torch.tensor([[5.0, 0.0]], device=cfg.device)
        
        # 规划
        traj, info = planner.plan(depth, rgb, goal)
        
        print(f"Trajectory: {traj.shape}")
        print(f"Info: {info}")
    
    print("\n✅ 完成!")

if __name__ == "__main__":
    main()
```

保存为 `complete_example.py` 并运行：
```bash
python complete_example.py
```

---

🎉 恭喜！你已经掌握了 TiVP 的基础使用。
