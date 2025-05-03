# 基于粒子群优化（PSO）的CartPole控制方案

## 项目简介
本仓库提供了一种基于粒子群优化算法（PSO）解决OpenAI Gym CartPole-v1控制问题的完整实现。包含以下核心功能：
-  **无梯度优化**：通过群体智能搜索最优策略参数
-  **参数保存**：自动保存训练得到的最佳策略参数
-  **实时渲染**：可视化展示训练后的控制效果

## 快速开始

### 环境配置
```bash
# 安装依赖
pip install -r requirements.txt
```
```bash
# 运行训练脚本
python pso_cartpole.py
```
```bash
# 运行渲染脚本
python render_cartpole.py
```

## 关键参数(pso_cartpole)
- **粒子数**：`num_particles`，默认30
- **迭代次数**：`generations`，默认50
- **惯性权重**：`w`，默认0.729
