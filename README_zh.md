# 辩论模拟环境 (Debate Gym Environment) - Qwen2.5-1.5B版本

本仓库包含一个模拟多智能体辩论的gym环境，多个语言模型代理通过协作尝试解决问题。环境的目标是优化代理之间的权重以最大化准确率。

## 概述

辩论环境由多个LLM代理组成，它们讨论问题并尝试通过协商达成正确答案。该环境可以：

- 使用可配置参数模拟多代理辩论
- 使用强化学习优化代理通信模式
- 测试不同的协作解决问题策略

## 代码结构

项目由以下主要组件构成：

### 核心模块

1. **debate_gym_env.py**
   - 主要的Gym环境实现，符合OpenAI Gym标准接口
   - 包含`DebateEnv`类，处理环境重置、步进、奖励计算等
   - 实现了可自定义参数的多智能体交互机制

2. **client.py**
   - 封装与语言模型的交互接口
   - 包含`QwenClient`类，用于Qwen2.5-1.5B模型的调用
   - 提供了兼容性的`LlamaClient`别名，简化迁移

3. **gen_adaptive.py**
   - 包含辩论控制器的实现，管理智能体间的交互规则
   - 提供答案解析、提示构建和投票聚合功能
   - 包含`AdaptiveDebateController`类，处理权重计算和信任阈值动态调整

### 辅助功能

1. **run_temp.py**
   - 提供快速运行环境的脚本
   - 设置基本参数并初始化环境
   - 简化测试和演示流程

### 数据结构

项目使用以下主要数据结构：

1. **观察空间 (Observation Space)**
   - 包含当前轮次信息
   - 智能体之间的相似度矩阵
   - 智能体的当前答案
   - 答案历史记录

2. **动作空间 (Action Space)**
   - 代理之间的权重矩阵 (n_agents x n_agents 维度)
   - 每个值介于0.0到1.0之间

3. **奖励机制**
   - 基于正确性的奖励
   - 基于共识度的奖励
   - 基于改进程度的奖励

## 通信机制

辩论环境实现了一个复杂的通信模式：

1. **权重计算**
   - 基于答案相似性
   - 基于文本嵌入相似性
   - 动态调整的信任阈值

2. **消息传递**
   - 根据权重矩阵确定可见性
   - 添加关键性标记（[Critical]、[Reference]、[Background]）
   - 构建结构化JSON响应

3. **多数决策**
   - 使用投票机制确定最终答案
   - 提供信心水平和分歧度量

## 快速开始

### 前提条件

确保已安装以下依赖：

```bash
pip install gym numpy pandas tqdm sentence-transformers torch transformers
```

### 运行示例

要快速测试环境，运行：

```bash
# 使Qwen2.5-1.5B模型
./run_debate.sh --model "Qwen/Qwen2.5-1.5B" --device cuda

# 或使用更简单的命令（使用默认设置）
./run_debate.sh
```

这将：
- 创建一个有3个代理和3轮辩论的环境
- 使用启发式代理（基于答案相似性的权重）运行一个回合

## 使用环境

在您自己的代码中创建和使用环境：

```python
from debate_gym_env import DebateEnv
import numpy as np

# 创建环境
env = DebateEnv(
    num_agents=3,             # 辩论者数量
    debate_rounds=3,          # 辩论轮数
    model_name="Qwen/Qwen2.5-1.5B",  # 模型名称
    device="cuda",            # 设备（cuda或cpu）
    base_threshold=0.0,       # 初始信任阈值
    final_threshold=0.8,      # 最终信任阈值
    use_embeddings=True       # 使用嵌入进行代理相似性
)

# 重置环境
obs = env.reset()

# 运行单个回合
done = False
while not done:
    # 创建动作（代理之间的权重）
    # 例如：随机权重
    action = np.random.random((env.num_agents, env.num_agents))
    
    # 执行环境步骤
    obs, reward, done, info = env.step(action)
    
    # 可视化当前状态
    env.render()

# 清理资源
env.close()
```

## 环境参数

`DebateEnv`类接受以下参数：

- `num_agents` (int): 辩论代理数量（默认：3）
- `debate_rounds` (int): 辩论轮数（默认：5）
- `model_name` (str): 模型名称（默认："Qwen/Qwen2.5-1.5B"）
- `device` (str): 设备（默认："cuda"）
- `base_threshold` (float): 初始信任阈值（默认：0.0）
- `final_threshold` (float): 最终信任阈值（默认：0.8）
- `outlier_threshold` (float): 异常值检测阈值（默认：0.5）
- `min_weight` (float): 代理交互最小权重（默认：0.1）
- `max_weight` (float): 代理交互最大权重（默认：0.9）
- `use_outlier` (bool): 是否使用异常值检测（默认：True）
- `equality_weight` (float): 嵌入和答案相似性之间的权重（默认：0.5）
- `seed` (int): 随机种子（默认：42）
- `use_embeddings` (bool): 是否使用嵌入进行代理相似性（默认：True）

## 实现细节

### 权重计算逻辑

权重计算涉及多个因素：
1. **相似度矩阵** - 使用文本嵌入计算智能体间的语义相似度
2. **答案一致性** - 检查智能体是否提供相同答案
3. **动态阈值** - 随辩论进行调整信任阈值，初始鼓励多样性，后期鼓励共识
4. **稳定性得分** - 奖励保持一致观点的智能体

### 智能体交互过程

1. 环境重置时，所有智能体独立回答问题
2. 每个步骤中，根据权重矩阵确定智能体可见性
3. 智能体根据可见信息更新自己的分析和答案
4. 环境计算奖励，更新相似度矩阵和答案历史
5. 达到最大轮数或满足终止条件时结束

## 数据结构

环境使用位于`cais/data/test/`目录中CSV格式的问题数据集。每个CSV应包含以下列：
- 问题文本
- 选项A
- 选项B
- 选项C
- 选项D
- 正确答案（A、B、C或D）

## 扩展和定制

您可以通过以下方式扩展此环境：

1. **替换底层模型** - 修改`client.py`以支持其他LLM
2. **自定义奖励函数** - 在`debate_gym_env.py`的`_calculate_reward`方法中修改
3. **新增评估指标** - 扩展`info`字典以包含更多性能指标
4. **调整通信机制** - 修改`AdaptiveDebateController`中的权重计算逻辑

## 高级使用

对于更高级的使用，例如在此环境上训练RL代理，您可以与Stable Baselines3或RLlib等框架集成：

```python
from stable_baselines3 import PPO
from debate_gym_env import DebateEnv

# 创建环境
env = DebateEnv(num_agents=3, debate_rounds=3)

# 创建并训练代理
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# 保存模型
model.save("ppo_debate")
```

## 故障排除

- **CUDA内存不足**：如果遇到内存问题，尝试减少代理数量或使用更小的嵌入模型，也可以使用`--device cpu`选项
- **没有有效答案**：检查模型是否正确响应所需格式
- **模型下载问题**：确保您有良好的网络连接，因为Qwen2.5-1.5B模型会从HuggingFace下载
