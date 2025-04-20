```markdown
# 多智能体通信环境改造编程作业

## 作业目标
将现有代码中的多LLM智能体交互逻辑改造成符合OpenAI Gym规范的强化学习环境，实现可量化的通信决策机制。

---

## 环境规范

### 观察空间（Observation）
```python
Dict({
    "agent_outputs": Box(low=-inf, high=inf, shape=(n_llm_agent, output_dim)),
    "token_usage": Box(low=0, high=inf, shape=(n_llm_agent,))
})
```
- 包含两个核心要素：
  1. 所有智能体当前轮次的输出（向量形式）
  2. 各智能体累计的token消耗量

### 动作空间（Action）
```python
MultiBinary(n_agents)  # 每个两个LLM Agent之间作为一个Agent，管理一个二值通信决策
```
- 每个元素对应一个智能体：
  - 0: 禁止两个LLM Agent当前轮次通信
  - 1: 允许两个LLM Agent当前轮次通信

### 奖励函数（Reward）
```
reward = group_vote_correctness - communication_cost
```
- 计算要素：
  1. 群体投票正确性指标（1=正确，0=错误）
  2. 通信成本 = sum(各智能体token消耗) * cost_factor（建议0.01）

### 环境终止条件
- 达到最大回合数（建议100轮）
- 群体投票达成连续正确决策（建议连续5次正确）

---

## 实现要求
1. 继承`gym.Env`并实现标准接口
2. 实现通信决策机制：
   - 以两个LLM agent之间能否通信作为一个RL agent
   - 允许通信的两个LLM agent可访问对面的信息（双向）
   - 禁止通信的两个LLM agent不可以访问对面的信息
3. 维护token计数器：
   - 每次通信增加对应智能体的token计数
4. 实现群体投票机制：
   - 基于多数决原则判定最终决策

---

## 示例代码框架
```python
import gym
from gym import spaces
import numpy as np

class MultiAgentCommEnv(gym.Env):
    def __init__(self, n_llm_agents=3, max_steps=100):
        super().__init__()
        self.n_llm_agents = n_llm_agents
        self.n_agents = self.n_llm_agents * (self.n_llm_agents-1) // 2
        self.max_steps = max_steps
        
        # 定义观察空间
        self.observation_space = spaces.Dict({
            "agent_outputs": spaces.Box(low=-np.inf, high=np.inf, shape=(n_llm_agents, 64)),
            "token_usage": spaces.Box(low=0, high=np.inf, shape=(n_llm_agents,))
        }) # 需要 nomic 等方式将输出字符串转化为数值
        
        # 定义动作空间
        self.action_space = spaces.MultiBinary(self.n_agents)
        
    def reset(self):
        # 初始化环境状态
        self.current_step = 0
        self.token_counts = np.zeros(self.n_agents)
        self.consecutive_correct = 0
        return self._get_obs()
    
    def step(self, actions):
        # 执行通信决策
        communication_cost = 0
        for i, action in enumerate(actions):
            if action == 1:
                self._perform_communication(i)
                communication_cost += self.token_counts[i]
        
        # 获取群体投票结果
        group_decision, is_correct = self._get_group_decision()
        
        # 计算奖励
        reward = is_correct - 0.01 * communication_cost
        
        # 更新终止条件
        self.current_step += 1
        done = (self.current_step >= self.max_steps) or \
               (self.consecutive_correct >= 5)
        
        return self._get_obs(), reward, done, {}
    
    # 以下方法需自行实现
    def _perform_communication(self, agent_id):
        """处理智能体通信逻辑"""
        raise NotImplementedError
    
    def _get_group_decision(self):
        """获取群体投票结果"""
        raise NotImplementedError

    # 其他能够达成最终需求的方法请酌情实现
```

---

## 提交要求
1. 完整可运行的Gym环境代码
2. 测试用例（至少包含）：
   - 基础通信场景测试
   - 奖励计算正确性验证
   - 终止条件触发测试
3. 实验报告（PDF格式）：
   - 环境设计思路
   - 通信机制实现方案
   - 测试结果分析


## 附加说明
1. 建议参考OpenAI Gym官方文档：[https://gymnasium.farama.org](https://gymnasium.farama.org)
2. 注意处理智能体间的信息同步问题
3. 通信内容编码方式可自行设计
4. 最终决策任务建议使用简单任务（如代码中的GSM8K和MMLU数据集）
5. Agent可以使用Mock agent（例如收到任何prompt只会回复ok）
6. 截止日期：2025年4月25日 23:59
```

请根据具体需求调整以下参数：
1. 智能体数量（n_agents）
2. 最大回合数（max_steps）
3. 通信成本系数（cost_factor）
4. 终止条件阈值
5. 具体决策任务设置

建议在环境实现完成后使用gym.make()进行验证：
```python
env = gym.make('MultiAgentComm-v0')
obs = env.reset()
```