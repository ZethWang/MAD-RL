# Debate Gym Environment - Qwen2.5-1.5B Version

This repository contains a gym environment that simulates a debate between multiple language model agents trying to solve problems collaboratively. The goal is to optimize the weights between debaters to maximize accuracy.

## Overview

The debate environment consists of multiple LLM agents that discuss a problem and try to reach the correct answer through deliberation. The environment enables:

- Simulating multi-agent debates with configurable parameters
- Using reinforcement learning to optimize agent communication patterns
- Testing different strategies for collaborative problem-solving

## Quick Start

### Prerequisites

Ensure you have the following dependencies installed:

```bash
pip install gym numpy pandas tqdm sentence-transformers torch transformers
```

### Running the Example

To quickly test the environment, run:

```bash
# Using Qwen2.5-1.5B model
./run_debate.sh --model "Qwen/Qwen2.5-1.5B" --device cuda

# Or with default settings
./run_debate.sh
```

This will:
- Create a debate environment with 3 agents and 3 debate rounds
- Run one episode using a heuristic agent (weights based on answer similarity)

## Using the Environment

To create and use the environment in your own code:

```python
from debate_gym_env import DebateEnv
import numpy as np

# Create the environment
env = DebateEnv(
    num_agents=3,             # Number of debaters
    debate_rounds=3,          # Rounds of debate
    model_name="Qwen/Qwen2.5-1.5B",  # Model name
    device="cuda",            # Device (cuda or cpu)
    base_threshold=0.0,       # Initial trust threshold
    final_threshold=0.8,      # Final trust threshold
    use_embeddings=True       # Use embeddings for agent similarity
)

# Reset the environment
obs = env.reset()

# Run a single episode
done = False
while not done:
    # Create an action (weights between agents)
    # For example: random weights
    action = np.random.random((env.num_agents, env.num_agents))
    
    # Step the environment
    obs, reward, done, info = env.step(action)
    
    # Visualize the current state
    env.render()

# Clean up
env.close()
```

## Environment Parameters

The `DebateEnv` class accepts the following parameters:

- `num_agents` (int): Number of debate agents (default: 3)
- `debate_rounds` (int): Number of debate rounds (default: 5)
- `model_name` (str): Model name (default: "Qwen/Qwen2.5-1.5B")
- `device` (str): Device (default: "cuda")
- `base_threshold` (float): Initial trust threshold (default: 0.0)
- `final_threshold` (float): Final trust threshold (default: 0.8)
- `outlier_threshold` (float): Threshold for outlier detection (default: 0.5)
- `min_weight` (float): Minimum weight for agent interaction (default: 0.1)
- `max_weight` (float): Maximum weight for agent interaction (default: 0.9)
- `use_outlier` (bool): Whether to use outlier detection (default: True)
- `equality_weight` (float): Weight between embeddings and answer similarity (default: 0.5)
- `seed` (int): Random seed (default: 42)
- `use_embeddings` (bool): Whether to use embeddings for agent similarity (default: True)

## Data Structure

The environment uses question datasets in CSV format located in the `cais/data/test/` directory. Each CSV should have columns for:
- Question text
- Option A
- Option B
- Option C
- Option D
- Correct answer (A, B, C, or D)

## Advanced Usage

For more advanced usage, such as training RL agents on this environment, you can integrate it with frameworks like Stable Baselines3 or RLlib:

```python
from stable_baselines3 import PPO
from debate_gym_env import DebateEnv

# Create environment
env = DebateEnv(num_agents=3, debate_rounds=3)

# Create and train an agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Save the model
model.save("ppo_debate")
```

## Troubleshooting

- **CUDA out of memory**: If you encounter memory issues, try reducing the number of agents or using a smaller embedding model, or use the `--device cpu` option
- **No valid answers**: Check that the model is properly responding in the required format
- **Model download issues**: Ensure you have a good internet connection as Qwen2.5-1.5B model will be downloaded from HuggingFace
