# Debate Gym Environment - Qwen2.5-1.5B Version

This repository contains a gym environment that simulates a debate between multiple language model agents trying to solve problems collaboratively. The goal is to optimize the weights between debaters to maximize accuracy.

## Overview

The debate environment consists of multiple LLM agents that discuss a problem and try to reach the correct answer through deliberation. The environment enables:

- Simulating multi-agent debates with configurable parameters
- Using reinforcement learning to optimize agent communication patterns
- Testing different strategies for collaborative problem-solving

## Code Structure

The project consists of the following main components:

### Core Modules

1. **debate_gym_env.py**
   - Main implementation of the Gym environment compliant with OpenAI Gym standards
   - Contains the `DebateEnv` class which handles environment resets, steps, reward calculation
   - Implements customizable multi-agent interaction mechanisms
   - Defines observation and action spaces for reinforcement learning integration

2. **client.py**
   - Encapsulates interaction with language models through a unified interface
   - Contains the `QwenClient` class for working with Qwen2.5-1.5B models
   - Provides the `LlamaClient` alias for backward compatibility
   - Manages tokenization, generation, and response formatting

3. **gen_adaptive.py**
   - Contains the debate controller implementation that manages agent interaction rules
   - Provides answer parsing, prompt construction, and voting aggregation
   - Houses the `AdaptiveDebateController` class to handle weight calculations and dynamic trust threshold adjustments
   - Implements answer parsing and JSON response validation utilities

### Helper Modules

1. **run_debate.sh**
   - Provides a simple bash script to run the environment with default or custom parameters
   - Handles dependency checking and environment setup
   - Offers a convenient command-line interface for testing different configurations

2. **run_gym_example.py**
   - Demonstrates how to use the environment with simple heuristic and random agents
   - Provides example code for integrating the environment in custom applications

3. **debate_logger.py**
   - Implements comprehensive logging functionality for tracking debates
   - Records debate progress, agent answers, and performance metrics
   - Exports results to structured JSON files for analysis

### Data Structures

The project employs several key data structures:

1. **Observation Space**
   - Contains current round information (normalized to [0,1])
   - Similarity matrix between agents (based on embeddings and answer agreement)
   - Current agent answers (encoded as numeric values)
   - Answer history across all debate rounds

2. **Action Space**
   - Weight matrix between agents (n_agents × n_agents dimensions)
   - Each value is between 0.0 and 1.0, indicating communication intensity

3. **Reward System**
   - Correctness-based rewards (highest for correct majority answer)
   - Consensus-based rewards (encouraging agreement among agents)
   - Improvement-based rewards (for changing from wrong to right)

## Communication Mechanism

The debate environment implements a sophisticated communication pattern:

1. **Weight Calculation**
   - Based on answer similarity
   - Based on text embedding similarity
   - Dynamically adjusted trust threshold
   - Stability scores for consistent agents

2. **Message Passing**
   - Visibility determined by weight matrix
   - Critical importance markers ([Critical], [Reference], [Background])
   - Structured JSON responses with summaries and analyses

3. **Majority Decision**
   - Voting mechanism for final answer determination
   - Confidence levels and disagreement metrics
   - Optional outlier detection to filter extreme opinions

## Key Classes

### DebateEnv

The `DebateEnv` class extends `gym.Env` and provides the core simulation functionality:

```python
class DebateEnv(gym.Env):
    def __init__(self, num_agents=3, debate_rounds=5, ...):
        # Initializes the environment with specified parameters
        
    def reset(self):
        # Resets the environment and returns initial observation
        
    def step(self, action):
        # Takes a step using the provided action (weight matrix)
        # Returns observation, reward, done status, and info
        
    def _calculate_reward(self, prev_answers):
        # Calculates reward based on improvement and correctness
        
    def render(self, mode='human'):
        # Visualizes the current environment state
```

### AdaptiveDebateController

The `AdaptiveDebateController` manages the dynamics of agent interactions:

```python
class AdaptiveDebateController:
    def __init__(self, num_agents, total_rounds, ...):
        # Initializes the debate controller
        
    def compute_trust_threshold(self, current_round):
        # Dynamically calculates trust threshold
        
    def compute_weights(self, sim_matrix, current_round, answer_history):
        # Computes weight matrix based on similarity and debate stage
        
    def compute_stability(self, answer_history):
        # Calculates answer stability scores
        
    def construct_prompt(self, weight, agent_response, agent_idx):
        # Constructs prompt based on weight and agent response
```

### QwenClient

The `QwenClient` handles interaction with the Qwen2.5-1.5B model:

```python
class QwenClient:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B", device="cuda"):
        # Initializes the client with specified model
        
    def create_chat_completion(self, messages, max_tokens=None, temperature=0.8, stop=None):
        # Generates a response based on the provided messages
```

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
- Log the results to the logs directory

### Running with Custom Parameters

You can customize the debate environment with various parameters:

```bash
./run_debate.sh \
  --agents 4 \
  --rounds 5 \
  --model "Qwen/Qwen2.5-1.5B" \
  --device cuda \
  --episodes 10 \
  --base-threshold 0.1 \
  --final-threshold 0.9 \
  --outlier-threshold 0.6
```

For a full list of available parameters, run:

```bash
./run_debate.sh --help
```

## Using the Environment in Your Code

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

## Implementation Details

### Weight Calculation Logic

The weight calculation involves several factors:
1. **Similarity Matrix** - Calculates semantic similarity between agents using text embeddings
2. **Answer Consistency** - Checks if agents provide the same answers
3. **Dynamic Threshold** - Adjusts trust threshold during debate, initially encouraging diversity then consensus
4. **Stability Scores** - Rewards agents that maintain consistent opinions

### Agent Interaction Process

1. When the environment resets, all agents independently answer the question
2. In each step, agent visibility is determined by the weight matrix
3. Agents update their analysis and answers based on visible information
4. The environment calculates rewards, updates similarity matrix and answer history
5. The process ends when maximum rounds are reached or termination condition is met

## Data Structure

The environment uses question datasets in CSV format located in the `cais/data/test/` directory. Each CSV should have columns for:
- Question text
- Option A
- Option B
- Option C
- Option D
- Correct answer (A, B, C, or D)

## Extension and Customization

You can extend this environment in several ways:

1. **Replace Underlying Model** - Modify `client.py` to support other LLMs
2. **Customize Reward Function** - Modify the `_calculate_reward` method in `debate_gym_env.py`
3. **Add Evaluation Metrics** - Extend the `info` dictionary to include more performance metrics
4. **Adjust Communication Mechanism** - Modify the weight calculation logic in `AdaptiveDebateController`

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

# Test the trained agent
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
```

## Logging and Analysis

The environment includes comprehensive logging functionality through the `DebateLogger` class:

```python
from debate_gym_env import DebateEnv
from debate_logger import DebateLogger, run_debate_with_logging

# Initialize logger
params = {"agents": 3, "debate_rounds": 3, "model": "Qwen/Qwen2.5-1.5B"}
logger = DebateLogger(log_file="logs/debate_run.log", params=params)

# Create environment
env = DebateEnv(num_agents=3, debate_rounds=3)

# Run with logging
run_debate_with_logging(env, episodes=10, logger=logger)

# Save results
result_file = logger.save_results()
```

This will generate detailed JSON logs containing:
- Performance metrics for each debate
- Agent responses and answers for each round
- Similarity and weight matrices over time
- Overall accuracy and token usage statistics

## Troubleshooting

- **CUDA out of memory**: If you encounter memory issues, try reducing the number of agents or using a smaller embedding model, or use the `--device cpu` option
- **No valid answers**: Check that the model is properly responding in the required format
- **Model download issues**: Ensure you have a good internet connection as Qwen2.5-1.5B model will be downloaded from HuggingFace
- **JSON parsing errors**: The system tries to ensure proper JSON formatting but occasionally responses may be malformed. Check the model outputs if you encounter persistent errors

## License

This project is available under the MIT License.


