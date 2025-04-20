import numpy as np
from debate_gym_env import DebateEnv
import torch

def run_random_agent(env, episodes=2):
    """
    Run a random agent in the debate environment
    
    Args:
        env: Debate gym environment
        episodes: Number of episodes to run
    """
    for episode in range(episodes):
        print(f"\n=== Episode {episode+1} ===")
        obs = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Random action: randomly weighted attention between agents
            action = np.random.random((env.num_agents, env.num_agents))
            
            # Step the environment
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            print(f"Round {env.current_round}")
            print(f"Reward: {reward:.2f}")
            print(f"Majority answer: {info['majority_answer']}")
            print(f"Correct answer: {info['correct_answer']}")
            print(f"Is correct: {info['is_correct']}")
            
        print(f"\nEpisode {episode+1} finished with total reward: {total_reward:.2f}")
        print(f"Final majority answer: {info['majority_answer']}, Correct: {info['correct_answer']}")
        print(f"Success: {info['is_correct']}")

def run_heuristic_agent(env, episodes=2):
    """
    Run a simple heuristic agent that increases weights between agents with similar answers
    
    Args:
        env: Debate gym environment
        episodes: Number of episodes to run
    """
    for episode in range(episodes):
        print(f"\n=== Episode {episode+1} ===")
        obs = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Heuristic: increase weights between agents with similar answers
            similarity_matrix = obs['similarity_matrix']
            action = np.zeros((env.num_agents, env.num_agents))
            
            # Set weights higher for similar agents
            for i in range(env.num_agents):
                for j in range(env.num_agents):
                    action[i, j] = 0.2 + 0.8 * similarity_matrix[i, j]
            
            # Step the environment
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            print(f"Round {env.current_round}")
            print(f"Reward: {reward:.2f}")
            print(f"Majority answer: {info['majority_answer']}")
            print(f"Is correct: {info['is_correct']}")
            
        print(f"\nEpisode {episode+1} finished with total reward: {total_reward:.2f}")
        print(f"Final majority answer: {info['majority_answer']}, Correct: {info['correct_answer']}")
        print(f"Success: {info['is_correct']}")

if __name__ == "__main__":
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Create the environment
    env = DebateEnv(
        num_agents=3,
        debate_rounds=3,
        model_name="Qwen/Qwen2.5-1.5B",
        device="cuda",
        base_threshold=0.0,
        final_threshold=0.8,
        use_embeddings=True
    )
    
    print("Running random agent...")
    run_random_agent(env, episodes=1)
    
    print("\nRunning heuristic agent...")
    run_heuristic_agent(env, episodes=1)
    
    env.close()
