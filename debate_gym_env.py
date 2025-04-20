import gym
import numpy as np
import random
import pandas as pd
import copy
import json
import re
import time
import os
from typing import Dict, List, Any, Tuple, Optional, Union
from glob import glob
from sentence_transformers import SentenceTransformer
from client import QwenClient
from gen_adaptive import (
    AdaptiveDebateController,
    SYSTEM_MESSAGE,
    generate_answer,
    parse_answer,
    validate_json_response,
    find_majority,
    construct_debate_message,
)

class DebateEnv(gym.Env):
    """
    A gym environment for the adaptive debate system using Qwen2.5-1.5B model.
    
    This environment simulates a debate between multiple agents trying to solve
    a problem. The goal of the agent controlling this environment is to optimize 
    the weights between debaters to maximize accuracy.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        num_agents: int = 3,
        debate_rounds: int = 5,
        base_threshold: float = 0.0,
        final_threshold: float = 0.8,
        outlier_threshold: float = 0.5,
        min_weight: float = 0.1,
        max_weight: float = 0.9,
        use_outlier: bool = True,
        equality_weight: float = 0.5,
        model_name: str = "Qwen/Qwen2.5-1.5B",
        device: str = "cuda",
        seed: int = 42,
        use_embeddings: bool = True,
    ):
        super().__init__()
        
        self.num_agents = num_agents
        self.debate_rounds = debate_rounds
        self.equality_weight = equality_weight
        self.use_embeddings = use_embeddings
        self.model_name = model_name
        self.device = device
        
        # Set random seeds
        np.random.seed(seed)
        random.seed(seed)
        
        # Initialize controller
        self.controller = AdaptiveDebateController(
            num_agents=num_agents,
            total_rounds=debate_rounds,
            base_threshold=base_threshold,
            final_threshold=final_threshold,
            similarity_outlier_threshold=outlier_threshold,
            min_weight=min_weight,
            max_weight=max_weight,
            use_outlier=use_outlier,
        )
        
        # Initialize Qwen client (we'll use a single model instance for all agents)
        self.llama_client = QwenClient(model_name=model_name, device=device)
        
        # Load embedding model
        if self.use_embeddings:
            self.embedding_model = SentenceTransformer(
                model_name_or_path = "nomic-ai/nomic-embed-text-v1", 
                trust_remote_code=True,
                device = device
            )
        
        # Create data directory if it doesn't exist
        os.makedirs("cais/data/test", exist_ok=True)
        
        # Load problems or create sample if none exist
        self.tasks = glob("cais/data/test/*.csv")
        if not self.tasks:
            # Create a sample CSV if no data exists
            sample_df = pd.DataFrame({
                'question': [
                    "What is the capital of France?",
                    "What is the chemical symbol for gold?",
                    "Which planet is closest to the Sun?",
                    "What is 2+2?",
                    "Who wrote Romeo and Juliet?"
                ],
                'A': ['Paris', 'Au', 'Earth', '3', 'Charles Dickens'],
                'B': ['London', 'Ag', 'Venus', '4', 'William Shakespeare'],
                'C': ['Berlin', 'Fe', 'Mercury', '5', 'Jane Austen'],
                'D': ['Madrid', 'Cu', 'Mars', '6', 'Mark Twain'],
                'answer': ['A', 'A', 'C', 'B', 'B']
            })
            sample_path = "cais/data/test/sample_questions.csv"
            sample_df.to_csv(sample_path, index=False)
            self.tasks = [sample_path]
            
        self.dfs = [pd.read_csv(task) for task in self.tasks]
        
        # Define action and observation spaces

        # Action space: a matrix representing the interaction weights between agents.
        # Each value is a float in [0, 1], indicating how much agent i trusts agent j.
        # The shape is (num_agents x num_agents), allowing full pairwise communication.
        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0, 
            shape=(self.num_agents, self.num_agents),
            dtype=np.float32
        )

        # Observation space: provides the agent with state information including:
        # - Current debate round (normalized to [0, 1])
        # - Similarity matrix (how similar each agent's response is to others')
        # - Each agent's current answer (encoded as 0=A, 1=B, etc.)
        # - Historical answers over all debate rounds
        embedding_dim = 100  # Set embedding dimension for representation, optional use

        self.observation_space = gym.spaces.Dict({
            'round': gym.spaces.Box(
                low=0, high=1, shape=(1,), dtype=np.float32
            ),  # Normalized progress of current debate round
            
            'similarity_matrix': gym.spaces.Box(
                low=-1, high=1, shape=(num_agents, num_agents), dtype=np.float32
            ),  # Cosine similarity or answer agreement between agents
            
            'agent_answers': gym.spaces.Box(
                low=0, high=4, shape=(num_agents,), dtype=np.float32
            ),  # Each agent's selected answer for current round (0=A, ..., 3=D)
            
            'answer_history': gym.spaces.Box(
                low=0, high=4, shape=(debate_rounds, num_agents), dtype=np.float32
            )  # Answer history for each round and agent (used for tracking stability)
        })

        # -----------------------------
        # Internal environment state
        # -----------------------------

        self.current_round = 0  # Tracks the current debate round (starts at 0)

        self.agent_contexts = None  # Holds dialogue history for each agent
        self.question = None        # Current question being debated
        self.answer = None          # Ground-truth answer for the current question
        self.options = None         # Multiple choice options for the question
        self.sim_matrix = None      # Similarity matrix between agent responses
        self.text_answer_this_round = None  # Answers given by each agent in the current round
        self.answer_history = []    # Full history of answers over all rounds

        self.reset_needed = True    # Indicates if environment needs to be reset before next step

                
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset the environment and return initial observation"""
        
        # Select a random problem
        df = random.choice(self.dfs)
        idx = random.randint(0, len(df)-1)
        
        # Extract question, options, and answer
        self.question = df.iloc[idx, 0]
        a = df.iloc[idx, 1]
        b = df.iloc[idx, 2]
        c = df.iloc[idx, 3]
        d = df.iloc[idx, 4]
        self.answer = df.iloc[idx, 5]
        self.options = [a, b, c, d]
        
        formatted_question = f"{self.question}:A) {a}, B) {b}, C) {c}, D) {d}."
        
        # Initialize agent contexts
        self.agent_contexts = [
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": f"Can you answer the following question as accurately as possible? {formatted_question}?"}
            ] for _ in range(self.num_agents)
        ]
        
        # Reset environment state
        self.current_round = 0
        self.text_answer_this_round = [None] * self.num_agents
        self.answer_history = []
        self.sim_matrix = np.ones((self.num_agents, self.num_agents)) * 0.5  # Initialize with medium similarity
        self.reset_needed = False
        
        # Generate initial responses
        for i, agent_context in enumerate(self.agent_contexts):
            completion = generate_answer(agent_context, self.llama_client)
            response = completion["choices"][0]["message"]["content"]
            json_response = validate_json_response(response)
            assistant_message = json_response['independent_analysis']
            
            agent_context.append({"role": "assistant", "content": assistant_message})
            self.agent_contexts[i] = agent_context
            
            text_answer = json_response.get('answer', '')
            if text_answer:
                answer_idx = ord(text_answer) - ord('A') if len(text_answer) > 0 else -1
                self.text_answer_this_round[i] = answer_idx
            else:
                self.text_answer_this_round[i] = -1
        
        # Create embeddings and similarity matrix
        if self.use_embeddings:
            context = ['search_document: ' + s for s in 
                       [ctx[-1]["content"] for ctx in self.agent_contexts]]
            embeddings = self.embedding_model.encode(context, normalize_embeddings=True)
            self.sim_matrix = np.inner(embeddings, embeddings)
        
        self.answer_history.append(copy.deepcopy(self.text_answer_this_round))
        
        # Convert answers to one-hot encoding
        answer_history_array = np.zeros((self.debate_rounds, self.num_agents))
        answer_history_array[0] = np.array(self.text_answer_this_round)
        
        # Return observation
        return {
            'round': np.array([self.current_round / (self.debate_rounds - 1)], dtype=np.float32),
            'similarity_matrix': self.sim_matrix.astype(np.float32),
            'agent_answers': np.array(self.text_answer_this_round, dtype=np.float32),
            'answer_history': answer_history_array.astype(np.float32)
        }
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """
        Take a step in the environment using the provided action.
        
        Args:
            action: Weight matrix for agent interactions
            
        Returns:
            observation: Next observation
            reward: Reward for this step
            done: Whether episode is done
            info: Additional information
        """
        if self.reset_needed:
            raise RuntimeError("Environment needs to be reset before stepping.")
        
        # Ensure action is properly formatted
        weights_matrix = np.clip(action, 0.0, 1.0)
        
        # Normalize weights to sum to 1 for each agent
        row_sums = weights_matrix.sum(axis=1, keepdims=True)
        weights_matrix = np.where(row_sums > 0, weights_matrix / row_sums, 0.0)
        
        # Increment round
        self.current_round += 1
        done = self.current_round >= self.debate_rounds
        
        # Store previous answers for reward calculation
        prev_answers = copy.deepcopy(self.text_answer_this_round)
        
        # Update agent contexts and generate new responses
        for i, agent_context in enumerate(self.agent_contexts):
            # Create debate message using weights
            messages = construct_debate_message(
                self.agent_contexts,
                3 * self.current_round - 1,
                weights_matrix[i],
                agent_indices=list(range(self.num_agents)),
                controller=self.controller
            )
            
            agent_context.extend(messages)
            
            # Generate response
            completion = generate_answer(agent_context[-2:], self.llama_client)
            response = completion["choices"][0]["message"]["content"]
            json_response = validate_json_response(response)
            assistant_message = json_response['independent_analysis']
            
            agent_context.append({"role": "assistant", "content": assistant_message})
            self.agent_contexts[i] = agent_context
            
            # Extract answer
            text_answer = json_response.get('answer', '')
            if text_answer:
                answer_idx = ord(text_answer) - ord('A') if len(text_answer) > 0 else -1
                self.text_answer_this_round[i] = answer_idx
            else:
                self.text_answer_this_round[i] = -1
        
        # Update similarity matrix
        if self.use_embeddings:
            context = ['search_document: ' + s for s in 
                      [ctx[-1]["content"] for ctx in self.agent_contexts]]
            embeddings = self.embedding_model.encode(context, normalize_embeddings=True)
            self.sim_matrix = np.inner(embeddings, embeddings)
        
        # Calculate answer similarity
        answers = self.text_answer_this_round
        answer_sim = np.zeros((len(answers), len(answers)))
        for i in range(len(answers)):
            for j in range(len(answers)):
                answer_sim[i,j] = 1.0 if answers[i] == answers[j] else 0.0
                
        # Combine similarity metrics
        alpha = self.equality_weight
        self.sim_matrix = (1 - alpha) * self.sim_matrix + alpha * answer_sim
        
        # Update answer history
        self.answer_history.append(copy.deepcopy(self.text_answer_this_round))
        
        # Calculate reward
        reward = self._calculate_reward(prev_answers)
        
        # Check if done
        if done:
            self.reset_needed = True
        
        # Create observation - Fixed answer history array creation
        answer_history_array = np.zeros((self.debate_rounds, self.num_agents))
        for i in range(len(self.answer_history)):
            if i < self.debate_rounds:  # Ensure we don't exceed array bounds
                answers = self.answer_history[i]
                answer_history_array[i] = np.array(answers)
        
        observation = {
            'round': np.array([self.current_round / (self.debate_rounds - 1)], dtype=np.float32),
            'similarity_matrix': self.sim_matrix.astype(np.float32),
            'agent_answers': np.array(self.text_answer_this_round, dtype=np.float32),
            'answer_history': answer_history_array.astype(np.float32)
        }
        
        # Additional info
        info = {
            'majority_answer': find_majority([chr(int(a) + ord('A')) for a in self.text_answer_this_round if a >= 0]),
            'correct_answer': self.answer,
            'is_correct': find_majority([chr(int(a) + ord('A')) for a in self.text_answer_this_round if a >= 0]) == self.answer,
            'question': self.question,
            'options': self.options
        }
        
        return observation, reward, done, info
    
    def _calculate_reward(self, prev_answers: List[int]) -> float:
        """Calculate the reward based on improvement and correctness"""
        # Calculate majority answer
        answer_counts = {}
        for ans in self.text_answer_this_round:
            if ans >= 0:  # Skip invalid answers
                letter = chr(int(ans) + ord('A'))
                answer_counts[letter] = answer_counts.get(letter, 0) + 1
        
        majority_answer = max(answer_counts.items(), key=lambda x: x[1])[0] if answer_counts else 'E'
        
        # Reward for correctness
        correctness_reward = 1.0 if majority_answer == self.answer else -0.1
        
        # Reward for consensus (agreement among agents)
        max_count = max(answer_counts.values()) if answer_counts else 0
        consensus_reward = max_count / self.num_agents
        
        # Reward for improvement from previous round
        prev_answer_counts = {}
        for ans in prev_answers:
            if ans >= 0:
                letter = chr(int(ans) + ord('A'))
                prev_answer_counts[letter] = prev_answer_counts.get(letter, 0) + 1
        
        prev_majority = max(prev_answer_counts.items(), key=lambda x: x[1])[0] if prev_answer_counts else 'E'
        improvement_reward = 0.5 if prev_majority != self.answer and majority_answer == self.answer else 0.0
        
        # Combine rewards
        total_reward = correctness_reward + 0.3 * consensus_reward + improvement_reward
        
        return total_reward
    
    def render(self, mode='human'):
        """Render the environment"""
        if mode != 'human':
            raise NotImplementedError("Only human mode is supported")
        
        print(f"\n--- Round {self.current_round} ---")
        print(f"Question: {self.question}")
        print(f"Options: A) {self.options[0]}, B) {self.options[1]}, C) {self.options[2]}, D) {self.options[3]}")
        print(f"Correct Answer: {self.answer}")
        
        print("\nCurrent Answers:")
        for i, ans in enumerate(self.text_answer_this_round):
            if ans >= 0:
                letter = chr(int(ans) + ord('A'))
                print(f"Agent {i}: {letter}")
            else:
                print(f"Agent {i}: Invalid")
        
        majority = find_majority([chr(int(a) + ord('A')) for a in self.text_answer_this_round if a >= 0])
        print(f"Majority Answer: {majority} {'✓' if majority == self.answer else '✗'}")
        
        print("\nSimilarity Matrix:")
        print(np.round(self.sim_matrix, 2))
    
    def close(self):
        """Clean up resources"""
        pass


# Test code to verify the environment works
if __name__ == "__main__":
    env = DebateEnv(
        num_agents=3,
        debate_rounds=3,
        port_start=8080
    )
    
    print("Testing DebateEnv:")
    obs = env.reset()
    print(f"Initial observation shape: {obs['similarity_matrix'].shape}")
    
    for i in range(3):
        # Random action (weight matrix)
        action = np.random.random((env.num_agents, env.num_agents))
        
        # Take a step
        obs, reward, done, info = env.step(action)
        
        print(f"Step {i+1} - Reward: {reward}, Done: {done}")
        env.render()
        
        if done:
            break
            
    env.close()
