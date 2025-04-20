import sys
import json
import os
import time
from datetime import datetime
import numpy as np
from typing import List, Dict, Any, Optional

class TeeLogger:
    """同时将输出写入终端和日志文件"""
    def __init__(self, filename, mode='a'):
        self.terminal = sys.stdout
        self.log_file = open(filename, mode)
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        
    def close(self):
        self.log_file.close()

class DebateLogger:
    """辩论系统的日志记录和结果管理"""
    
    def __init__(self, log_file: str, params: Dict[str, Any]):
        """
        初始化日志记录器
        
        Args:
            log_file: 日志文件路径
            params: 运行参数字典
        """
        self.log_file = log_file
        self.params = params
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.debate_history = []
        self.start_time = time.time()
        
        # 确保目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        os.makedirs("results", exist_ok=True)
        
        # 设置日志输出重定向
        sys.stdout = TeeLogger(log_file, 'a')
        sys.stderr = TeeLogger(log_file, 'a')
        
        # 记录系统信息
        self._log_system_info()
        self._log_parameters()
    
    def _log_system_info(self):
        """记录系统信息"""
        print("\n===== 系统信息 =====")
        import platform
        print(f"主机名: {platform.node()}")
        print(f"操作系统: {platform.platform()}")
        print(f"Python版本: {platform.python_version()}")
        
        # 尝试记录GPU信息
        try:
            import torch
            print(f"PyTorch版本: {torch.__version__}")
            if torch.cuda.is_available():
                print(f"CUDA可用: 是")
                print(f"CUDA版本: {torch.version.cuda}")
                print(f"GPU数量: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            else:
                print("CUDA可用: 否")
        except ImportError:
            print("PyTorch未安装")
        
        print("=" * 30)
    
    def _log_parameters(self):
        """记录运行参数"""
        print("\n===== 运行参数 =====")
        for key, value in self.params.items():
            print(f"{key}: {value}")
        print("=" * 30 + "\n")
    
    def record_episode(self, episode_data: Dict[str, Any]):
        """
        记录一个回合的数据
        
        Args:
            episode_data: 回合数据字典
        """
        self.debate_history.append(episode_data)
    
    def save_results(self) -> str:
        """
        保存结果到JSON文件
        
        Returns:
            str: 结果文件路径
        """
        result_file = f"results/debate_results_{self.timestamp}.json"
        
        # 计算正确回合数
        total_episodes = len(self.debate_history)
        correct_episodes = sum(1 for ep in self.debate_history if ep.get("is_correct", False))
        
        results_summary = {
            "parameters": self.params,
            "summary": {
                "total_episodes": total_episodes,
                "correct_episodes": correct_episodes,
                "accuracy": float(correct_episodes) / total_episodes if total_episodes > 0 else 0,
                "timestamp": datetime.now().isoformat(),
                "run_time_seconds": time.time() - self.start_time
            },
            "history": self.debate_history
        }
        
        with open(result_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\n详细结果已保存到: {result_file}")
        return result_file
    
    def close(self):
        """关闭日志文件"""
        if isinstance(sys.stdout, TeeLogger):
            sys.stdout.close()
        if isinstance(sys.stderr, TeeLogger):
            sys.stderr.close()
        
        # 恢复标准输出
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

def run_debate_with_logging(env, episodes=1, logger=None):
    """
    运行辩论环境并记录日志
    
    Args:
        env: 辩论环境实例
        episodes: 运行回合数
        logger: 日志记录器实例
    
    Returns:
        int: 正确回合数
    """
    total_correct = 0
    
    for episode in range(episodes):
        print(f"\n==== 回合 {episode+1}/{episodes} ====")
        episode_start_time = time.time()
        obs = env.reset()
        done = False
        total_reward = 0
        
        # 存储当前回合的历史
        episode_history = {
            "episode": episode + 1,
            "question": env.question,
            "options": env.options,
            "correct_answer": env.answer,
            "rounds": []
        }
        
        # 输出关于当前问题的信息
        print(f"问题: {env.question}")
        print(f"选项: A) {env.options[0]}, B) {env.options[1]}, C) {env.options[2]}, D) {env.options[3]}")
        print(f"正确答案: {env.answer}")
        print("初始回合观察到的矩阵形状:", obs['similarity_matrix'].shape)
        
        round_num = 0
        while not done:
            round_start_time = time.time()
            
            # 使用简单的启发式策略生成权重矩阵
            similarity_matrix = obs['similarity_matrix']
            action = np.zeros((env.num_agents, env.num_agents))
            
            # 对角线上设置为1（自己权重最高）
            for i in range(env.num_agents):
                action[i, i] = 1.0
                
            # 其他智能体的权重与相似度成比例
            for i in range(env.num_agents):
                for j in range(env.num_agents):
                    if i != j:
                        action[i, j] = 0.3 + 0.7 * similarity_matrix[i, j]
            
            # 确保行和为1
            row_sums = action.sum(axis=1, keepdims=True)
            action = action / row_sums
            
            print(f"\n--- 辩论轮次 {round_num+1} ---")
            print("应用的权重矩阵:")
            print(np.round(action, 2))
            
            # 执行步骤
            obs, reward, done, info = env.step(action)
            total_reward += reward
            round_num += 1
            
            # 收集当前轮次的答案和分析
            agent_answers = []
            for i, ans in enumerate(env.text_answer_this_round):
                if ans >= 0:
                    letter = chr(int(ans) + ord('A'))
                    agent_answers.append({"agent_id": i, "answer": letter})
                else:
                    agent_answers.append({"agent_id": i, "answer": "无效"})
            
            # 记录本轮详细信息
            round_info = {
                "round": round_num,
                "time_taken": time.time() - round_start_time,
                "reward": float(reward),
                "majority_answer": info['majority_answer'],
                "is_correct": info['is_correct'],
                "agent_answers": agent_answers,
                "weight_matrix": action.tolist(),
                "similarity_matrix": obs['similarity_matrix'].tolist()
            }
            
            episode_history["rounds"].append(round_info)
            
            print(f"奖励: {reward:.2f}")
            print(f"主流答案: {info['majority_answer']}")
            print(f"是否正确: {'✓' if info['is_correct'] else '✗'}")
            
            # 输出当前每个智能体的答案
            answers = []
            for i, ans in enumerate(env.text_answer_this_round):
                if ans >= 0:
                    letter = chr(int(ans) + ord('A'))
                    answers.append(f"智能体 {i}: {letter}")
                else:
                    answers.append(f"智能体 {i}: 无效")
            print("当前答案:", ", ".join(answers))
            
            time.sleep(0.5)  # 短暂暂停，使输出更易读
        
        is_correct = info['is_correct']
        if is_correct:
            total_correct += 1
        
        episode_history["total_reward"] = float(total_reward)
        episode_history["is_correct"] = is_correct
        episode_history["time_taken"] = time.time() - episode_start_time
        
        # 将当前回合添加到历史记录
        if logger:
            logger.record_episode(episode_history)
        
        print(f"\n回合 {episode+1} 完成，总奖励: {total_reward:.2f}")
        print(f"最终主流答案: {info['majority_answer']}, 正确答案: {info['correct_answer']}")
        print(f"结果: {'✓ 正确' if is_correct else '✗ 错误'}")
    
    # 打印总体准确率
    if episodes > 1:
        accuracy = total_correct / episodes * 100
        print(f"\n总体准确率: {accuracy:.1f}% ({total_correct}/{episodes})")
    
    return total_correct
