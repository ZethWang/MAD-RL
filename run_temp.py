import torch
import sys
import os
import time
from debate_gym_env import DebateEnv
from debate_logger import DebateLogger, run_debate_with_logging

if __name__ == "__main__":
    # 记录参数
    params = {
        "agents": 3,
        "debate_rounds": 5,
        "model": "/mnt/public/code/zzy/wzh/dynamic_train/dt-data/mars/ckpt/Qwen2.5-1.5B-Instruct",
        "device": "cuda",
        "episodes": 10,
        "base_threshold": 0.0,
        "final_threshold": 0.8,
        "outlier_threshold": 0.5,
        "min_weight": 0.1,
        "max_weight": 0.9,
        "use_outlier": True,
        "equality_weight": 0.5,
        "use_embeddings": True
    }
    
    # 初始化日志记录器
    logger = DebateLogger(log_file="logs/debate_run_20250420_085547.log", params=params)
    
    # 确保可用的CUDA内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # 创建环境
    env = DebateEnv(
        num_agents=3,
        debate_rounds=5,
        model_name="/mnt/public/code/zzy/wzh/dynamic_train/dt-data/mars/ckpt/Qwen2.5-1.5B-Instruct",
        device="cuda",
        base_threshold=0.0,
        final_threshold=0.8,
        outlier_threshold=0.5,
        min_weight=0.1,
        max_weight=0.9,
        use_outlier=True,
        equality_weight=0.5,
        use_embeddings=True
    )
    
    print("运行辩论环境...")
    try:
        run_debate_with_logging(env, episodes=10, logger=logger)
        # 保存结果
        result_file = logger.save_results()
    finally:
        # 关闭资源
        env.close()
        logger.close()
