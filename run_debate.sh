#!/bin/bash

# 设置颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 获取当前时间戳，用于命名日志文件
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs"
mkdir -p $LOG_DIR
LOG_FILE="${LOG_DIR}/debate_run_${TIMESTAMP}.log"

# 打印带颜色的信息函数，同时保存到日志文件
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a $LOG_FILE
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a $LOG_FILE
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a $LOG_FILE
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a $LOG_FILE
}

# 记录开始时间
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
print_info "辩论环境运行开始时间: $START_TIME"
print_info "欢迎使用辩论环境一键运行脚本 (Qwen2.5-1.5B版本)"

# 检查数据目录
if [ ! -d "cais/data/test" ]; then
    print_warning "未找到数据目录 cais/data/test，创建示例目录..."
    mkdir -p cais/data/test
    
    # 创建示例数据文件
    cat > cais/data/test/example.csv << EOF
question,A,B,C,D,answer
"What is the capital of France?",Paris,London,Berlin,Madrid,A
"Which planet is closest to the Sun?",Earth,Venus,Mercury,Mars,C
"What is 2+2?",3,4,5,6,B
"Who wrote Romeo and Juliet?",Charles Dickens,William Shakespeare,Jane Austen,Mark Twain,B
"What is the chemical symbol for gold?",Au,Ag,Fe,Cu,A
EOF
    print_success "创建了示例数据文件"
fi

# 默认配置
NUM_AGENTS=3
DEBATE_ROUNDS=5
MODEL_NAME="/mnt/public/code/zzy/wzh/dynamic_train/dt-data/mars/ckpt/Qwen2.5-1.5B-Instruct"
DEVICE="cuda"
EPISODE_COUNT=1
BASE_THRESHOLD=0.0
FINAL_THRESHOLD=0.8
OUTLIER_THRESHOLD=0.5
MIN_WEIGHT=0.1
MAX_WEIGHT=0.9
USE_OUTLIER="True"
EQUALITY_WEIGHT=0.5
USE_EMBEDDINGS="True"

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --agents)
            NUM_AGENTS="$2"
            shift 2
            ;;
        --rounds)
            DEBATE_ROUNDS="$2"
            shift 2
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --episodes)
            EPISODE_COUNT="$2"
            shift 2
            ;;
        --base-threshold)
            BASE_THRESHOLD="$2"
            shift 2
            ;;
        --final-threshold)
            FINAL_THRESHOLD="$2"
            shift 2
            ;;
        --outlier-threshold)
            OUTLIER_THRESHOLD="$2"
            shift 2
            ;;
        --min-weight)
            MIN_WEIGHT="$2"
            shift 2
            ;;
        --max-weight)
            MAX_WEIGHT="$2"
            shift 2
            ;;
        --use-outlier)
            USE_OUTLIER="$2"
            shift 2
            ;;
        --equality-weight)
            EQUALITY_WEIGHT="$2"
            shift 2
            ;;
        --use-embeddings)
            USE_EMBEDDINGS="$2"
            shift 2
            ;;
        --help)
            echo "使用方法: ./run_debate.sh [选项]" | tee -a $LOG_FILE
            echo "选项:" | tee -a $LOG_FILE
            echo "  --agents N            设置代理数量 (默认: 3)" | tee -a $LOG_FILE
            echo "  --rounds N            设置辩论轮数 (默认: 3)" | tee -a $LOG_FILE
            echo "  --model NAME          设置模型名称 (默认: /mnt/public/code/zzy/wzh/dynamic_train/dt-data/mars/ckpt/Qwen2.5-1.5B-Instruct)" | tee -a $LOG_FILE
            echo "  --device DEV          设置设备 (默认: cuda)" | tee -a $LOG_FILE
            echo "  --episodes N          设置运行回合数 (默认: 1)" | tee -a $LOG_FILE
            echo "  --base-threshold N    设置初始信任阈值 (默认: 0.0)" | tee -a $LOG_FILE
            echo "  --final-threshold N   设置最终信任阈值 (默认: 0.8)" | tee -a $LOG_FILE
            echo "  --outlier-threshold N 设置离群点阈值 (默认: 0.5)" | tee -a $LOG_FILE
            echo "  --min-weight N        设置最小权重 (默认: 0.1)" | tee -a $LOG_FILE
            echo "  --max-weight N        设置最大权重 (默认: 0.9)" | tee -a $LOG_FILE
            echo "  --use-outlier BOOL    是否使用离群点检测 (默认: true)" | tee -a $LOG_FILE
            echo "  --equality-weight N   设置平等权重 (默认: 0.5)" | tee -a $LOG_FILE
            echo "  --use-embeddings BOOL 是否使用嵌入 (默认: true)" | tee -a $LOG_FILE
            echo "  --help                显示此帮助信息" | tee -a $LOG_FILE
            exit 0
            ;;
        *)
            print_error "未知参数: $1"
            exit 1
            ;;
    esac
done

# 检查依赖库
print_info "检查Python依赖..."
python -c "
import importlib, sys
required_packages = ['gym', 'numpy', 'pandas', 'torch', 'transformers', 'sentence_transformers']
missing_packages = []
for package in required_packages:
    try:
        importlib.import_module(package)
    except ImportError:
        missing_packages.append(package)
if missing_packages:
    print(','.join(missing_packages))
    sys.exit(1)
sys.exit(0)
" || {
    missing=$(python -c "
import importlib, sys
required_packages = ['gym', 'numpy', 'pandas', 'torch', 'transformers', 'sentence_transformers']
missing_packages = []
for package in required_packages:
    try:
        importlib.import_module(package)
    except ImportError:
        missing_packages.append(package)
if missing_packages:
    print(','.join(missing_packages))
")
    print_error "缺少必要的Python库: $missing"
    print_warning "请安装缺少的库: pip install $missing"
    exit 1
}
print_success "所有依赖都已安装"

# 创建简化版的运行脚本
cat > run_temp.py << EOF
import torch
import sys
import os
import time
from debate_gym_env import DebateEnv
from debate_logger import DebateLogger, run_debate_with_logging

if __name__ == "__main__":
    # 记录参数
    params = {
        "agents": ${NUM_AGENTS},
        "debate_rounds": ${DEBATE_ROUNDS},
        "model": "${MODEL_NAME}",
        "device": "${DEVICE}",
        "episodes": ${EPISODE_COUNT},
        "base_threshold": ${BASE_THRESHOLD},
        "final_threshold": ${FINAL_THRESHOLD},
        "outlier_threshold": ${OUTLIER_THRESHOLD},
        "min_weight": ${MIN_WEIGHT},
        "max_weight": ${MAX_WEIGHT},
        "use_outlier": ${USE_OUTLIER},
        "equality_weight": ${EQUALITY_WEIGHT},
        "use_embeddings": ${USE_EMBEDDINGS}
    }
    
    # 初始化日志记录器
    logger = DebateLogger(log_file="${LOG_FILE}", params=params)
    
    # 确保可用的CUDA内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # 创建环境
    env = DebateEnv(
        num_agents=${NUM_AGENTS},
        debate_rounds=${DEBATE_ROUNDS},
        model_name="${MODEL_NAME}",
        device="${DEVICE}",
        base_threshold=${BASE_THRESHOLD},
        final_threshold=${FINAL_THRESHOLD},
        outlier_threshold=${OUTLIER_THRESHOLD},
        min_weight=${MIN_WEIGHT},
        max_weight=${MAX_WEIGHT},
        use_outlier=${USE_OUTLIER},
        equality_weight=${EQUALITY_WEIGHT},
        use_embeddings=${USE_EMBEDDINGS}
    )
    
    print("运行辩论环境...")
    try:
        run_debate_with_logging(env, episodes=${EPISODE_COUNT}, logger=logger)
        # 保存结果
        result_file = logger.save_results()
    finally:
        # 关闭资源
        env.close()
        logger.close()
EOF

# 运行Python脚本
print_info "启动辩论环境..."
print_info "使用配置:"
print_info "- 代理数量: ${NUM_AGENTS}"
print_info "- 辩论轮数: ${DEBATE_ROUNDS}"
print_info "- 模型: ${MODEL_NAME}"
print_info "- 设备: ${DEVICE}"
print_info "- 回合数: ${EPISODE_COUNT}"
print_info "- 初始信任阈值: ${BASE_THRESHOLD}"
print_info "- 最终信任阈值: ${FINAL_THRESHOLD}"
print_info "- 其他超参数: 离群点阈值=${OUTLIER_THRESHOLD}, 权重范围=[${MIN_WEIGHT},${MAX_WEIGHT}], 平等权重=${EQUALITY_WEIGHT}"

python run_temp.py

# 记录结束时间
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
ELAPSED=$(( $(date -d "$END_TIME" +%s) - $(date -d "$START_TIME" +%s) ))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$(( ELAPSED % 60 ))

print_info "辩论环境运行结束时间: $END_TIME"
print_info "总运行时间: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"

# 清理临时文件
print_info "清理临时文件..."
# rm run_temp.py

print_success "辩论环境运行完成！"
print_success "日志文件: ${LOG_FILE}"
print_success "结果文件: results/debate_results_${TIMESTAMP}.json"
