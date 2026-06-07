#!/bin/bash
#SBATCH --job-name=nla-rl
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=16:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/rl_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/rl_%j.err
# RL GRPO: 8-GPU DDP with HF generate. Simpler and verified working.
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export HF_HOME=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/huggingface
mkdir -p "$HF_HOME"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
echo "=== RL GRPO: 8-GPU DDP, HF generate ==="
torchrun --nproc_per_node=8 --master_port=29500 \
    -m nla.training.train_rl \
    --data "$PROJ/data/rl_train.parquet" \
    --model-name Qwen/Qwen3-4B \
    --actor-ckpt "$PROJ/checkpoints/actor_sft" \
    --critic-ckpt "$PROJ/checkpoints/critic_sft" \
    --output-dir "$PROJ/checkpoints/rl" \
    --n-samples 8 --rollout-batch 16 --global-batch-size 1024 \
    --micro-batch-size 1 \
    --lr-actor 1.41e-5 --lr-critic 1.41e-5 \
    --kl-coef 0.01 --max-response-len 150 --max-length 2048 \
    --num-steps 1000 --save-every 100 --ddp
echo "=== RL done ==="
