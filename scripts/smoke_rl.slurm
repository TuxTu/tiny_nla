#!/bin/bash
#SBATCH --job-name=nla-smoke-rl
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=00:30:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/smoke_rl_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/smoke_rl_%j.err

# RL GRPO smoke test: GPU0=SGLang, GPU1-7=DDP training (7 GPUs), 2 steps

set -euo pipefail

CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export HF_HOME=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/huggingface
mkdir -p "$HF_HOME"

PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla

echo "=== Smoke test: RL GRPO 8 GPUs, 2 steps ==="
echo "Node: $(hostname)"
echo "GPU0=SGLang, GPU1-7=DDP training (7 GPUs)"

torchrun --nproc_per_node=8 --master_port=29503 \
    -m nla.training.train_rl \
    --data "$PROJ/data/smoke_rl_train.parquet" \
    --model-name Qwen/Qwen3-4B \
    --actor-ckpt "$PROJ/checkpoints/smoke_actor" \
    --critic-ckpt "$PROJ/checkpoints/smoke_critic" \
    --output-dir "$PROJ/checkpoints/smoke_rl" \
    --n-samples 4 \
    --rollout-batch 2 \
    --micro-batch-size 1 \
    --lr-actor 1.41e-5 \
    --lr-critic 1.41e-5 \
    --kl-coef 0.01 \
    --max-response-len 150 \
    --max-length 2048 \
    --num-steps 2 \
    --save-every 100 \
    --ddp

echo "=== Smoke RL done ==="
