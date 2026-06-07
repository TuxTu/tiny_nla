#!/bin/bash
#SBATCH --job-name=nla-smoke-rl
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=00:20:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/smoke_rl_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/smoke_rl_%j.err
# RL smoke test: 8-GPU DDP, HF generate, 5 steps on 64-row smoke data.
# Verifies end-to-end pipeline (injection → generation → reward → policy gradient).
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export HF_HOME=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/huggingface
mkdir -p "$HF_HOME"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla

echo "=== RL smoke test: 8-GPU DDP, HF generate, 5 steps ==="
# rollout_batch=1 per GPU = 8 prompts total × 8 samples = 64 global_batch
torchrun --nproc_per_node=8 --master_port=29550 \
    -m nla.training.train_rl \
    --data "$PROJ/data/smoke_rl_train.parquet" \
    --model-name Qwen/Qwen3-4B \
    --actor-ckpt "$PROJ/checkpoints/smoke_actor" \
    --critic-ckpt "$PROJ/checkpoints/smoke_critic" \
    --output-dir "$PROJ/checkpoints/smoke_rl" \
    --n-samples 8 --rollout-batch 1 --global-batch-size 64 \
    --micro-batch-size 1 \
    --lr-actor 1.41e-5 --lr-critic 1.41e-5 \
    --kl-coef 0.01 --max-response-len 150 --max-length 2048 \
    --num-steps 5 --save-every 100 --ddp

echo "=== RL smoke test done ==="
