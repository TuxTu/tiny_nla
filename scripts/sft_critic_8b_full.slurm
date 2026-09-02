#!/bin/bash
#SBATCH --job-name=nla-critic-8b-full
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/critic_sft_8b_full_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/critic_sft_8b_full_%j.err
# Critic SFT 8B full: resume from 64k checkpoint, 1 epoch on 449k rows (1754 steps)
# 8× A100-80GB FSDP, estimated ~4.5h
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export HF_HOME=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/huggingface
export TORCHINDUCTOR_CACHE_DIR=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/torchinductor   # else torch defaults to /tmp/torchinductor_$USER
mkdir -p "$HF_HOME"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla

echo "=== Critic SFT 8B Full (8 GPU FSDP, resume from 64k, 1754 steps ≈ 1 epoch) ==="
echo "Data: ar_sft_train_8b.parquet (449k rows)"
echo "Resuming from: $PROJ/checkpoints/critic_sft_8b_full"

torchrun --nproc_per_node=4 --master_port=29530 \
    -m nla.training.train_critic_sft \
    --data "$PROJ/data/ar_sft_train_8b.parquet" \
    --model-name Qwen/Qwen3-8B \
    --output-dir "$PROJ/checkpoints/critic_sft_8b_full" \
    --micro-batch-size 2 --global-batch-size 256 \
    --lr 2e-5 --min-lr 2e-6 --lr-warmup-iters 50 --lr-decay-style cosine \
    --num-steps 1754 --max-length 512 --value-head-depth 1 --save-every 500 --fsdp \
    --resume

echo "=== Critic SFT 8B Full done ==="
