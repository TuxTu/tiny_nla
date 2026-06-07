#!/bin/bash
#SBATCH --job-name=nla-base-loss
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/base_loss_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/base_loss_%j.err
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export HF_HOME=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/huggingface
mkdir -p "$HF_HOME"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla

# Run SFT with 1 step on base model → gives us the step-0 loss
# Critic
echo "=== Critic ==="
python -m nla.training.train_critic_sft \
    --data "$PROJ/data/ar_sft_train.parquet" \
    --model-name Qwen/Qwen3-4B --output-dir /tmp/base_critic \
    --micro-batch-size 8 --global-batch-size 64 \
    --lr 2e-5 --num-steps 1 --max-length 512 --save-every 100 2>&1 | grep "loss"

# Actor
echo "=== Actor ==="
python -m nla.training.train_actor_sft \
    --data "$PROJ/data/av_sft_train.parquet" \
    --model-name Qwen/Qwen3-4B --output-dir /tmp/base_actor \
    --micro-batch-size 4 --global-batch-size 32 \
    --lr 2e-5 --num-steps 1 --max-length 2048 --save-every 100 2>&1 | grep "loss"

echo "=== Done ==="
