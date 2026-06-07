#!/bin/bash
#SBATCH --job-name=nla-critic
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/critic_sft_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/critic_sft_%j.err
# Critic SFT: lr=2e-5 cosine→2e-6, warmup=50, global_batch=256, ~250 steps
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export HF_HOME=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/huggingface
mkdir -p "$HF_HOME"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
echo "=== Critic SFT Training (4 GPU DDP) ==="
torchrun --nproc_per_node=4 --master_port=29530 \
    -m nla.training.train_critic_sft \
    --data "$PROJ/data/ar_sft_train.parquet" \
    --model-name Qwen/Qwen3-4B \
    --output-dir "$PROJ/checkpoints/critic_sft" \
    --micro-batch-size 4 --global-batch-size 256 \
    --lr 2e-5 --min-lr 2e-6 --lr-warmup-iters 50 --lr-decay-style cosine \
    --num-steps 250 --max-length 512 --save-every 100 --ddp
echo "=== Critic SFT done ==="
