#!/bin/bash
#SBATCH --job-name=nla-actor
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/actor_sft_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/actor_sft_%j.err
# Actor SFT: lr=2e-5 cosine→2e-6, warmup=50, global_batch=256, ~250 steps
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export HF_HOME=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/huggingface
mkdir -p "$HF_HOME"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
echo "=== Actor SFT Training (4 GPU DDP) ==="
torchrun --nproc_per_node=4 --master_port=29531 \
    -m nla.training.train_actor_sft \
    --data "$PROJ/data/av_sft_train.parquet" \
    --model-name Qwen/Qwen3-4B \
    --output-dir "$PROJ/checkpoints/actor_sft" \
    --micro-batch-size 4 --global-batch-size 256 \
    --lr 2e-5 --min-lr 2e-6 --lr-warmup-iters 50 --lr-decay-style cosine \
    --num-steps 250 --max-length 2048 --save-every 100 --ddp
echo "=== Actor SFT done ==="
