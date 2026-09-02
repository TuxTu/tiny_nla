#!/bin/bash
#SBATCH --job-name=nla-critic-8b-4g
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/critic_sft_8b_4g_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/critic_sft_8b_4g_%j.err
# Critic SFT 8B on 4 GPU with FSDP CPU offload (fits 8B on 4 GPUs)
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export HF_HOME=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/huggingface
export TORCHINDUCTOR_CACHE_DIR=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/torchinductor   # else torch defaults to /tmp/torchinductor_$USER
mkdir -p "$HF_HOME"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
echo "=== Critic SFT 8B (4 GPU FSDP + CPU offload, 1 epoch) ==="
torchrun --nproc_per_node=4 --master_port=29530 \
    -m nla.training.train_critic_sft \
    --data "$PROJ/data/ar_sft_train_8b.parquet" \
    --model-name Qwen/Qwen3-8B \
    --output-dir "$PROJ/checkpoints/critic_sft_8b" \
    --micro-batch-size 1 --global-batch-size 256 \
    --lr 2e-5 --min-lr 2e-6 --lr-warmup-iters 50 --lr-decay-style cosine \
    --num-steps 1949 --max-length 512 --value-head-depth 1 --save-every 500 --fsdp
echo "=== Critic SFT 8B done ==="
