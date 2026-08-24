#!/bin/bash
#SBATCH --job-name=nla-smoke-rand
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/critic_sft_8b_randinit_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/critic_sft_8b_randinit_%j.err
# Smoke test: 50 steps WITH RANDOM INIT (matching original repo), 1000-row dataset
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export HF_HOME=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/huggingface
mkdir -p "$HF_HOME"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla

echo "=== Critic SFT 8B Smoke Test: RANDOM INIT (like original repo), 50 steps ==="
torchrun --nproc_per_node=4 --master_port=29530 \
    -m nla.training.train_critic_sft \
    --data "$PROJ/data/ar_sft_smoke_8b.parquet" \
    --model-name Qwen/Qwen3-8B \
    --output-dir "$PROJ/checkpoints/critic_sft_8b_randinit" \
    --micro-batch-size 1 --global-batch-size 16 \
    --lr 2e-5 --num-steps 50 --max-length 512 --value-head-depth 1 --save-every 100 --fsdp

echo "=== Done ==="
