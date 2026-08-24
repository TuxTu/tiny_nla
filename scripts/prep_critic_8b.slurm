#!/bin/bash
#SBATCH --job-name=nla-prep-critic
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-40GB:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/prep_critic_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/prep_critic_%j.err
# Run original prepare_critic_checkpoint.py to create truncated Qwen3-8B checkpoint
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export HF_HOME=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/huggingface
mkdir -p "$HF_HOME"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla

python -m nla.scripts.prepare_critic_checkpoint \
    --base-model Qwen/Qwen3-8B \
    --num-layers 24 \
    --dataset-sidecar "$PROJ/data/ar_sft_train_8b.parquet" \
    --output "$PROJ/checkpoints/critic_init_8b"

echo "=== Done: critic checkpoint at $PROJ/checkpoints/critic_init_8b ==="
