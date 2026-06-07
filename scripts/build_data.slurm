#!/bin/bash
#SBATCH --job-name=nla-build
#SBATCH --partition=berzelius-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/build_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/build_%j.err

set -euo pipefail

CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export HF_HOME=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/huggingface
mkdir -p "$HF_HOME"

PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
DATA=$PROJ/data
mkdir -p "$DATA" "$PROJ/logs"

echo "=== Build: $SPLIT ==="

# Build training parquet
python -m nla.training.build_training_data \
    --explained "$EXPLAINED" \
    --vectors "$DATA/${SPLIT}_vectors.parquet" \
    --tokenizer Qwen/Qwen3-4B \
    --output "$DATA/${SPLIT}_train.parquet" \
    --split-type "$SPLIT_TYPE"

echo "=== Build done ==="
ls -lh "$DATA/${SPLIT}_train.parquet"
