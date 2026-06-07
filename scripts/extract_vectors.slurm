#!/bin/bash
#SBATCH --job-name=nla-ext
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-40GB:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/ext_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/ext_%j.err

set -euo pipefail

CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export HF_HOME=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/huggingface

PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
DATA=$PROJ/data
mkdir -p "$DATA" "$PROJ/logs" "$HF_HOME"

echo "=== Extract: $INPUT (shard $SHARD_ID/$NUM_SHARDS) → $OUTPUT ==="
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python -m nla.datagen.extract_vectors \
    --input "$INPUT" \
    --model-name Qwen/Qwen3-4B \
    --output "$DATA/$OUTPUT" \
    --batch-size 16 --max-length 2048 \
    --shard-id "$SHARD_ID" --num-shards "$NUM_SHARDS"

echo "=== Done ==="
ls -lh "$DATA/${OUTPUT}.shard_$(printf '%03d' $SHARD_ID)"
