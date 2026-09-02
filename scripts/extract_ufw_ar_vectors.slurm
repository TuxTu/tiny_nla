#!/bin/bash
#SBATCH --job-name=nla-ar-vectors
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=04:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/ar_vectors_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/ar_vectors_%j.err

# Extract Qwen3-8B vectors for AR split only, then build final training data.

set -euo pipefail

CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export HF_HOME=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/huggingface
export TORCHINDUCTOR_CACHE_DIR=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/torchinductor   # else torch defaults to /tmp/torchinductor_$USER
mkdir -p "$HF_HOME"

PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"

echo "=== AR vector extraction (Qwen3-8B) ==="

python -m nla.datagen.extract_vectors \
    --input "$PROJ/data/ufw_8b/ar_sft_explained.parquet" \
    --model-name Qwen/Qwen3-8B \
    --output "$PROJ/data/ufw_8b/ar_sft_explained_vectors.parquet" \
    --batch-size 16 \
    --max-length 2048 \
    --gpus 4

echo ""
echo "=== Building final AR training parquet ==="

python -m nla.training.build_training_data \
    --explained "$PROJ/data/ufw_8b/ar_sft_explained.parquet" \
    --vectors "$PROJ/data/ufw_8b/ar_sft_explained_vectors.parquet" \
    --tokenizer Qwen/Qwen3-8B \
    --output "$PROJ/data/ar_sft_train_8b_ufw.parquet" \
    --split-type ar_sft

echo "Done: $PROJ/data/ar_sft_train_8b_ufw.parquet"
