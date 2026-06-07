#!/bin/bash
#SBATCH --job-name=nla-eval-base
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/eval_base_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/eval_base_%j.err
# Evaluate base model loss on 1024-row subsets (no training).
# Compares with SFT checkpoints to confirm SFT actually reduced loss.
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export HF_HOME=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/huggingface
mkdir -p "$HF_HOME"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla

echo "=== SFT results (from full training) ==="
echo "Critic SFT (250 steps, 124k rows): loss = 1.0604"
echo "Actor  SFT (250 steps, 124k rows): loss = 2.99"
echo ""

echo "=== Evaluating base critic loss (Qwen3-4B, no training) ==="
python -m nla.training.eval_base_loss \
    --data "$PROJ/data/eval_ar_sft_train.parquet" \
    --model-name Qwen/Qwen3-4B --task critic  --max-length 512   --batch-size 16

echo ""
echo "=== Evaluating base actor loss (Qwen3-4B, no training) ==="
python -m nla.training.eval_base_loss \
    --data "$PROJ/data/eval_av_sft_train.parquet" \
    --model-name Qwen/Qwen3-4B --task actor   --max-length 2048  --batch-size 2

echo ""
echo "=== Summary ==="
echo "If base loss >> SFT loss → SFT training works."
echo "If base loss ≈ SFT loss → SFT didn't learn anything useful."
echo "=== eval done ==="
