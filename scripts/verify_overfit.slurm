#!/bin/bash
#SBATCH --job-name=nla-overfit-test
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/overfit_test_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/overfit_test_%j.err

# Diagnostic: can the 64k-trained critic overfit to 4 samples?
set -euo pipefail

CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export HF_HOME=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/huggingface
mkdir -p "$HF_HOME"

PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
# Ensure our nla takes precedence over the original repo's nla
export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"

echo "=== Overfit Test 1: value_head only (fast) ==="
python "$PROJ/scripts/verify_overfit.py" \
    --ckpt "$PROJ/checkpoints/critic_sft_8b_64k" \
    --data "$PROJ/data/ar_sft_train_8b.parquet" \
    --n-samples 4 --n-steps 50 --lr 2e-5

echo ""
echo "=== Overfit Test 2: full model ==="
python "$PROJ/scripts/verify_overfit.py" \
    --ckpt "$PROJ/checkpoints/critic_sft_8b_64k" \
    --data "$PROJ/data/ar_sft_train_8b.parquet" \
    --n-samples 4 --n-steps 50 --lr 2e-5 \
    --train-full

echo "=== Done ==="
