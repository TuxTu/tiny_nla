#!/bin/bash
#SBATCH --job-name=nla-actor-smoke
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/actor_sft_8b_smoke_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/actor_sft_8b_smoke_%j.err
# Actor SFT 8B smoke test: FSDP ZeRO-3, 10 steps
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export HF_HOME=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/huggingface
mkdir -p "$HF_HOME"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
echo "=== Actor SFT 8B Smoke Test (4 GPU FSDP, 10 steps) ==="
torchrun --nproc_per_node=4 --master_port=29531 \
    -m nla.training.train_actor_sft \
    --data "$PROJ/data/av_sft_train_8b.parquet" \
    --model-name Qwen/Qwen3-8B \
    --output-dir "$PROJ/checkpoints/actor_sft_8b_smoke" \
    --micro-batch-size 1 --global-batch-size 256 \
    --lr 2e-5 --num-steps 10 --max-length 2048 --fsdp
echo "=== Smoke test done ==="
