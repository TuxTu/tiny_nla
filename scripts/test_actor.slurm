#!/bin/bash
#SBATCH --job-name=nla-test-actor
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/test_actor_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/test_actor_%j.err
# Small-scale real-data actor SFT test — 2 steps, verify pipeline before full run
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export HF_HOME=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/huggingface
mkdir -p "$HF_HOME"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
echo "=== Small-scale test: Actor SFT DDP 4 GPUs, 2 steps (real data) ==="
torchrun --nproc_per_node=4 --master_port=29521 \
    -m nla.training.train_actor_sft \
    --data "$PROJ/data/av_sft_train.parquet" \
    --model-name Qwen/Qwen3-4B \
    --output-dir "$PROJ/checkpoints/test_actor" \
    --micro-batch-size 4 --global-batch-size 64 \
    --lr 2e-5 --min-lr 2e-6 --lr-warmup-iters 10 --lr-decay-style cosine \
    --num-steps 2 --max-length 2048 --save-every 100 --ddp
echo "=== Test actor done ==="
