#!/bin/bash
#SBATCH --job-name=nla-actor-8b
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/actor_sft_8b_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/actor_sft_8b_%j.err
# Actor SFT 8B: Qwen3-8B, FSDP ZeRO-3, 1 epoch on 499k rows
# Copy data to local SSD to avoid NFS deadlock.
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export HF_HOME=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/huggingface
export TORCHINDUCTOR_CACHE_DIR=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/torchinductor   # else torch defaults to /tmp/torchinductor_$USER
mkdir -p "$HF_HOME"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla

# Copy data to local SSD
LOCAL_DIR="$PROJ/tmp/nla_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_DIR"
trap 'rm -rf "$LOCAL_DIR"' EXIT   # $PROJ/tmp is NOT node-wiped
echo "Copying data to $LOCAL_DIR ..."
cp "$PROJ/data/av_sft_train_8b.parquet" "$LOCAL_DIR/data.parquet"
echo "Copy done. Size: $(du -h "$LOCAL_DIR/data.parquet" | cut -f1)"

echo "=== Actor SFT 8B Training (8 GPU FSDP, 1949 steps ≈ 1 epoch) ==="
torchrun --nproc_per_node=8 --master_port=29531 \
    -m nla.training.train_actor_sft \
    --data "$LOCAL_DIR/data.parquet" \
    --model-name Qwen/Qwen3-8B \
    --output-dir "$PROJ/checkpoints/actor_sft_8b" \
    --micro-batch-size 2 --global-batch-size 256 \
    --lr 2e-5 --min-lr 2e-6 --lr-warmup-iters 50 --lr-decay-style cosine \
    --num-steps 1949 --max-length 2048 --save-every 500 --fsdp

echo "=== Actor SFT 8B done ==="
rm -rf "$LOCAL_DIR"
