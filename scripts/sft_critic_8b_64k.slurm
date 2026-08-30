#!/bin/bash
#SBATCH --job-name=nla-critic-8b-64k
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=10:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/critic_sft_8b_64k_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/critic_sft_8b_64k_%j.err
# Critic SFT 8B on 64k rows, 250 steps — per-step loss logging via tqdm stderr
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export HF_HOME=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/huggingface
export TORCHINDUCTOR_CACHE_DIR=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/torchinductor   # else torch defaults to /tmp/torchinductor_$USER
mkdir -p "$HF_HOME"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla

LOCAL_DIR="$PROJ/tmp/nla_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_DIR"
trap 'rm -rf "$LOCAL_DIR"' EXIT   # $PROJ/tmp is NOT node-wiped
echo "Copying data to $LOCAL_DIR ..."
cp "$PROJ/data/ar_sft_64k_8b.parquet" "$LOCAL_DIR/data.parquet"
cp "$PROJ/data/ar_sft_64k_8b.parquet.nla_meta.yaml" "$LOCAL_DIR/data.parquet.nla_meta.yaml"
echo "Copy done. Size: $(du -h "$LOCAL_DIR/data.parquet" | cut -f1)"

echo "=== Critic SFT 8B 64k (4 GPU FSDP, 250 steps, micro_batch=2) ==="
torchrun --nproc_per_node=4 --master_port=29530 \
    -m nla.training.train_critic_sft \
    --data "$LOCAL_DIR/data.parquet" \
    --model-name Qwen/Qwen3-8B \
    --output-dir "$PROJ/checkpoints/critic_sft_8b_64k" \
    --micro-batch-size 2 --global-batch-size 256 \
    --lr 2e-5 --min-lr 2e-6 --lr-warmup-iters 50 --lr-decay-style cosine \
    --num-steps 250 --max-length 512 --value-head-depth 1 --save-every 50 --fsdp

echo "=== Critic SFT 8B 64k done ==="
rm -rf "$LOCAL_DIR"
