#!/bin/bash
#SBATCH --job-name=nla-critic-smoke
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=00:40:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/smoke_v2_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/smoke_v2_%j.err

set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
export HF_HOME=$PROJ/cache/huggingface
export TORCHINDUCTOR_CACHE_DIR=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/torchinductor   # else torch defaults to /tmp/torchinductor_$USER
export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJ"

nvidia-smi --query-gpu=name,memory.total --format=csv

torchrun --nproc_per_node=4 --master_port=29601 \
    -m nla.training.train_critic_sft \
    --data       "$PROJ/data/ar_sft_train_8b_ufw.parquet" \
    --eval-data  "$PROJ/data/ar_eval_8b_ufw.parquet" \
    --model-name "$PROJ/checkpoints/critic_init_8b_ufw" \
    --layer-index 24 \
    --output-dir "$PROJ/checkpoints/critic_smoke_v2" \
    --micro-batch-size 16 --global-batch-size 256 \
    --lr 2e-5 --min-lr 2e-6 --lr-warmup-iters 2 --lr-decay-style cosine \
    --num-steps 6 --max-length 512 --value-head-depth 1 \
    --save-every 100000 --eval-every 3 --eval-samples 256 --eval-batch-size 32 \
    --param-dtype float32 --fsdp

echo "=== SMOKE DONE ==="
