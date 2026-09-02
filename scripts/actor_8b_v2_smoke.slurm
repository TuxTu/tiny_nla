#!/bin/bash
#SBATCH --job-name=nla-actor-smoke
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=01:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/actor_smoke_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/actor_smoke_%j.err
# Smoke: 6 steps on the JOINED AV data. Catches OOM, verifies the held-out eval
# runs, and verifies the final save no longer deadlocks under FSDP.
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
export HF_HOME=$PROJ/cache/huggingface
export TORCHINDUCTOR_CACHE_DIR=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/torchinductor   # else torch defaults to /tmp/torchinductor_$USER
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJ"

torchrun --nproc_per_node=4 --master_port=29541 \
    -m nla.training.train_actor_sft \
    --data "$PROJ/data/av_sft_train_8b_ufw_joined.parquet" \
    --model-name Qwen/Qwen3-8B \
    --output-dir "$PROJ/checkpoints/actor_smoke_v2" \
    --micro-batch-size 4 --global-batch-size 256 \
    --lr 2e-5 --min-lr 2e-6 --lr-warmup-iters 2 --lr-decay-style cosine \
    --num-steps 6 --max-length 512 --save-every 100000 \
    --eval-data "$PROJ/data/av_eval_8b_ufw_joined.parquet" \
    --eval-every 3 --eval-samples 128 --eval-batch-size 8 \
    --param-dtype float32 --fsdp

echo "=== SMOKE DONE ==="
ls -la "$PROJ/checkpoints/actor_smoke_v2"
