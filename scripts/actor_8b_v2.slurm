#!/bin/bash
#SBATCH --job-name=nla-actor-8b-v2
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/actor_8b_v2_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/actor_8b_v2_%j.err
# Actor (AV) SFT, Qwen3-8B, from scratch on the JOINED UFW data.
# Mirrors the critic run that reached +65.7% FVE: 1 epoch = 249,449/256 = 974 steps,
# fp32 master weights, cosine 2e-5 -> 2e-6, warmup 50.
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
export HF_HOME=$PROJ/cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJ"

LOCAL_DIR="/tmp/nla_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_DIR"
for f in av_sft_train_8b_ufw_joined av_eval_8b_ufw_joined; do
    cp "$PROJ/data/$f.parquet" "$LOCAL_DIR/"
    cp "$PROJ/data/$f.parquet.nla_meta.yaml" "$LOCAL_DIR/"
done
echo "data staged on local SSD: $(du -sh "$LOCAL_DIR" | cut -f1)"

torchrun --nproc_per_node=4 --master_port=29542 \
    -m nla.training.train_actor_sft \
    --data "$LOCAL_DIR/av_sft_train_8b_ufw_joined.parquet" \
    --model-name Qwen/Qwen3-8B \
    --output-dir "$PROJ/checkpoints/actor_sft_8b_v2" \
    --micro-batch-size 4 --global-batch-size 256 \
    --lr 2e-5 --min-lr 2e-6 --lr-warmup-iters 50 --lr-decay-style cosine \
    --num-steps 974 --max-length 512 --save-every 200 \
    --eval-data "$LOCAL_DIR/av_eval_8b_ufw_joined.parquet" \
    --eval-every 100 --eval-samples 512 --eval-batch-size 8 \
    --param-dtype float32 --fsdp

echo "=== ACTOR SFT DONE ==="
rm -rf "$LOCAL_DIR"
