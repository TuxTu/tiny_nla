#!/bin/bash
#SBATCH --job-name=nla-lossmask
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/lossmask_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/lossmask_%j.err
#
# CONTROLLED TEST: loss bug, or just too few steps?
#
# Two runs identical in EVERY respect -- same data, scale 300, 300 steps,
# same warmup -- except the loss objective:
#   masked   = current code (response-only CE)          -> ladder got +0.1451
#   unmasked = pre-2026-08-23 code (pad scored, /allpos)
#
# The confound this removes: the only null-gap run with >=300 steps
# (17358927, +0.0013) had the unmasked loss AND scale 160 AND 487 steps, while
# every positive-gap run had the masked loss AND scale 300 AND >=300 steps.
# Three variables moved at once. This moves exactly one.
#
# unmasked ~ +0.00  -> the loss mask is causal.
# unmasked ~ +0.14  -> the loss mask was only a REPORTING bug and step count
#                      is what actually gates the channel.
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

TRAIN="$PROJ/data/split50/av_sft_train.parquet"
EVAL="$PROJ/data/av_eval_8b_ufw_joined.parquet"

for MODE in unmasked masked; do
  OUT="$PROJ/checkpoints/actor_lossmask_$MODE"
  EXTRA=""
  [ "$MODE" = "unmasked" ] && EXTRA="--legacy-unmasked-loss"
  echo "################ MODE=$MODE ################"
  torchrun --nproc_per_node=4 --master_port=29733 \
      -m nla.training.train_actor_sft \
      --data "$TRAIN" --model-name Qwen/Qwen3-8B --output-dir "$OUT" \
      --injection-scale 300 $EXTRA \
      --micro-batch-size 4 --global-batch-size 256 \
      --lr 2e-5 --min-lr 2e-6 --lr-warmup-iters 30 --lr-decay-style cosine \
      --num-steps 300 --max-length 512 --save-every 100000 \
      --eval-data "$EVAL" --eval-every 100 --eval-samples 256 --eval-batch-size 8 \
      --param-dtype float32 --fsdp

  echo "---- MODE=$MODE : ablation on HELD-OUT rows ----"
  python scripts/actor_vector_ablation.py \
      --actor-ckpt "$OUT" --data "$EVAL" --sidecar "$EVAL.nla_meta.yaml" \
      --injection-scale 300 --n-samples 256 --batch-size 8
  rm -rf "$OUT"
done
echo "=== LOSSMASK CONTROL DONE ==="
