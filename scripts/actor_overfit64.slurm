#!/bin/bash
#SBATCH --job-name=nla-av-overfit
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=02:30:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/av_overfit_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/av_overfit_%j.err
# CAN the actor condition on the vector AT ALL?
#
# The injection-scale A/B came back null on both arms (gap +0.0027 / +0.0003),
# so the question is no longer "which scale" but "is the vector -> text path
# functional". This is the actor's version of the critic overfit test that
# settled the same question in June.
#
# 64 rows, 64 distinct documents, 300 epochs. Every one of those 64 prompts is
# BYTE-IDENTICAL -- the template is fixed and the only thing that varies is the
# injected vector. So the ONLY channel through which the model can tell the 64
# responses apart is the vector. If CE collapses toward 0, the path works and
# the null gap is an optimization/scale-of-data problem. If CE stalls at the
# marginal-entropy floor (~1.6), the path is mechanically broken and no amount
# of training data would have helped.
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
export HF_HOME=$PROJ/cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJ"

DATA="$PROJ/data/split50/av_overfit64.parquet"
OUT="$PROJ/checkpoints/actor_overfit64"
STEPS=${STEPS:-300}
SCALE=${SCALE:-300}

torchrun --nproc_per_node=4 --master_port=29811 \
    -m nla.training.train_actor_sft \
    --data "$DATA" \
    --model-name Qwen/Qwen3-8B \
    --output-dir "$OUT" \
    --injection-scale "$SCALE" \
    --micro-batch-size 4 --global-batch-size 64 \
    --lr 2e-5 --min-lr 2e-6 --lr-warmup-iters 10 --lr-decay-style cosine \
    --num-steps "$STEPS" --max-length 512 --save-every 100000 \
    --eval-data "$DATA" --eval-every 25 --eval-samples 64 --eval-batch-size 8 \
    --param-dtype float32 --fsdp

echo "---- vector-conditioning ablation on the memorised rows ----"
python scripts/actor_vector_ablation.py \
    --actor-ckpt "$OUT" --data "$DATA" --sidecar "$DATA.nla_meta.yaml" \
    --injection-scale "$SCALE" --n-samples 64 --batch-size 8

echo "=== AV OVERFIT DONE ==="
