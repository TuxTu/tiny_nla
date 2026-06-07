#!/bin/bash
# Smoke test: verify training code works with Qwen3-0.6B on tiny synthetic data.
# Run on login node (1 GPU) or submit as SLURM job.
set -euo pipefail

module load Miniforge3/25.3.1-0 2>/dev/null || true
source /software/sse/generic/manual/ssetools/conda/hpc_conda_wrap2.sh 2>/dev/null || true
conda activate tiny_nla

PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
TEST=$PROJ/data/test_smoke
mkdir -p "$TEST" "$PROJ/checkpoints/test"

echo "=== Smoke Test: Critic SFT ==="
python -m nla.training.train_critic_sft \
    --data "$PROJ/data/ar_sft_train.parquet" \
    --model-name Qwen/Qwen3-0.6B \
    --output-dir "$PROJ/checkpoints/test/critic" \
    --micro-batch-size 1 --num-steps 3 --max-length 256 \
    --lr 1e-5 --save-every 10 2>&1 | tail -5
echo "Critic SFT: OK"

echo "=== Smoke Test: Actor SFT ==="
python -m nla.training.train_actor_sft \
    --data "$PROJ/data/av_sft_train.parquet" \
    --model-name Qwen/Qwen3-0.6B \
    --output-dir "$PROJ/checkpoints/test/actor" \
    --micro-batch-size 1 --num-steps 3 --max-length 256 \
    --lr 1e-5 --save-every 10 2>&1 | tail -5
echo "Actor SFT: OK"

echo "=== Smoke Test: RL (no SGLang) ==="
python -m nla.training.train_rl \
    --data "$PROJ/data/rl_train.parquet" \
    --model-name Qwen/Qwen3-0.6B \
    --actor-ckpt "$PROJ/checkpoints/test/actor" \
    --critic-ckpt "$PROJ/checkpoints/test/critic" \
    --output-dir "$PROJ/checkpoints/test/rl" \
    --n-samples 2 --rollout-batch 1 --micro-batch-size 1 \
    --num-steps 2 --kl-coef 0 --max-response-len 50 \
    --lr-actor 1e-5 --lr-critic 1e-5 --save-every 10 2>&1 | tail -5
echo "RL: OK"

echo ""
echo "=== All smoke tests passed ==="
