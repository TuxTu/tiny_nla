#!/bin/bash
#SBATCH --job-name=nla-rl-sg
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=16:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/rl_sglang_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/rl_sglang_%j.err
# Full RL GRPO: GPU0=SGLang, GPU1-7=DDP training + ZeroRedundancyOptimizer
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export HF_HOME=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/huggingface
mkdir -p "$HF_HOME"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla

# ── Launch SGLang on GPU 0 ───────────────────────────────────────────────
ACTOR_CKPT="$PROJ/checkpoints/actor_sft"
SGLANG_PORT=30000
SGLANG_LOG="$PROJ/logs/sglang_rl_${SLURM_JOB_ID}.log"
echo "Starting SGLang (GPU 0) ..."
module load buildenv-gcccuda/12.4.1-gcc13.3.0
export LD_LIBRARY_PATH=/software/sse/manual/GCC/13.3.0/lib64:$LD_LIBRARY_PATH
export TRITON_CACHE_DIR=/tmp/triton_cache_$SLURM_JOB_ID
mkdir -p "$TRITON_CACHE_DIR"
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
    --model-path "$ACTOR_CKPT" --host 0.0.0.0 --port "$SGLANG_PORT" \
    --mem-fraction-static 0.70 --disable-radix-cache --trust-remote-code \
    --disable-cuda-graph \
    > "$SGLANG_LOG" 2>&1 &
SGLANG_PID=$!
for i in $(seq 1 120); do
    if curl -s "http://localhost:$SGLANG_PORT/health" > /dev/null 2>&1; then
        echo "SGLang healthy after ${i}s."; break
    fi
    if ! kill -0 "$SGLANG_PID" 2>/dev/null; then
        echo "SGLang died!"; tail -30 "$SGLANG_LOG"; exit 1
    fi
    sleep 1
done
sleep 2

# ── Training (GPUs 1-7) ───────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc_per_node=7 --master_port=29550 \
    -m nla.training.train_rl \
    --data "$PROJ/data/rl_train.parquet" \
    --model-name Qwen/Qwen3-4B \
    --actor-ckpt "$ACTOR_CKPT" \
    --critic-ckpt "$PROJ/checkpoints/critic_sft" \
    --output-dir "$PROJ/checkpoints/rl" \
    --n-samples 8 --rollout-batch 2 --global-batch-size 112 \
    --micro-batch-size 1 \
    --lr-actor 1.41e-5 --lr-critic 1.41e-5 \
    --kl-coef 0.01 --max-response-len 256 --max-length 2048 \
    --num-steps 250 --save-every 100 \
    --ddp --use-sglang --sglang-external --sglang-gpu-id 0 --sglang-mem-fraction 0.70

kill "$SGLANG_PID" 2>/dev/null || true
wait "$SGLANG_PID" 2>/dev/null || true
echo "=== Full RL SGLang done ==="
