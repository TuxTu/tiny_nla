#!/bin/bash
#SBATCH --job-name=nla-smoke-rl-8b
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=01:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/smoke_rl_8b_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/smoke_rl_8b_%j.err

# RL GRPO smoke test for 8B v16: 4 GPU FSDP, gather→standalone→generate→broadcast
# Instead of summon_full_params (which conflicts with FSDP/gradient-checkpointing),
# we gather the full state dict, load it into a standalone model on rank 0,
# generate with standard HF .generate(), then broadcast results to all ranks.

set -euo pipefail

CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export HF_HOME=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/huggingface
mkdir -p "$HF_HOME"

PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== RL Smoke Test v16: 8B GRPO, 4 GPU FSDP (gather→standalone→broadcast) ==="
echo "Node: $(hostname)"
echo "Actor: $PROJ/checkpoints/actor_sft_8b_64k"
echo "Critic: $PROJ/checkpoints/critic_sft_8b_64k"

rm -rf "$PROJ/checkpoints/smoke_rl_8b"

torchrun --nproc_per_node=4 --master_port=29503 \
    -m nla.training.train_rl \
    --data "$PROJ/data/smoke_rl_8b.parquet" \
    --model-name Qwen/Qwen3-8B \
    --actor-ckpt "$PROJ/checkpoints/actor_sft_8b_64k" \
    --critic-ckpt "$PROJ/checkpoints/critic_sft_8b_64k" \
    --output-dir "$PROJ/checkpoints/smoke_rl_8b" \
    --n-samples 2 \
    --rollout-batch 1 \
    --micro-batch-size 1 \
    --lr-actor 1.41e-5 \
    --lr-critic 1.41e-5 \
    --kl-coef 0 \
    --max-response-len 150 \
    --max-length 2048 \
    --num-steps 10 \
    --save-every 100 \
    --fsdp

echo "=== Smoke RL 8B v16 done ==="
