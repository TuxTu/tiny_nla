#!/bin/bash
#SBATCH --job-name=nla-rl-smoke
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=02:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/rl_smoke_v2_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/rl_smoke_v2_%j.err
# GRPO smoke on the JOINED RL data with the loss-mask-fixed actor + split50 critic.
# Checks: rollouts produce parseable <explanation> tags, reward is not pinned at
# FAILED_EXTRACTION_REWARD=-2.0, advantages are finite, and the save is collective.
#
# Gating evidence (job 17363959, 500 held-out rows) -- RL has a real reward:
#   e2e FVE_nrm_meannorm  greedy +0.5207 | sampled +0.4581 | gold ceiling +0.6165
# GRPO rollouts are SAMPLED, so +0.4581 (74% of ceiling) is the starting point.
#
# --injection-scale 300 is REQUIRED and must match the actor SFT scale. RL
# sidecars carry injection_scale: null; train_rl.py used to fall back to
# 2.5*sqrt(4096)=160 silently, which measures -12% conditioning (+0.4048 vs
# +0.4612) and would have taxed every rollout with nothing in the log to show it.
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
export HF_HOME=$PROJ/cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$PROJ"

rm -rf "$PROJ/checkpoints/rl_smoke_v2"

torchrun --nproc_per_node=4 --master_port=29551 \
    -m nla.training.train_rl \
    --data "$PROJ/data/smoke_rl_8b_ufw.parquet" \
    --model-name Qwen/Qwen3-8B \
    --actor-ckpt  "$PROJ/checkpoints/actor_sft_8b_s50fix" \
    --critic-ckpt "$PROJ/checkpoints/critic_sft_8b_s50" \
    --output-dir  "$PROJ/checkpoints/rl_smoke_v2" \
    --injection-scale 300 \
    --n-samples 4 --rollout-batch 1 --micro-batch-size 1 \
    --lr-actor 1.41e-6 --lr-critic 1.41e-6 \
    --kl-coef 0.01 --max-response-len 150 --max-length 512 \
    --num-steps 10 --save-every 100 \
    --param-dtype float32 --fsdp

echo "=== RL SMOKE DONE ==="
