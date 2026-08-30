#!/bin/bash
#SBATCH --job-name=nla-rl-smoke
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=02:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/rl_smoke_fix_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/rl_smoke_fix_%j.err
#
# GRPO smoke with the FIXED actor (actor_sft_8b_s50fix) + split50 critic.
#
# Gated on the e2e coupling number, now measured: FVE_nrm_meannorm +0.5207
# against a +0.6165 gold ceiling (84.5% retained). The reward signal GRPO
# optimises is real, so a smoke test is now worth running -- before the fix it
# was -0.2793 and would have been optimising noise.
#
# --injection-scale 300 is MANDATORY and now enforced by train_rl.py. RL
# sidecars carry injection_scale: null, and the old 2.5*sqrt(d_model) fallback
# silently gave 160 while this checkpoint was SFT'd at 300 -- measured to cost
# ~12% of the conditioning gap (+0.4612 -> +0.4048).
#
# Checks: rollouts parse <explanation> tags, reward is not pinned at
# FAILED_EXTRACTION_REWARD=-2.0, advantages finite, FSDP save is collective.
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
export HF_HOME=$PROJ/cache/huggingface
export TORCHINDUCTOR_CACHE_DIR=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/torchinductor   # else torch defaults to /tmp/torchinductor_$USER
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$PROJ"

rm -rf "$PROJ/checkpoints/rl_smoke_s50fix"

torchrun --nproc_per_node=4 --master_port=29601 \
    -m nla.training.train_rl \
    --data "$PROJ/data/smoke_rl_8b_ufw.parquet" \
    --model-name Qwen/Qwen3-8B \
    --actor-ckpt  "$PROJ/checkpoints/actor_sft_8b_s50fix" \
    --critic-ckpt "$PROJ/checkpoints/critic_sft_8b_s50" \
    --output-dir  "$PROJ/checkpoints/rl_smoke_s50fix" \
    --injection-scale 300 \
    --n-samples 4 --rollout-batch 1 --micro-batch-size 1 \
    --lr-actor 1.41e-6 --lr-critic 1.41e-6 \
    --kl-coef 0.01 --max-response-len 150 --max-length 512 \
    --num-steps 10 --save-every 100 \
    --param-dtype float32 --fsdp

echo "=== RL SMOKE (s50fix) DONE ==="
