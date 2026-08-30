#!/bin/bash
#SBATCH --job-name=nla-rl-run1
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=20:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/rl_run1_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/rl_run1_%j.err
#
# First real GRPO run on the fixed actor. STAGED: 300 steps, then stop and
# evaluate properly before committing more compute.
#
# Why staged: train_rl.py has no --eval-every, so the only in-run signal is
# training reward -- and README's "Bitter Lesson" records that signal lying on
# the 4B runs (reward improved -1.1 -> -0.8 while every checkpoint produced
# degenerate output at eval). Reward alone is not evidence of success here.
#
# Config scaled from the reference production run (16xH100, global batch 1024,
# parity LR 1.41e-5, KL 0.01, 4199 rollouts -> fve_nrm 0.752). We have 4xA100:
# --freeze-critic: the OOM is in loss.backward() (train_rl.py:889), not
#   generation -- step 0 passes, step 1 dies, because AdamW allocates exp_avg /
#   exp_avg_sq lazily on the first .step(). With fp32 masters that is ~16 GB/GPU
#   for the actor and ~11 GB for the critic on top of ~22 GB of params and grads
#   plus the 8 GB frozen reference model: ~62 of 79 GB before any activations.
#   Freezing the critic drops its grads and optimiser states (~17 GB/GPU).
#   NOTE this diverges from the reference, which trains both jointly. Arguably
#   cleaner here: with both trainable the pair can co-adapt, the critic learning
#   to decode the actor's idiosyncratic phrasing and inflating reward without
#   improving interpretability. Our critic is already at +62.93% FVE.
#
#   MEASURED, not estimated. Three OOMs got us here:
#     8 prompts x 8 samples (17364972) -> OOM before step 0
#     2 x 8            (17365395) -> OOM at step 1 (AdamW states materialise)
#     2 x 8 + freeze   (17365455) -> OOM at step 1, still 68.9 GiB allocated
#   The ONLY configuration proven on this hardware is the smoke test's
#   1 prompt x 4 samples, which ran 10 steps with BOTH models trainable
#   (job 17364576). So: take that, change one variable (N 4 -> 8), and keep the
#   ~17 GB the frozen critic frees as headroom. max-length 320 not 512 as well:
#   prompt ~105 + response 150 = 255, so 512 was paying for padding we never use.
#
#   global batch 8 = 1 prompt x 8 samples. Earlier attempts, for the record: all
#     4 ranks hit 79.2/79.25 GiB -- three 8B models resident (actor, frozen
#     reference for KL, critic) plus fp32 optimiser states leaves little room,
#     and the KV cache for 64 concurrent generations does not fit.
#   parity LR 2.5e-6. The sqrt rule gives 1.77e-6 at batch 16; nudged up because
#     at 300 steps x 16 = 4,800 samples we are at 0.11% of the reference run's
#     4.3M samples, so an over-conservative LR would show nothing at all.
#
# SCOPE: this run cannot reach the reference's fve_nrm 0.752 -- not enough
# compute by three orders of magnitude. It is a stability check: does GRPO run
# 300 steps without mode collapse or length drift, and what is the steps/hour.
#   critic LR at parity, NOT the scan's 5e-5: TRAINING_NOTES flags that as a
#     30-step artifact, and a hotter critic rewards verbosity -> length drift.
#
# --n-samples 8 (not the smoke's 4): with --max-response-len 150 truncating
# ~26.6% of this actor's generations, N=4 lets a whole GRPO group fail together
# (smoke step 8: 4/4 -> std_r ~ 0 -> zero advantages -> step silently dropped).
# N=8 makes total-group failure much rarer, so truncation costs individual
# rollouts rather than entire prompts.
#
# --injection-scale 300 is mandatory: must match the actor's SFT scale.
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
export HF_HOME=$PROJ/cache/huggingface
export TORCHINDUCTOR_CACHE_DIR=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/torchinductor   # else torch defaults to /tmp/torchinductor_$USER
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd "$PROJ"

torchrun --nproc_per_node=4 --master_port=29611 \
    -m nla.training.train_rl \
    --data "$PROJ/data/rl_train_8b_ufw_joined.parquet" \
    --model-name Qwen/Qwen3-8B \
    --actor-ckpt  "$PROJ/checkpoints/actor_sft_8b_s50fix" \
    --critic-ckpt "$PROJ/checkpoints/critic_sft_8b_s50" \
    --output-dir  "$PROJ/checkpoints/rl_8b_run1" \
    --injection-scale 300 \
    --n-samples 8 --rollout-batch 1 --micro-batch-size 1 --freeze-critic \
    --lr-actor 1.5e-6 --lr-critic 1.5e-6 \
    --kl-coef 0.01 --kl-type per_seq \
    --max-response-len 150 --max-length 320 \
    --num-steps 300 --save-every 50 \
    --param-dtype float32 --fsdp

echo "=== RL RUN1 DONE ==="
