#!/bin/bash
#SBATCH --job-name=nla-rl-run3
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/rl_run3_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/rl_run3_%j.err
#
# GRPO run 3. IDENTICAL to run2 except --lr-actor/--lr-critic 3.5e-6 -> 1.5e-6.
#
# run2 at step 50 came out WORSE than run1 at step 50 (retrieval 59.7% vs 65.3%,
# F1 0.3400 vs 0.3797) despite the fixed reward and 8x the data, and its
# |actor_loss| grew 5.8 -> 200 over 175 steps. That growth is policy drift from
# the SFT reference, not a scale artefact: the 150->224 cap accounts for at most
# ~2.2x of it (per_seq KL is a squared difference of SUMS, so it scales ~length^2).
#
# The sqrt-rule LR (3.5e-6 at global batch 64) assumes a bigger batch is a better
# estimate of the SAME gradient. Here the reward itself is noisy, so a bigger
# batch AND a bigger step compounded the drift. 1.5e-6 matches run1, isolating LR
# as the only variable against run2.
#
# If |actor_loss| still climbs, the next lever is --kl-type per_token, which is
# length-normalised -- mean_t((logP-logPref)^2) instead of (sum_t ...)^2.
#
# --- inherited from run2 ---
# run1 (17365534) LOWERED e2e FVE_nrm_meannorm from +0.5207 to
# +0.4922 -- 300 steps of optimising critic reconstruction produced WORSE
# reconstruction. Two causes, both fixed here:
#
# 1. ONE GROUP PER STEP. run1 used --rollout-batch 1, so every gradient came
#    from a single activation vector and consecutive updates pulled in
#    unrelated directions. Now 8 groups/step. Enabled by two memory fixes:
#      - frozen critic loaded bf16 (it is NOT FSDP-wrapped when frozen, so at
#        fp32 it sat as a 22.4 GB unsharded copy on every rank)
#      - reference model FSDP-sharded (was a 16 GB unsharded copy per rank)
#    Static memory 70.4 -> 47.2 GB/GPU.
#
# 2. TRUNCATION CLIFF. --max-response-len 150 cut 26.6% of generations before
#    their closing tag -> FAILED_EXTRACTION_REWARD=-2.0. Those outliers inflate
#    the per-group std, compressing the spread among VALID rollouts ~7x (0.44
#    vs 3.14 in simulation) -- so GRPO heard "do not truncate" loudly and
#    "reconstruct better" as a whisper. run1 duly got shorter (median 127->110)
#    without getting better. At 224: 1 failure in 320 probe rollouts (0.3%).
#    NOTE --max-length must exceed prompt(~105) + 224, hence 400 not 320.
#
# Rollout also now runs on ALL FOUR ranks (was rank 0 only, 3 GPUs idle);
# ordering after the all_gather is unit-tested since gold_idx = s_idx // N
# makes sample order load-bearing.
#
# TARGET: beat SFT's e2e FVE +0.5207. Between +0.4922 and +0.5207 means the
# fixes helped but did not close the gap.
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

torchrun --nproc_per_node=4 --master_port=29651 \
    -m nla.training.train_rl \
    --data "$PROJ/data/rl_train_8b_ufw_joined.parquet" \
    --model-name Qwen/Qwen3-8B \
    --actor-ckpt  "$PROJ/checkpoints/actor_sft_8b_s50fix" \
    --critic-ckpt "$PROJ/checkpoints/critic_sft_8b_s50" \
    --output-dir  "$PROJ/checkpoints/rl_8b_run3" \
    --injection-scale 300 --freeze-critic \
    --n-samples 8 --rollout-batch 8 --micro-batch-size 1 \
    --lr-actor 1.5e-6 --lr-critic 1.5e-6 \
    --kl-coef 0.01 --kl-type per_seq \
    --max-response-len 224 --max-length 400 \
    --num-steps 300 --save-every 50 \
    --param-dtype float32 --fsdp

echo "=== RL RUN3 DONE ==="
