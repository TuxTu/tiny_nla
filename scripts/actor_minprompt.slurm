#!/bin/bash
#SBATCH --job-name=nla-actor-minp
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=01:30:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/actor_minp_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/actor_minp_%j.err
#
# Feasibility: can the actor learn vector-conditioning from a 15-token prompt?
#
# The SFT preamble is ~104 tokens and IDENTICAL on every example, so it cannot
# discriminate between examples -- post-SFT it can only be a fixed mode selector.
# The OOD ablation (job 17421679) showed the trained actor degrades when the
# preamble is stripped, but that measured prompt brittleness, not the
# counterfactual: format lives in the TARGETS (every gold response opens with
# "<explanation>"), so an actor trained on the short prompt should relearn it.
#
# Prompt here is "<concept>@</concept>" only = 15 tokens. No prefill needed:
# the response teaches the tag. Canonical neighbours (29,522) are unchanged.
#
# 500 steps ~= 1 epoch. Reference curve from the 104-token run (job 17361668):
#   step 250 -> CE 1.4227    step 500 -> CE 1.3464    floor at step 950 -> 1.3072
# Its final vector-ablation gap was +0.4593.
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
export HF_HOME=$PROJ/cache/huggingface
export TORCHINDUCTOR_CACHE_DIR=$PROJ/cache/torchinductor   # else torch defaults to /tmp/torchinductor_$USER
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJ"

LOCAL_DIR="$PROJ/tmp/nla_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_DIR"
trap 'rm -rf "$LOCAL_DIR"' EXIT   # $PROJ/tmp is NOT node-wiped
cp "$PROJ/data/minprompt/av_sft_train.parquet"* "$LOCAL_DIR/"
cp "$PROJ/data/minprompt/av_eval.parquet"*      "$LOCAL_DIR/"

torchrun --nproc_per_node=4 --master_port=$((20000 + RANDOM % 10000)) \
  -m nla.training.train_actor_sft \
    --data "$LOCAL_DIR/av_sft_train.parquet" \
    --eval-data "$LOCAL_DIR/av_eval.parquet" \
    --model-name Qwen/Qwen3-8B \
    --output-dir "$PROJ/checkpoints/actor_sft_8b_minprompt" \
    --injection-scale 300 \
    --global-batch-size 256 --micro-batch-size 4 \
    --num-steps 250 --lr 2e-5 \
    --eval-every 50 --eval-samples 512 --eval-batch-size 8 --eval-gap \
    --save-every 125 --fsdp
