#!/bin/bash
#SBATCH --job-name=nla-prompt-abl
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/prompt_abl_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/prompt_abl_%j.err
#
# Prompt-length ablation on the ALREADY TRAINED actor. No training.
#
# The ~90-token researcher preamble is constant across every SFT example, so it
# cannot discriminate between examples — post-SFT it can only be a fixed mode
# selector. This sweeps 7 prompts from that preamble down to a bare injection
# marker and reports surviving end-to-end FVE.
#
# Caveat baked into the read: the actor never saw these prompts, so a drop
# confounds "preamble was load-bearing" with "brittle to prompt shift". Only a
# retrain separates them. FVE holding up is the informative outcome.
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

ACTOR=${ACTOR:-$PROJ/checkpoints/actor_sft_8b_s50fix}
CRITIC=${CRITIC:-$PROJ/checkpoints/critic_sft_8b_s50}
N=${N:-200}
TAG=$(basename "$ACTOR")

echo "actor=$ACTOR"
echo "critic=$CRITIC  n=$N"

python scripts/actor_prompt_ablation.py \
    --model-name Qwen/Qwen3-8B \
    --actor-ckpt "$ACTOR" \
    --critic-ckpt "$CRITIC" \
    --data "$PROJ/data/av_eval_8b_ufw_joined.parquet" \
    --n-samples "$N" \
    --batch-size 8 \
    --greedy \
    --jsonl "$PROJ/logs/prompt_abl_${TAG}_${SLURM_JOB_ID}.jsonl"

echo
echo "=========== SHUFFLE CONTROL ==========="
python scripts/actor_prompt_ablation.py \
    --model-name Qwen/Qwen3-8B \
    --actor-ckpt "$ACTOR" \
    --critic-ckpt "$CRITIC" \
    --data "$PROJ/data/av_eval_8b_ufw_joined.parquet" \
    --n-samples "$N" \
    --batch-size 8 \
    --greedy --shuffle-control \
    --only full,tags_prefill,bare_raw \
    --jsonl "$PROJ/logs/prompt_abl_${TAG}_${SLURM_JOB_ID}_shuf.jsonl"
