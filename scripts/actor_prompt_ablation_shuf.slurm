#!/bin/bash
#SBATCH --job-name=nla-abl-shuf
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/prompt_abl_shuf_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/prompt_abl_shuf_%j.err
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
export HF_HOME=$PROJ/cache/huggingface
export TORCHINDUCTOR_CACHE_DIR=$PROJ/cache/torchinductor
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJ"
python scripts/actor_prompt_ablation.py \
    --model-name Qwen/Qwen3-8B \
    --actor-ckpt "$PROJ/checkpoints/actor_sft_8b_s50fix" \
    --critic-ckpt "$PROJ/checkpoints/critic_sft_8b_s50" \
    --data "$PROJ/data/av_eval_8b_ufw_joined.parquet" \
    --n-samples 200 --batch-size 8 --greedy --shuffle-control \
    --only full,no_persona,tags_prefill,bare_chat,bare_raw \
    --jsonl "$PROJ/logs/prompt_abl_shuf_${SLURM_JOB_ID}.jsonl"
