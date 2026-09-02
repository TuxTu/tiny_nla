#!/bin/bash
#SBATCH --job-name=nla-rlck
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/rlck_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/rlck_%j.err
# Is the RL checkpoint better, or just shorter?
#
# extract_fail went 1-3/8 early to 0/8 for four consecutive checkpoints while
# reward improved. Two readings: the actor reconstructs better, or it learned to
# write under the 150-token cap and is dodging FAILED_EXTRACTION_REWARD=-2.0.
# Reward alone cannot separate them; response length + retrieval accuracy can.
#
# SFT baseline (job 17363730, greedy): median 127 tok, 8.0% over 150,
# 500/500 unique, retrieval 68.4%.
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
export HF_HOME=$PROJ/cache/huggingface
export TORCHINDUCTOR_CACHE_DIR=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/torchinductor   # else torch defaults to /tmp/torchinductor_$USER
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJ"
CKPT=${CKPT:-$PROJ/checkpoints/rl_snapshots/step50_actor}  # never /tmp: forbidden on Berzelius AND node-local, so a login-node copy is invisible here
DATA=$PROJ/data/av_eval_8b_ufw_joined.parquet

echo "########## generation quality: $CKPT ##########"
python scripts/actor_gen_inspect.py \
    --actor-ckpt "$CKPT" --data "$DATA" --sidecar "$DATA.nla_meta.yaml" \
    --injection-scale 300 --n-samples 300 --batch-size 16 \
    --max-new-tokens 200 --greedy --n-show 3 \
    --jsonl "$PROJ/logs/rlck_${SLURM_JOB_ID}_greedy.jsonl"

echo ""
echo "########## vector conditioning (SFT was +0.4593) ##########"
python scripts/actor_vector_ablation.py \
    --actor-ckpt "$CKPT" --data "$DATA" --sidecar "$DATA.nla_meta.yaml" \
    --injection-scale 300 --n-samples 256 --batch-size 8
echo "=== RL CKPT EVAL DONE ==="
