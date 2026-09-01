#!/bin/bash
#SBATCH --job-name=bestofn
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/bestofn_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/bestofn_%j.err
#
# Is the 0/290 exact-match ceiling SEARCH or CAPACITY?
#
# Every number so far is pass@1 -- one greedy generation per vector. Sample 32
# instead and ask whether ANY of them is right. Vec2Text (2310.06816) shows
# single-pass inversion is the hard way (~30 BLEU) while iterative search gets
# 92% exact; best-of-n is the crudest version of that search.
#
#   greedy            what we measured
#   best-of-32 by cos rerank on cos(v_target, E(candidate)) -- a real method,
#                     uses only the target vector we are given
#   ORACLE best-of-32 rerank on true similarity to the target code -- cheating,
#                     but upper-bounds what ANY reranker over these samples can do
#
# If the ORACLE is still ~0, the samples do not contain the right program and a
# vec2text-style corrector would have nothing to correct toward.
#
# Runs on the PRESERVED step-1000 checkpoint, so it does not touch exp1ep3's
# live output dir.
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

python scripts/eval_best_of_n.py \
    --actor-ckpt "$PROJ/checkpoints/code_exp1_step1000" \
    --data "$PROJ/data/coderec/exp1_eval.parquet" \
    --n-samples 100 --n-candidates 32 --temperature 0.8 \
    --out "$PROJ/logs/bestofn_step1000.json"
