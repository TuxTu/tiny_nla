#!/bin/bash
#SBATCH --job-name=bon-ast
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=160G
#SBATCH --time=02:30:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/bon_ast_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/bon_ast_%j.err
#
# best-of-32 with AST metrics, over the FULL 300-row window.
#
# Two fixes over the earlier runs:
#   - 300 rows, not 100. bon-c used 100 and its window contained only 1 of the
#     3 rows known to be achievable (idx 28, 130, 143), so its 0/100 measured
#     sample size rather than model quality.
#   - --centre-mean. Candidates are encoded RAW by the frozen model, but exp1s
#     stores CENTRED targets; without subtracting the same mean the cosine
#     compares incompatible representations and the reranking arm is noise.
#
# Reports exact, norm-AST equal (exact modulo renaming, literals kept) and
# skeleton>0.9 (graded shape) for greedy / best-of-32-by-cos / oracle.
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
    --actor-ckpt "$PROJ/checkpoints/code_exp1s" \
    --data "$PROJ/data/coderec/exp1s_eval.parquet" \
    --centre-mean "$PROJ/data/coderec/exp1s_mean.npy" \
    --n-samples 300 --n-candidates 32 --temperature 0.8 \
    --out "$PROJ/logs/bon_ast_exp1s.json"
