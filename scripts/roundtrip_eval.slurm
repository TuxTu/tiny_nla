#!/bin/bash
#SBATCH --job-name=roundtrip
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/roundtrip_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/roundtrip_%j.err
#
# v -> AV -> code' -> E(code') -> v'. The direct analogue of the text pipeline's
# FVE, and free of the ~0.15 boilerplate floor that makes lexical token overlap
# nearly uninterpretable for Python.
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

echo "########## exp1 @250 steps (gap +0.0522, 40 samples) ##########"
python scripts/eval_code_roundtrip.py \
    --gen logs/code_exp1_gen.jsonl --gen-shuf logs/code_exp1_gen_shuf.jsonl \
    --out logs/roundtrip_exp1_step250.json

echo; echo "########## exp1long @500 steps (gap +0.2444, 300 samples) ##########"
python scripts/eval_code_roundtrip.py \
    --gen logs/code_exp1_step500_correct.jsonl --gen-shuf logs/code_exp1_step500_shuf.jsonl \
    --out logs/roundtrip_exp1_step500.json
