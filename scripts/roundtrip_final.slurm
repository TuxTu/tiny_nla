#!/bin/bash
#SBATCH --job-name=rt-final
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/rt_final_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/rt_final_%j.err
#
# Round-trip on exp1long's FINAL (step-1000) generations, plus a re-run of the
# two earlier checkpoints with the corrected (normalised) FVE, so all three
# points of the gap-vs-quality curve are in the same units.
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
for pt in "step250:logs/code_exp1_step250_correct.jsonl:logs/code_exp1_step250_shuf.jsonl" \
          "step500:logs/code_exp1_step500_correct.jsonl:logs/code_exp1_step500_shuf.jsonl" \
          "step1000:logs/code_exp1_gen.jsonl:logs/code_exp1_gen_shuf.jsonl"; do
  tag=${pt%%:*}; rest=${pt#*:}; g=${rest%%:*}; gs=${rest#*:}
  [ -s "$g" ] || { echo "### $tag: $g missing, skipping"; continue; }
  echo; echo "########## $tag ##########"
  python scripts/eval_code_roundtrip.py --gen "$g" --gen-shuf "$gs" \
      --out "logs/roundtrip_exp1_${tag}.json" || echo "($tag failed)"
done
