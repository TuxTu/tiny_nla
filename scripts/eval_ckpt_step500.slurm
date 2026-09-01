#!/bin/bash
#SBATCH --job-name=eval500
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/eval500_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/eval500_%j.err
#
# Mid-curve datapoint: exp1long's step-500 checkpoint (gap +0.2444) against the
# step-250 one we already measured (gap +0.0522). Two points on the
# gap-vs-generation-quality curve beat one, and this does not wait for step 1000.
# Evaluated on a COPY: exp1long rewrites checkpoints/code_exp1_long every 250
# steps, so reading it live would race the step-750 save.
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
CKPT=$PROJ/checkpoints/code_exp1_step500
for tag in "" "--shuffle-control"; do
  name=$([ -z "$tag" ] && echo correct || echo shuf)
  echo "=== step500 $name ==="
  python scripts/eval_code_recon.py --actor-ckpt "$CKPT" \
      --data "$PROJ/data/coderec/exp1_eval.parquet" \
      --n-samples 300 $tag --out "$PROJ/logs/code_exp1_step500_${name}.jsonl"
done
