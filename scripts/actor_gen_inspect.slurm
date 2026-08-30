#!/bin/bash
#SBATCH --job-name=nla-gen-inspect
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/gen_inspect_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/gen_inspect_%j.err
# Read the retrained actor's ACTUAL output. The +0.4593 ablation gap says the
# vector carries information; it does not say the text is any good, nor that
# greedy decoding stopped collapsing (old actor: 25 unique / 500).
# Same 500 held-out rows, greedy first (the collapse-prone mode) then sampled.
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

ACTOR=$PROJ/checkpoints/actor_sft_8b_s50fix
DATA=$PROJ/data/av_eval_8b_ufw_joined.parquet
N=500

for MODE in --greedy ""; do
  TAG=$([ -n "$MODE" ] && echo greedy || echo sampled)
  echo ""
  echo "########## decoding = $TAG ##########"
  python scripts/actor_gen_inspect.py \
      --actor-ckpt "$ACTOR" --data "$DATA" --sidecar "$DATA.nla_meta.yaml" \
      --injection-scale 300 --n-samples $N --batch-size 16 \
      --max-new-tokens 200 --n-show 8 $MODE \
      --jsonl "$PROJ/logs/gen_${SLURM_JOB_ID}_${TAG}.jsonl"
done
echo "=== GEN INSPECT DONE ==="
