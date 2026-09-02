#!/bin/bash
#SBATCH --job-name=nla-e2e-eval
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/e2e_eval_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/e2e_eval_%j.err
# End-to-end NLA FVE: gold vector -> actor generates explanation -> critic reconstructs.
# This is the real pipeline metric; the +65.7% number was critic-only on GOLD labels.
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

ACTOR=${ACTOR:-$PROJ/checkpoints/actor_sft_8b_s50}
CRITIC=${CRITIC:-$PROJ/checkpoints/critic_sft_8b_s50}
N=${N:-500}

echo "actor=$ACTOR"
echo "critic=$CRITIC"

for MODE in --greedy ""; do
  LABEL=$([ -n "$MODE" ] && echo "greedy" || echo "sampled(T=1.0)")
  echo ""
  echo "############ END-TO-END, decoding = $LABEL ############"
  python scripts/nla_eval_fve.py \
      --model-name Qwen/Qwen3-8B \
      --actor-ckpt  "$ACTOR" \
      --critic-ckpt "$CRITIC" \
      --sidecar "$PROJ/data/av_eval_8b_ufw_joined.parquet.nla_meta.yaml" \
      --data    "$PROJ/data/av_eval_8b_ufw_joined.parquet" \
      --n-samples "$N" --max-new-tokens 256 $MODE \
      --jsonl "$PROJ/logs/e2e_${SLURM_JOB_ID}_$([ -n "$MODE" ] && echo greedy || echo sampled).jsonl"
done

echo ""
echo "############ CRITIC-ONLY on the SAME held-out rows (upper bound) ############"
python scripts/nla_eval_fve.py \
    --model-name Qwen/Qwen3-8B --critic-only \
    --critic-ckpt "$CRITIC" \
    --sidecar "$PROJ/data/ar_eval_8b_ufw_joined.parquet.nla_meta.yaml" \
    --data    "$PROJ/data/ar_eval_8b_ufw_joined.parquet" \
    --n-samples "$N"

echo "=== E2E EVAL DONE ==="
