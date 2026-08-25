#!/bin/bash
#SBATCH --job-name=nla-e2e-v2
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/e2e_v2_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/e2e_v2_%j.err
# End-to-end NLA FVE, rerun after fixing the prompt-leak in extract_explanation's
# input (scripts/nla_eval_fve.py decoded prompt+completion, so the "explanation"
# scored by the critic was ~60 tokens of instruction boilerplate + the real text).
#
# Order matters: the GOLD upper bound runs FIRST on the SAME rows as the e2e
# passes, so the actor's contribution is the difference between two numbers
# measured on one row set with one critic.
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
export HF_HOME=$PROJ/cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJ"

ACTOR=${ACTOR:-$PROJ/checkpoints/actor_sft_8b_s50fix}
CRITIC=${CRITIC:-$PROJ/checkpoints/critic_sft_8b_s50}
DATA=$PROJ/data/av_eval_8b_ufw_joined.parquet
N=${N:-500}
echo "actor=$ACTOR"; echo "critic=$CRITIC"; echo "n=$N"

echo ""
echo "########## A. CRITIC-ONLY on GOLD explanations, same rows (UPPER BOUND) ##########"
python scripts/nla_eval_fve.py \
    --model-name Qwen/Qwen3-8B --critic-only \
    --critic-ckpt "$CRITIC" \
    --sidecar "$DATA.nla_meta.yaml" --data "$DATA" --n-samples "$N"

for MODE in --greedy ""; do
  TAG=$([ -n "$MODE" ] && echo greedy || echo sampled)
  echo ""
  echo "########## B. END-TO-END, decoding = $TAG ##########"
  python scripts/nla_eval_fve.py \
      --model-name Qwen/Qwen3-8B \
      --actor-ckpt "$ACTOR" --critic-ckpt "$CRITIC" \
      --sidecar "$DATA.nla_meta.yaml" --data "$DATA" \
      --n-samples "$N" --max-new-tokens 256 $MODE \
      --jsonl "$PROJ/logs/e2ev2_${SLURM_JOB_ID}_${TAG}.jsonl"
done

echo "=== E2E V2 DONE ==="
