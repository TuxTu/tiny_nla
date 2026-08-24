#!/bin/bash
#SBATCH --job-name=nla-pre-rl
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/pre_rl_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/pre_rl_%j.err
#
# Two prerequisites before spending GPUs on GRPO.
#
# A. Does injection scale actually matter at inference?
#    RL sidecars carry injection_scale: null, so train_rl.py silently used
#    2.5*sqrt(4096)=160 while actor_sft_8b_s50fix was SFT'd at 300. The earlier
#    160-vs-300 A/B was run at 150 steps -- too few for the vector channel to
#    exist at all -- so it could not have detected a scale effect. Measure the
#    same checkpoint at both scales.
#
# B. Do the two SFT models COUPLE? RL's reward is the critic reconstructing the
#    actor's generated explanation, so a deeply negative e2e FVE means GRPO is
#    optimising noise. The last e2e number (-27.93%) predates the actor fix.
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
export HF_HOME=$PROJ/cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJ"

ACTOR=$PROJ/checkpoints/actor_sft_8b_s50fix
CRITIC=$PROJ/checkpoints/critic_sft_8b_s50
DATA=$PROJ/data/av_eval_8b_ufw_joined.parquet

echo "########## A. injection-scale sensitivity (same ckpt, two scales) ##########"
for S in 300 160; do
  echo "---- scale $S ----"
  python scripts/actor_vector_ablation.py \
      --actor-ckpt "$ACTOR" --data "$DATA" --sidecar "$DATA.nla_meta.yaml" \
      --injection-scale $S --n-samples 256 --batch-size 8
done

echo ""
echo "########## B. end-to-end coupling: actor -> critic ##########"
echo "--- B0. critic-only on GOLD explanations (upper bound, same rows) ---"
python scripts/nla_eval_fve.py \
    --model-name Qwen/Qwen3-8B --critic-only --critic-ckpt "$CRITIC" \
    --sidecar "$DATA.nla_meta.yaml" --data "$DATA" --n-samples 500

for MODE in --greedy ""; do
  TAG=$([ -n "$MODE" ] && echo greedy || echo sampled)
  echo ""
  echo "--- B1. e2e, decoding=$TAG ---"
  python scripts/nla_eval_fve.py \
      --model-name Qwen/Qwen3-8B \
      --actor-ckpt "$ACTOR" --critic-ckpt "$CRITIC" \
      --sidecar "$DATA.nla_meta.yaml" --data "$DATA" \
      --n-samples 500 --max-new-tokens 256 $MODE \
      --jsonl "$PROJ/logs/prerl_${SLURM_JOB_ID}_${TAG}.jsonl"
done
echo "=== PRE-RL CHECKS DONE ==="
