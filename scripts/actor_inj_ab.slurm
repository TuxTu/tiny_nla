#!/bin/bash
#SBATCH --job-name=nla-inj-ab
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=03:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/inj_ab_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/inj_ab_%j.err
# A/B on injection_scale, with the loss-masking fix applied to BOTH arms so the
# only difference between them is the scale.
#
#   arm A: injection_scale=300  (original's rule: round number just above the
#                                dataset mean norm of 268)
#   arm B: injection_scale=160  (what we trained at: 60% of the mean)
#
# Metric is NOT train loss -- it is the real-vs-shuffled CE gap, which is what
# the original treats as the signal ("real-vs-rand gap (~0.21) is the signal,
# not train loss"). Ours at 487 steps with scale=160 was 0.0013.
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
export HF_HOME=$PROJ/cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJ"

TRAIN="$PROJ/data/split50/av_sft_train.parquet"
EVAL="$PROJ/data/av_eval_8b_ufw_joined.parquet"
STEPS=${STEPS:-150}

for SCALE in 300 160; do
  OUT="$PROJ/checkpoints/actor_inj${SCALE}"
  echo ""
  echo "################ ARM injection_scale=$SCALE ################"
  torchrun --nproc_per_node=4 --master_port=$((29700 + SCALE % 100)) \
      -m nla.training.train_actor_sft \
      --data "$TRAIN" \
      --model-name Qwen/Qwen3-8B \
      --output-dir "$OUT" \
      --injection-scale "$SCALE" \
      --micro-batch-size 4 --global-batch-size 256 \
      --lr 2e-5 --min-lr 2e-6 --lr-warmup-iters 15 --lr-decay-style cosine \
      --num-steps "$STEPS" --max-length 512 --save-every 100000 \
      --eval-data "$EVAL" --eval-every 50 --eval-samples 512 --eval-batch-size 8 \
      --param-dtype float32 --fsdp

  echo "---- vector-conditioning ablation, scale=$SCALE ----"
  python scripts/actor_vector_ablation.py \
      --actor-ckpt "$OUT" --data "$EVAL" --sidecar "$EVAL.nla_meta.yaml" \
      --injection-scale "$SCALE" --n-samples 512 --batch-size 8
done

echo "=== INJ A/B DONE ==="
