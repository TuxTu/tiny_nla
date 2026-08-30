#!/bin/bash
#SBATCH --job-name=nla-av-ladder
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/av_ladder_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/av_ladder_%j.err
# WHERE does the vector->text channel die?
#
# The 64-row overfit run reached CE 0.0001 with a +0.198 shuffled gap, so the
# injection path, the FSDP hook and the gradient flow are all functional -- the
# actor CAN condition on the vector. On the full 124k set the same recipe gets
# +0.003. So this is a generalisation/data problem, not a wiring problem.
#
# Fixed optimisation budget (300 steps x 256 = 76.8k samples) at every rung, so
# the ONLY thing varying is how many distinct (vector, explanation) pairs the
# model sees. Each rung is scored twice:
#   - on its OWN training rows  -> did it learn the mapping at all?
#   - on held-out eval rows     -> did that mapping generalise?
# Gap big on train / zero on held-out = pure data starvation.
# Gap zero on both past some N   = the model gives up on the channel as the
#                                  marginal gets easier to fit.
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

EVAL="$PROJ/data/av_eval_8b_ufw_joined.parquet"
STEPS=${STEPS:-300}
SCALE=300

for N in 1024 8192 65536 full; do
  if [ "$N" = "full" ]; then TRAIN="$PROJ/data/split50/av_sft_train.parquet"
  else TRAIN="$PROJ/data/split50/av_ladder_${N}.parquet"; fi
  OUT="$PROJ/checkpoints/actor_ladder_${N}"
  echo ""
  echo "################ RUNG N=$N ################"
  torchrun --nproc_per_node=4 --master_port=29822 \
      -m nla.training.train_actor_sft \
      --data "$TRAIN" --model-name Qwen/Qwen3-8B --output-dir "$OUT" \
      --injection-scale "$SCALE" \
      --micro-batch-size 4 --global-batch-size 256 \
      --lr 2e-5 --min-lr 2e-6 --lr-warmup-iters 30 --lr-decay-style cosine \
      --num-steps "$STEPS" --max-length 512 --save-every 100000 \
      --eval-data "$EVAL" --eval-every 100 --eval-samples 256 --eval-batch-size 8 \
      --param-dtype float32 --fsdp

  echo "---- N=$N : ablation on TRAIN rows (can it learn the mapping?) ----"
  python scripts/actor_vector_ablation.py \
      --actor-ckpt "$OUT" --data "$TRAIN" --sidecar "$TRAIN.nla_meta.yaml" \
      --injection-scale "$SCALE" --n-samples 256 --batch-size 8
  echo "---- N=$N : ablation on HELD-OUT rows (did it generalise?) ----"
  python scripts/actor_vector_ablation.py \
      --actor-ckpt "$OUT" --data "$EVAL" --sidecar "$EVAL.nla_meta.yaml" \
      --injection-scale "$SCALE" --n-samples 256 --batch-size 8
  rm -rf "$OUT"
done

echo "=== AV LADDER DONE ==="
