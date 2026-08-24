#!/bin/bash
#SBATCH --job-name=nla-actor-fix
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=16:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/actor_fix_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/actor_fix_%j.err
#
# Actor retrain on the SAME split50 data as job 17358927, but with the
# response-only loss mask (landed 13:08, i.e. AFTER that run) and
# --injection-scale 300.
#
# 17358927 trained with sft_loss(logits, labels) and no loss_mask: CE was
# averaged over prompt+response+pad instead of response alone, so both the
# reported CE and the GRADIENT on response tokens came out ~2x too small, and
# right-pad (== eos, trivially predictable) supplied ~15% free targets. Its
# vector-ablation gap was +0.0013. The ladder's full rung, same data, fixed
# loss, only 300 steps, reached +0.1451 held-out.
#
# 3 epochs here because the ladder's gap was still climbing monotonically at
# step 300 (+0.0977 at 65k pairs -> +0.1451 at 125k) and held-out CE had not
# flattened (1.5997 -> 1.5165 -> 1.4751).
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
OUT="$PROJ/checkpoints/actor_sft_8b_s50fix"
GLOBAL_BATCH=256
EPOCHS=3
N_ROWS=$(python -c "import pyarrow.parquet as pq;print(pq.ParquetFile('$TRAIN').metadata.num_rows)")
NUM_STEPS=$(( N_ROWS * EPOCHS / GLOBAL_BATCH ))
echo "train rows=$N_ROWS  epochs=$EPOCHS  -> num_steps=$NUM_STEPS"

LOCAL_DIR="/tmp/nla_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_DIR"
cp "$TRAIN" "$TRAIN.nla_meta.yaml" "$EVAL" "$EVAL.nla_meta.yaml" "$LOCAL_DIR/"

torchrun --nproc_per_node=4 --master_port=29677 \
    -m nla.training.train_actor_sft \
    --data "$LOCAL_DIR/$(basename "$TRAIN")" \
    --model-name Qwen/Qwen3-8B \
    --output-dir "$OUT" \
    --injection-scale 300 \
    --micro-batch-size 4 --global-batch-size $GLOBAL_BATCH \
    --lr 2e-5 --min-lr 2e-6 --lr-warmup-iters 50 --lr-decay-style cosine \
    --num-steps $NUM_STEPS --max-length 512 --save-every 250 \
    --eval-data "$LOCAL_DIR/$(basename "$EVAL")" \
    --eval-every 50 --eval-samples 512 --eval-batch-size 8 \
    --param-dtype float32 --fsdp
rm -rf "$LOCAL_DIR"

echo "---- vector ablation on HELD-OUT rows ----"
python scripts/actor_vector_ablation.py \
    --actor-ckpt "$OUT" --data "$EVAL" --sidecar "$EVAL.nla_meta.yaml" \
    --injection-scale 300 --n-samples 512 --batch-size 8

echo "=== ACTOR S50FIX DONE ==="
