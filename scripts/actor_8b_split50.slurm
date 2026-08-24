#!/bin/bash
#SBATCH --job-name=nla-actor-s50
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/actor_s50_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/actor_s50_%j.err
# Actor (AV) SFT on half B of the labelled AV pool -- 12,500 documents, doc-disjoint
# from the critic's half A and from the eval / RL pools.
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
GLOBAL_BATCH=256
N_ROWS=$(python -c "import pyarrow.parquet as pq;print(pq.ParquetFile('$TRAIN').metadata.num_rows)")
NUM_STEPS=$((N_ROWS / GLOBAL_BATCH))
echo "train rows=$N_ROWS  -> num_steps=$NUM_STEPS (1 epoch)"

LOCAL_DIR="/tmp/nla_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_DIR"
cp "$TRAIN" "$TRAIN.nla_meta.yaml" "$EVAL" "$EVAL.nla_meta.yaml" "$LOCAL_DIR/"
echo "data staged: $(du -sh "$LOCAL_DIR" | cut -f1)"

torchrun --nproc_per_node=4 --master_port=29543 \
    -m nla.training.train_actor_sft \
    --data "$LOCAL_DIR/$(basename "$TRAIN")" \
    --model-name Qwen/Qwen3-8B \
    --output-dir "$PROJ/checkpoints/actor_sft_8b_s50" \
    --micro-batch-size 4 --global-batch-size $GLOBAL_BATCH \
    --lr 2e-5 --min-lr 2e-6 --lr-warmup-iters 50 --lr-decay-style cosine \
    --num-steps $NUM_STEPS --max-length 512 --save-every 200 \
    --eval-data "$LOCAL_DIR/$(basename "$EVAL")" \
    --eval-every 50 --eval-samples 512 --eval-batch-size 8 \
    --param-dtype float32 --fsdp

echo "=== ACTOR SPLIT50 DONE ==="
rm -rf "$LOCAL_DIR"
