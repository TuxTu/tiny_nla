#!/bin/bash
#SBATCH --job-name=nla-critic-v2
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/critic_v2_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/critic_v2_%j.err

# Critic SL, Qwen3-8B, take 2 — after fixing:
#   1. bf16 master weights  -> fp32 masters, bf16 compute (--param-dtype float32)
#   2. double truncation    -> prepared 25-layer checkpoint loaded as-is (K=24)
#   3. FSDP without DistributedSampler -> ranks now see disjoint shards
#   4. final _save() called on rank 0 only -> collective deadlock + SIGABRT,
#      which silently destroyed every previous run's FINAL checkpoint
#   5. THE BIG ONE: build_training_data paired explained<->vectors POSITIONALLY
#      while extract_vectors emits sorted(doc_id) order -> 99.96% of rows had an
#      explanation attached to the wrong activation vector. Data rebuilt with a
#      real (doc_id, n_raw_tokens) join.
# Adds held-out FVE on 2048 rows from 914 documents with zero train overlap.
#
# Reference: original repo, Qwen2.5-7B, 21 layers, 250k rows, ~1000 steps,
# batch 256, lr 2e-5->2e-6 cosine warmup 50 => FVE 37.5%.

set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
export HF_HOME=$PROJ/cache/huggingface
export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJ"

TRAIN="$PROJ/data/ar_sft_train_8b_ufw_joined.parquet"
EVAL="$PROJ/data/ar_eval_8b_ufw_joined.parquet"
INIT="$PROJ/checkpoints/critic_init_8b_ufw"
OUT="$PROJ/checkpoints/critic_sft_8b_v3"

GLOBAL_BATCH=256
MICRO=${MICRO:-16}
N_ROWS=$(python -c "import pyarrow.parquet as pq; print(pq.ParquetFile('$TRAIN').metadata.num_rows)")
NUM_STEPS=$((N_ROWS / GLOBAL_BATCH))

echo "rows=$N_ROWS  global_batch=$GLOBAL_BATCH  steps=$NUM_STEPS (1 epoch)  micro=$MICRO"
python -c "import json;print('critic init num_hidden_layers =', json.load(open('$INIT/config.json'))['num_hidden_layers'])"
nvidia-smi --query-gpu=name,memory.total --format=csv

torchrun --nproc_per_node=4 --master_port=29617 \
    -m nla.training.train_critic_sft \
    --data "$TRAIN" --eval-data "$EVAL" \
    --model-name "$INIT" --layer-index 24 \
    --output-dir "$OUT" \
    --micro-batch-size $MICRO --global-batch-size $GLOBAL_BATCH \
    --lr 2e-5 --min-lr 2e-6 --lr-warmup-iters 50 --lr-decay-style cosine \
    --num-steps $NUM_STEPS --max-length 512 --value-head-depth 1 \
    --save-every 100000 --save-best \
    --eval-every 100 --eval-samples 2048 --eval-batch-size 32 \
    --param-dtype float32 --fsdp

echo "=== TRAINING COMPLETE ==="
