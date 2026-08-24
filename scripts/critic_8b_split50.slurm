#!/bin/bash
#SBATCH --job-name=nla-critic-s50
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/critic_s50_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/critic_s50_%j.err
# Critic (AR) SFT on half A of the labelled AV pool -- 12,500 documents, doc-disjoint
# from the actor's half B. Identical hyperparameters to the run that reached +65.72%
# on 249k rows; only the data volume changes (249k -> 125k).
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
export HF_HOME=$PROJ/cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJ"

TRAIN="$PROJ/data/split50/ar_sft_train.parquet"
EVAL="$PROJ/data/ar_eval_8b_ufw_joined.parquet"
INIT="$PROJ/checkpoints/critic_init_8b_ufw"
OUT="$PROJ/checkpoints/critic_sft_8b_s50"
GLOBAL_BATCH=256
MICRO=${MICRO:-16}
N_ROWS=$(python -c "import pyarrow.parquet as pq;print(pq.ParquetFile('$TRAIN').metadata.num_rows)")
NUM_STEPS=$((N_ROWS / GLOBAL_BATCH))
echo "train rows=$N_ROWS  -> num_steps=$NUM_STEPS (1 epoch)"

torchrun --nproc_per_node=4 --master_port=29618 \
    -m nla.training.train_critic_sft \
    --data "$TRAIN" --eval-data "$EVAL" \
    --model-name "$INIT" --layer-index 24 \
    --output-dir "$OUT" \
    --micro-batch-size $MICRO --global-batch-size $GLOBAL_BATCH \
    --lr 2e-5 --min-lr 2e-6 --lr-warmup-iters 50 --lr-decay-style cosine \
    --num-steps $NUM_STEPS --max-length 512 --value-head-depth 1 \
    --save-every 100000 --save-best \
    --eval-every 50 --eval-samples 2048 --eval-batch-size 32 \
    --param-dtype float32 --fsdp

echo "=== CRITIC SPLIT50 DONE ==="
