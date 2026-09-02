#!/bin/bash
#SBATCH --job-name=orig-critic-8b
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/orig_critic_sft_8b_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/orig_critic_sft_8b_%j.err

# Critic SFT using the ORIGINAL natural_language_autoencoders repo + miles framework
# on Qwen3-8B with our AR-SFT data (449k rows).
# Uses Ray + FSDP (miles manages distributed workers internally, no torchrun).
# 4× A100-80GB, 1 epoch ≈ 1754 steps.

set -euo pipefail

CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export HF_HOME=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/huggingface
export TORCHINDUCTOR_CACHE_DIR=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/torchinductor   # else torch defaults to /tmp/torchinductor_$USER
mkdir -p "$HF_HOME"

PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
ORIG_REPO=/proj/assert-berzelius/users/x_tuhan/garage/natural_language_autoencoders
MILES_REPO=/proj/assert-berzelius/users/x_tuhan/garage/miles

# Ensure original NLA imports take precedence over our local nla/.
# Run from a scratch dir under $PROJ/tmp so Python's CWD isn't our project dir.
WORKDIR="$PROJ/tmp/orig_critic_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"
trap 'rm -rf "$WORKDIR"' EXIT   # $PROJ/tmp is NOT node-wiped
cd "$WORKDIR"

echo "=== Original Repo Critic SFT: Qwen3-8B ==="
echo "Node: $(hostname)"
echo "Data: $PROJ/data/ar_sft_train_8b.parquet (449k rows)"
echo "Workdir: $WORKDIR"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# --- Step 1: Prepare critic checkpoint (one-time, CPU-only) ---
CRITIC_INIT="$PROJ/checkpoints/critic_init_8b_orig"
if [ ! -f "$CRITIC_INIT/nla_meta.yaml" ]; then
    echo "=== Step 1: Preparing critic checkpoint (truncate Qwen3-8B to 25 layers) ==="
    rm -rf "$CRITIC_INIT"
    python -m nla.scripts.prepare_critic_checkpoint \
        --base-model Qwen/Qwen3-8B \
        --num-layers 25 \
        --dataset-sidecar "$PROJ/data/ar_sft_train_8b.parquet" \
        --output "$CRITIC_INIT"
    echo "Critic init saved to $CRITIC_INIT"
else
    echo "=== Step 1: Critic checkpoint exists at $CRITIC_INIT ==="
fi

# --- Step 2: Verify imports ---
echo "=== Step 2: Verifying imports ==="
export PYTHONPATH="$ORIG_REPO:$MILES_REPO${PYTHONPATH:+:$PYTHONPATH}"
python -c "
from miles.utils.arguments import parse_args
from nla.train_actor import NLAFSDPActor
from nla.loss import nla_critic_loss
from nla.rollout.sft_critic import generate_rollout
from nla.data_source import NLADataSource
from nla.config import load_nla_config, resolve_sidecar_source, load_nla_config_from_args
from nla.models import NLACriticModel
print('All imports OK')
"

# --- Step 3: Start Ray ---
echo "=== Step 3: Starting Ray cluster (4 GPUs) ==="
# Kill any stale Ray instance
ray stop 2>/dev/null || true
sleep 2

# Stage data in $PROJ/tmp (no /tmp on Berzelius)
LOCAL_DIR="$PROJ/tmp/nla_data_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_DIR"
trap 'rm -rf "$LOCAL_DIR" "$WORKDIR"' EXIT
cp "$PROJ/data/ar_sft_train_8b.parquet" "$LOCAL_DIR/data.parquet"
cp "$PROJ/data/ar_sft_train_8b.parquet.nla_meta.yaml" "$LOCAL_DIR/data.parquet.nla_meta.yaml"
echo "Data copied to $LOCAL_DIR"

# Start Ray head node with explicit GPU count matching SLURM allocation
# CUDA_VISIBLE_DEVICES is set by SLURM based on --gres=gpu:4
ray start --head --num-gpus=4 --num-cpus=32 --port=6379
echo "Ray cluster started"

# --- Step 4: Run miles train.py (critic SFT) ---
echo "=== Step 4: Running critic SFT via miles train.py ==="

SAVE_DIR="$PROJ/checkpoints/orig_critic_sft_8b"
rm -rf "$SAVE_DIR"

python "$MILES_REPO/train.py" \
    --train-backend fsdp \
    --custom-actor-cls-path nla.train_actor.NLAFSDPActor \
    --nla-model-is-critic \
    --loss-type custom_loss \
    --custom-loss-function-path nla.loss.nla_critic_loss \
    --debug-train-only \
    --disable-compute-advantages-and-returns \
    --rollout-function-path nla.rollout.sft_critic.generate_rollout \
    --data-source-path nla.data_source.NLADataSource \
    --prompt-data "$LOCAL_DIR/data.parquet" \
    --input-key prompt \
    --hf-checkpoint "$CRITIC_INIT" \
    --save "$SAVE_DIR" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 4 \
    --rollout-batch-size 256 \
    --global-batch-size 256 \
    --micro-batch-size 8 \
    --lr 2e-5 --min-lr 2e-6 --lr-warmup-iters 50 --lr-decay-style cosine \
    --n-samples-per-prompt 1 \
    --num-epoch 1 \
    --save-interval 500

echo "=== Original Critic SFT done ==="
echo "Output: $SAVE_DIR"

# --- Cleanup ---
ray stop
rm -rf "$LOCAL_DIR" "$WORKDIR"
