#!/bin/bash
#SBATCH --job-name=nla-prep
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-40GB:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/prep_%A_%a.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/prep_%A_%a.err
#SBATCH --array=0-2

# ============================================================================
# Data preparation: extract vectors from Qwen3-4B, split by doc_id, build training parquets.
#
# Array job: 3 tasks × 4 GPUs each = 12 GPUs total for extraction.
#   Task 0: docs[0:N/3]
#   Task 1: docs[N/3:2N/3]
#   Task 2: docs[2N/3:]
#
# After all array tasks complete, task 0 merges shards and builds training data.
# ============================================================================

set -euo pipefail

module load Miniforge3/25.3.1-0
source /software/sse/generic/manual/ssetools/conda/hpc_conda_wrap2.sh
conda activate tiny_nla

export HF_HOME=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/huggingface
export HF_HUB_CACHE=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/huggingface/hub
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
DATA=$PROJ/data
LOGS=$PROJ/logs
mkdir -p "$DATA" "$LOGS"

DATASET="TuHan/qwen3-nla-250k"
MODEL="Qwen/Qwen3-4B"
EXPLAINED_PARQUET="$DATA/nla_250k_explained.parquet"
VECTORS_PARQUET="$DATA/nla_250k_vectors.parquet"

SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}
NUM_SHARDS=3

echo "============================================"
echo "Task $SHARD_ID / $NUM_SHARDS"
echo "============================================"

# ---- Step 1: Download explained dataset (all tasks, but only once) ----------
if [ ! -f "$EXPLAINED_PARQUET" ]; then
    echo "Downloading $DATASET ..."
    python -c "
from datasets import load_dataset
ds = load_dataset('$DATASET', split='train')
ds.to_parquet('$EXPLAINED_PARQUET')
print(f'Downloaded {len(ds)} rows → $EXPLAINED_PARQUET')
"
else
    echo "$EXPLAINED_PARQUET already exists — skipping download"
fi

# ---- Step 2: Extract vectors (each task handles its shard) ------------------
SHARD_OUT="$VECTORS_PARQUET.shard_${SHARD_ID}"
if [ -f "$SHARD_OUT" ]; then
    echo "$SHARD_OUT already exists — skipping extraction"
else
    echo "Extracting vectors with Qwen3-4B (shard $SHARD_ID/$NUM_SHARDS, 4 GPUs)..."
    python -m nla.datagen.extract_vectors \
        --input "$EXPLAINED_PARQUET" \
        --model-name "$MODEL" \
        --output "$VECTORS_PARQUET" \
        --batch-size 16 \
        --max-length 2048 \
        --gpus 4 \
        --shard-id "$SHARD_ID" \
        --num-shards "$NUM_SHARDS"
    echo "Extraction done → $SHARD_OUT"
fi

# ---- Step 3: Merge shards + split + build (only task 0) ---------------------
if [ "$SHARD_ID" -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "Waiting for all shards, then merging + building"
    echo "============================================"

    # Wait for all shard files
    for i in $(seq 0 $((NUM_SHARDS - 1))); do
        SHARD_FILE="$VECTORS_PARQUET.shard_$i"
        while [ ! -f "$SHARD_FILE" ]; do
            echo "Waiting for $SHARD_FILE ..."
            sleep 30
        done
    done

    # Merge shards
    MERGED="$VECTORS_PARQUET"
    if [ -f "$MERGED" ]; then
        echo "$MERGED already exists — skipping merge"
    else
        echo "Merging shards..."
        python -c "
import pyarrow.parquet as pq
import pyarrow as pa
tables = []
for i in range($NUM_SHARDS):
    t = pq.read_table('${VECTORS_PARQUET}.shard_' + str(i))
    tables.append(t)
merged = pa.concat_tables(tables)
pq.write_table(merged, '$MERGED')
print(f'Merged {merged.num_rows} rows → $MERGED')
"
    fi

    # ---- Split by doc_id ----------------------------------------------------
    echo ""
    echo "Splitting by doc_id into AV/AR/RL..."
    python -m nla.datagen.split_positions \
        --input "$EXPLAINED_PARQUET" \
        --output-dir "$DATA/splits" \
        --av-sft-frac 0.25 --ar-sft-frac 0.25 --rl-frac 0.50 \
        --seed 42

    # Split vectors the same way (use same doc_id sets)
    echo "Splitting vectors by doc_id..."
    for SPLIT in av_sft ar_sft rl; do
        echo "  Splitting vectors for $SPLIT..."
        python -c "
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc

# Read the explained split to get doc_ids
expl = pq.read_table('$DATA/splits/${SPLIT}_explained.parquet')
split_docs = set(expl.column('doc_id').to_pylist())

# Read vectors
vecs = pq.read_table('$MERGED')
mask = pc.is_in(vecs.column('doc_id'), value_set=pa.array(sorted(split_docs), type=pa.string()))
subset = vecs.filter(mask)
pq.write_table(subset, '$DATA/splits/${SPLIT}_vectors.parquet')
print(f'  {SPLIT}: {subset.num_rows} rows')
"
    done

    # ---- Build training data for 50% of docs (half data) --------------------
    echo ""
    echo "Building training parquets (50% subsample)..."
    TOKENIZER="$MODEL"

    # Compute half-doc counts per split
    for SPLIT in av_sft ar_sft rl; do
        EXPL="$DATA/splits/${SPLIT}_explained.parquet"
        VECS="$DATA/splits/${SPLIT}_vectors.parquet"
        if [ "$SPLIT" = "rl" ]; then
            OUT="$DATA/${SPLIT}_train.parquet"
        else
            OUT="$DATA/${SPLIT}_train.parquet"
        fi

        # Count total docs
        TOTAL_DOCS=$(python -c "
import pyarrow.parquet as pq
t = pq.read_table('$EXPL')
print(len(set(t.column('doc_id').to_pylist())))
")
        # Use half
        HALF_DOCS=$((TOTAL_DOCS / 2))
        echo "  $SPLIT: $TOTAL_DOCS docs → subsample to $HALF_DOCS docs"

        # Build full first, then subsample
        python -m nla.training.build_training_data \
            --explained "$EXPL" \
            --vectors "$VECS" \
            --tokenizer "$TOKENIZER" \
            --output "$OUT" \
            --split-type "$SPLIT"

        # Subsample to half docs
        python -c "
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
import random

t = pq.read_table('$OUT')
docs = sorted(set(t.column('doc_id').to_pylist()))
rng = random.Random(42)
rng.shuffle(docs)
selected = set(docs[:$HALF_DOCS])
mask = pc.is_in(t.column('doc_id'), value_set=pa.array(sorted(selected), type=pa.string()))
subset = t.filter(mask)
pq.write_table(subset, '$OUT')
print(f'  $SPLIT: {len(selected)} docs, {subset.num_rows} rows → $OUT')
"
    done

    echo ""
    echo "=== Data preparation complete ==="
    echo "Training parquets:"
    ls -lh "$DATA"/*_train.parquet
fi

echo "Task $SHARD_ID done."
