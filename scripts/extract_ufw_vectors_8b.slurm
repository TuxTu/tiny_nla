#!/bin/bash
#SBATCH --job-name=nla-ufw-vectors
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=08:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/ufw_vectors_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/ufw_vectors_%j.err

# Phase 2 (GPU): Extract Qwen3-8B activation vectors + build training data.
# Depends on Phase 1 labeling completing.

set -euo pipefail

CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export HF_HOME=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/huggingface
export TORCHINDUCTOR_CACHE_DIR=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/torchinductor   # else torch defaults to /tmp/torchinductor_$USER
mkdir -p "$HF_HOME"

PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="$PROJ/configs/datagen_8b_ultrafineweb_100k.yaml"

# Wait for labeling to finish (with timeout)
echo "=== Waiting for Phase 1 labeling to complete ==="
MAX_WAIT=43200  # 12h timeout
WAITED=0
while [ ! -f "$PROJ/data/ufw_8b/_labeling_done" ]; do
    sleep 60
    WAITED=$((WAITED + 60))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: timed out waiting for labeling after ${MAX_WAIT}s"
        exit 1
    fi
    echo "  ... waited ${WAITED}s"
done
echo "Labeling signal file found. Starting vector extraction."

echo ""
echo "=== Phase 2: Qwen3-8B vector extraction ==="
echo "Config: $CONFIG"
echo ""

python -m nla.datagen.run_pipeline \
    --config "$CONFIG" \
    --stages vectors

echo ""
echo "=== Vector extraction complete ==="
echo "Vectors in: $PROJ/data/ufw_8b/"
ls -lh "$PROJ/data/ufw_8b/"*vectors* 2>/dev/null || echo "(checking)"

echo ""
echo "=== Building training parquets ==="
TOKENIZER="Qwen/Qwen3-8B"
DATA_DIR="$PROJ/data/ufw_8b"
OUT_DIR="$PROJ/data"

for SPLIT in av_sft ar_sft rl; do
    EXPL="$DATA_DIR/${SPLIT}_explained.parquet"
    if [ "$SPLIT" = "rl" ]; then
        EXPL="$DATA_DIR/rl.parquet"
    fi
    VECS="$DATA_DIR/${SPLIT}_explained_vectors.parquet"
    if [ "$SPLIT" = "rl" ]; then
        VECS="$DATA_DIR/rl_vectors.parquet"
    fi
    OUT="$OUT_DIR/${SPLIT}_train_8b_ufw.parquet"

    if [ ! -f "$EXPL" ] || [ ! -f "$VECS" ]; then
        echo "  $SPLIT: missing $EXPL or $VECS, skipping"
        continue
    fi

    echo "  Building $SPLIT ($EXPL + $VECS → $OUT)..."
    python -m nla.training.build_training_data \
        --explained "$EXPL" \
        --vectors "$VECS" \
        --tokenizer "$TOKENIZER" \
        --output "$OUT" \
        --split-type "$SPLIT"
done

echo ""
echo "=== All done ==="
echo "Training parquets:"
ls -lh "$OUT_DIR"/*_train_8b_ufw.parquet 2>/dev/null

# Verify zero doc overlap
python3 << 'PYEOF'
import pyarrow.parquet as pq, os

data_dir = '/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/data'
docs = {}
for split in ('av_sft', 'ar_sft', 'rl'):
    path = os.path.join(data_dir, f'{split}_train_8b_ufw.parquet')
    if os.path.exists(path):
        pf = pq.ParquetFile(path)
        batch = next(pf.iter_batches(batch_size=1000000))
        docs[split] = set(batch.column('doc_id').to_pylist())
        print(f'{split}: {pf.metadata.num_rows} rows, {len(docs[split])} unique docs')

for a, b in [('av_sft', 'ar_sft'), ('av_sft', 'rl'), ('ar_sft', 'rl')]:
    if a in docs and b in docs:
        overlap = docs[a] & docs[b]
        print(f'{a} ∩ {b}: {len(overlap)} overlapping docs (should be 0)')
PYEOF
