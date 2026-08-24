#!/bin/bash
#SBATCH --job-name=nla-rebuild-8b
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/rebuild_8b_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/rebuild_8b_%j.err

# Rebuild 8B training data with correct document-level split.
# Source: nla_250k_explained.parquet (labels) + av_sft_vectors_8b.parquet (8B vectors)
# Split: 25% AV, 25% AR, 50% RL (document-level, no overlap)

set -euo pipefail

CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export HF_HOME=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/huggingface
mkdir -p "$HF_HOME"

PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
DATA="$PROJ/data"
TOKENIZER="Qwen/Qwen3-8B"

# Ensure our nla takes precedence over the original repo's nla
export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"

WORK="$DATA/rebuild_8b"
mkdir -p "$WORK"

EXPLAINED="$DATA/nla_250k_explained.parquet"
VECTORS="$DATA/av_sft_vectors_8b.parquet"

echo "=== Rebuilding 8B training data (document-level split) ==="
echo "Labels: $EXPLAINED"
echo "Vectors: $VECTORS"
echo "Work dir: $WORK"

# Step 1: Split labels by document
echo ""
echo "=== Step 1: Split labels by document ==="
python -m nla.datagen.split_positions \
    --input "$EXPLAINED" \
    --output-dir "$WORK" \
    --av-sft-frac 0.25 --ar-sft-frac 0.25 --rl-frac 0.50 \
    --seed 42

# Step 2: Split vectors using the same document sets
echo ""
echo "=== Step 2: Split vectors by document ==="
for SPLIT in av_sft ar_sft rl; do
    EXPL="$WORK/${SPLIT}_explained.parquet"
    if [ ! -f "$EXPL" ]; then
        echo "  $EXPL not found — skipping"
        continue
    fi
    echo "  Splitting vectors for $SPLIT..."
    python -c "
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc

# Get doc_ids from this split
expl = pq.read_table('$EXPL')
split_docs = set(expl.column('doc_id').to_pylist())
print(f'  $SPLIT: {len(split_docs)} docs, {expl.num_rows} rows')

# Filter vectors to these docs
vecs = pq.read_table('$VECTORS')
mask = pc.is_in(vecs.column('doc_id'),
                value_set=pa.array(sorted(split_docs), type=pa.string()))
subset = vecs.filter(mask)
pq.write_table(subset, '$WORK/${SPLIT}_vectors.parquet')
print(f'  vectors: {subset.num_rows} rows → $WORK/${SPLIT}_vectors.parquet')
"
done

# Step 3: Build training parquets
echo ""
echo "=== Step 3: Build training parquets ==="
for SPLIT in av_sft ar_sft rl; do
    EXPL="$WORK/${SPLIT}_explained.parquet"
    VECS="$WORK/${SPLIT}_vectors.parquet"
    OUT="$DATA/${SPLIT}_train_8b.parquet"

    if [ ! -f "$EXPL" ] || [ ! -f "$VECS" ]; then
        echo "  $SPLIT: missing files — skipping"
        continue
    fi

    echo "  Building $SPLIT..."
    python -m nla.training.build_training_data \
        --explained "$EXPL" \
        --vectors "$VECS" \
        --tokenizer "$TOKENIZER" \
        --output "$OUT" \
        --split-type "$SPLIT"

    # Rename sidecar to match
    if [ -f "$OUT.nla_meta.yaml" ]; then
        echo "  $SPLIT: $(pq.ParquetFile('$OUT').metadata.num_rows) rows → $OUT"
    fi
done

# Step 4: Verify no document overlap
echo ""
echo "=== Step 4: Verify no document overlap ==="
python -c "
import pyarrow.parquet as pq

docs = {}
for split in ('av_sft', 'ar_sft', 'rl'):
    path = '$DATA/${split}_train_8b.parquet'
    try:
        pf = pq.ParquetFile(path)
        batch = next(pf.iter_batches(batch_size=1000000))
        docs[split] = set(batch.column('doc_id').to_pylist())
        print(f'{split}: {pf.metadata.num_rows} rows, {len(docs[split])} unique docs')
    except Exception as e:
        print(f'{split}: SKIPPED ({e})')

for a, b in [('av_sft', 'ar_sft'), ('av_sft', 'rl'), ('ar_sft', 'rl')]:
    if a in docs and b in docs:
        overlap = docs[a] & docs[b]
        print(f'{a} ∩ {b}: {len(overlap)} overlapping docs')
"

echo ""
echo "=== Done ==="
ls -lh "$DATA"/*_train_8b.parquet 2>/dev/null
