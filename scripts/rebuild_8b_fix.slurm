#!/bin/bash
#SBATCH --job-name=nla-rebuild-fix
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/rebuild_fix_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/rebuild_fix_%j.err

# Fix: Build AR and RL training data from already-split explained+vectors.
# Splits were already done by job 16913710 — just need to build training parquets.

set -euo pipefail

CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export HF_HOME=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/huggingface
mkdir -p "$HF_HOME"

PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
DATA="$PROJ/data"
TOKENIZER="Qwen/Qwen3-8B"

export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"

WORK="$DATA/rebuild_8b"

echo "=== Building training parquets from split data ==="

for SPLIT in ar_sft rl; do
    EXPL="$WORK/${SPLIT}_explained.parquet"
    VECS="$WORK/${SPLIT}_vectors.parquet"
    OUT="$DATA/${SPLIT}_train_8b.parquet"

    if [ ! -f "$EXPL" ] || [ ! -f "$VECS" ]; then
        echo "  $SPLIT: explained or vectors file missing, skipping"
        continue
    fi

    echo "Building $SPLIT ..."
    python -m nla.training.build_training_data \
        --explained "$EXPL" \
        --vectors "$VECS" \
        --tokenizer "$TOKENIZER" \
        --output "$OUT" \
        --split-type "$SPLIT"

    echo "  $SPLIT done."
done

echo ""
echo "=== Verify ==="
python3 << 'PYEOF'
import pyarrow.parquet as pq
import os

data_dir = '/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/data'
for split in ('ar_sft', 'rl'):
    path = os.path.join(data_dir, f'{split}_train_8b.parquet')
    if os.path.exists(path):
        pf = pq.ParquetFile(path)
        batch = next(pf.iter_batches(batch_size=100000))
        docs = set(batch.column('doc_id').to_pylist())
        n_tokens = batch.column('n_raw_tokens').to_pylist()
        from collections import Counter
        counts = Counter(batch.column('doc_id').to_pylist())
        avg = sum(counts.values()) / len(counts)
        print(f'{split}: {pf.metadata.num_rows} rows, {len(docs)} docs, {avg:.1f} rows/doc, max={max(counts.values())}')

# Verify doc-level split (no overlap with AV)
av_path = os.path.join(data_dir, 'av_sft_train_8b.parquet')
if os.path.exists(av_path):
    av = pq.ParquetFile(av_path)
    av_batch = next(av.iter_batches(batch_size=100000))
    av_docs = set(av_batch.column('doc_id').to_pylist())

    for split in ('ar_sft', 'rl'):
        path = os.path.join(data_dir, f'{split}_train_8b.parquet')
        if os.path.exists(path):
            pf = pq.ParquetFile(path)
            batch = next(pf.iter_batches(batch_size=100000))
            split_docs = set(batch.column('doc_id').to_pylist())
            overlap = av_docs & split_docs
            print(f'av_sft ∩ {split}: {len(overlap)} overlapping docs (should be 0)')
PYEOF

echo "=== Done ==="
