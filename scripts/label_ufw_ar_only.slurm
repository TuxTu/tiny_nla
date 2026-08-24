#!/bin/bash
#SBATCH --job-name=nla-label-ar
#SBATCH --partition=berzelius-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/label_ar_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/label_ar_%j.err

# Re-run AR labeling only — resumes from 914/50000 docs already labeled.
# AV and RL are already done. This only calls api_explain on the AR bucket.

set -euo pipefail

CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export HF_HOME=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/huggingface
mkdir -p "$HF_HOME"

PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"

# Extract DeepSeek key from Claude settings
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    DEEPSEEK_API_KEY=$(python3 -c "import json; print(json.load(open('$HOME/.claude/settings.json'))['env'].get('ANTHROPIC_API_KEY',''))" 2>/dev/null || true)
    export DEEPSEEK_API_KEY
fi

POOL="$PROJ/data/pool_ufw_8b"
AR_POS="$POOL/ar/positions.parquet"
AR_EXPL="$POOL/ar/explained.parquet"

echo "=== AR labeling (resume from 914 docs) ==="
echo "Positions: $(python3 -c "import pyarrow.parquet as pq; print(pq.ParquetFile('$AR_POS').metadata.num_rows)" 2>/dev/null) rows"
echo "Already labeled: $(python3 -c "import pyarrow.parquet as pq; print(pq.ParquetFile('$AR_EXPL').metadata.num_rows)" 2>/dev/null) rows"

python -m nla.datagen.api_explain \
    --input "$AR_POS" \
    --output "$AR_EXPL" \
    --provider deepseek \
    --provider-model deepseek-v4-flash \
    --provider-concurrency 64 \
    --provider-max-tokens 400 \
    --chunk-size 512 \
    --resume \
    --cache-from "$AR_EXPL"

echo ""
echo "=== AR labeling done ==="
echo "Rows now: $(python3 -c "import pyarrow.parquet as pq; print(pq.ParquetFile('$AR_EXPL').metadata.num_rows)" 2>/dev/null)"

# Rebuild output AR parquet with full labeling
echo "=== Rebuilding ar_sft_explained.parquet ==="
python3 << 'PYEOF'
import pyarrow.parquet as pq, pyarrow as pa, pyarrow.compute as pc

pool_ar = '/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/data/pool_ufw_8b/ar/explained.parquet'
out_ar = '/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/data/ufw_8b/ar_sft_explained.parquet'

# Load all AR positions from pool with explanations
table = pq.read_table(pool_ar)
pq.write_table(table, out_ar)
print(f'Wrote {table.num_rows} rows → {out_ar}')
PYEOF

echo ""
echo "=== Submitting Phase 2 vector extraction for AR ==="
sbatch "$PROJ/scripts/extract_ufw_ar_vectors.slurm"

echo "Done."
