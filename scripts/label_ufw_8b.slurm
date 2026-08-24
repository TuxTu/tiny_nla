#!/bin/bash
#SBATCH --job-name=nla-label-ufw
#SBATCH --partition=berzelius-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/label_ufw_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/label_ufw_%j.err

# Phase 1 (CPU): Ultra-FineWeb → positions → DeepSeek labels → split output.
# No GPU needed — frees GPU allocation for other jobs.

set -euo pipefail

CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export HF_HOME=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/huggingface
mkdir -p "$HF_HOME"

PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"

# Extract DeepSeek API key from Claude settings (stored as ANTHROPIC_API_KEY)
# The key is a DeepSeek key — base URL routes to https://api.deepseek.com/anthropic
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    DEEPSEEK_API_KEY=$(python3 -c "import json; print(json.load(open('$HOME/.claude/settings.json'))['env'].get('ANTHROPIC_API_KEY',''))" 2>/dev/null || true)
    if [ -z "$DEEPSEEK_API_KEY" ]; then
        echo "ERROR: DEEPSEEK_API_KEY not set and not found in ~/.claude/settings.json"
        exit 1
    fi
    export DEEPSEEK_API_KEY
fi

CONFIG="$PROJ/configs/datagen_8b_ultrafineweb_100k.yaml"
echo "=== Phase 1: Ultra-FineWeb labeling pipeline ==="
echo "Config: $CONFIG"
echo "Stages: extract → explain → output"
echo ""

python -m nla.datagen.run_pipeline \
    --config "$CONFIG" \
    --stages extract,explain,output

echo ""
echo "=== Phase 1 complete ==="
echo "Output: $PROJ/data/ufw_8b/"
ls -lh "$PROJ/data/ufw_8b/" 2>/dev/null || echo "(output dir check)"

# Write a signal file so Phase 2 knows it can start
echo "READY" > "$PROJ/data/ufw_8b/_labeling_done"

echo ""
echo "=== Submit Phase 2 (GPU vector extraction) ==="
sbatch "$PROJ/scripts/extract_ufw_vectors_8b.slurm"
