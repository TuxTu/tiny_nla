#!/bin/bash
#SBATCH --job-name=nla-eval-ufw
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/eval_ufw_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/eval_ufw_%j.err

# Held-out FVE eval: trained UFW critic on the 9042 true-AR UFW rows (zero doc
# overlap with training, same UltraFineWeb + DeepSeek distribution).

set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export HF_HOME=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/huggingface
export TORCHINDUCTOR_CACHE_DIR=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/torchinductor   # else torch defaults to /tmp/torchinductor_$USER
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"

UFW="$PROJ/data/ufw_8b"
EVAL="$PROJ/data/ar_eval_8b_ufw.parquet"

echo "=== Build held-out AR eval parquet (true-AR UFW, 9042 rows) ==="
if [ ! -f "$EVAL" ]; then
    # Filter explained to match vector keys, then build AR-format parquet.
    python3 << PYEOF
import pyarrow.parquet as pq, pyarrow as pa
expl = pq.read_table("$UFW/ar_sft_explained.parquet")
vecs = pq.read_table("$UFW/ar_sft_explained_vectors.parquet")
vk = set(zip(vecs.column('doc_id').to_pylist(), vecs.column('n_raw_tokens').to_pylist()))
ek = list(zip(expl.column('doc_id').to_pylist(), expl.column('n_raw_tokens').to_pylist()))
mask = pa.array([k in vk for k in ek], type=pa.bool_())
pq.write_table(expl.filter(mask), "$UFW/ar_sft_explained_filt.parquet")
print("filtered:", expl.num_rows, "->", int(sum(mask.to_pylist())))
PYEOF
    python -m nla.training.build_training_data \
        --explained "$UFW/ar_sft_explained_filt.parquet" \
        --vectors "$UFW/ar_sft_explained_vectors.parquet" \
        --tokenizer Qwen/Qwen3-8B \
        --output "$EVAL" \
        --split-type ar_sft
fi

echo ""
echo "=== Run FVE eval on trained UFW critic ==="
python "$PROJ/scripts/nla_eval_fve.py" \
    --critic-only \
    --critic-ckpt "$PROJ/checkpoints/critic_sft_8b_ufw" \
    --data "$EVAL" \
    --sidecar "$EVAL.nla_meta.yaml" \
    --n-samples 2000

echo "=== Done ==="
