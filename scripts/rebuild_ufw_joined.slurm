#!/bin/bash
#SBATCH --job-name=nla-rebuild-join
#SBATCH --partition=berzelius-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/rebuild_join_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/rebuild_join_%j.err

# Rebuild AR train + held-out eval parquets with a REAL join on
# (doc_id, n_raw_tokens). The previous build paired columns positionally while
# extract_vectors.py emits sorted(doc_id) order -> 99.96% of rows had an
# explanation attached to the wrong activation vector.
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
export HF_HOME=$PROJ/cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJ"
UFW="$PROJ/data/ufw_8b"

echo "=== AR train (249k) ==="
python -m nla.training.build_training_data \
    --explained "$UFW/av_sft_explained_filtered.parquet" \
    --vectors   "$UFW/av_sft_explained_vectors.parquet" \
    --tokenizer Qwen/Qwen3-8B --split-type ar_sft \
    --output "$PROJ/data/ar_sft_train_8b_ufw_joined.parquet"

echo "=== AR held-out eval (9042) ==="
python -m nla.training.build_training_data \
    --explained "$UFW/ar_sft_explained_filt.parquet" \
    --vectors   "$UFW/ar_sft_explained_vectors.parquet" \
    --tokenizer Qwen/Qwen3-8B --split-type ar_sft \
    --output "$PROJ/data/ar_eval_8b_ufw_joined.parquet"

echo "=== VERIFY: every row's vector must equal the vectors-parquet entry for its key ==="
python - <<'PY'
import pyarrow.parquet as pq, numpy as np
def check(built, vecs_path, n_probe=3000):
    b = pq.read_table(built)
    v = pq.read_table(vecs_path)
    vk = list(zip(v.column('doc_id').to_pylist(), v.column('n_raw_tokens').to_pylist()))
    vpos = {k: i for i, k in enumerate(vk)}
    d = v.column('activation_vector').combine_chunks().type.list_size
    V = v.column('activation_vector').combine_chunks().values.to_numpy().reshape(-1, d)
    B = b.column('activation_vector').combine_chunks().values.to_numpy().reshape(-1, d)
    bk = list(zip(b.column('doc_id').to_pylist(), b.column('n_raw_tokens').to_pylist()))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(bk), size=min(n_probe, len(bk)), replace=False)
    bad = sum(not np.array_equal(B[i], V[vpos[bk[i]]]) for i in idx)
    print(f"  {built}: {b.num_rows} rows, probed {len(idx)}, MISMATCHES={bad}")
    assert bad == 0, "join still wrong"
check("data/ar_sft_train_8b_ufw_joined.parquet", "data/ufw_8b/av_sft_explained_vectors.parquet")
check("data/ar_eval_8b_ufw_joined.parquet",      "data/ufw_8b/ar_sft_explained_vectors.parquet")
print("JOIN VERIFIED")
PY
echo "=== REBUILD DONE ==="
