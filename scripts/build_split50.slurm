#!/bin/bash
#SBATCH --job-name=nla-split50
#SBATCH --partition=berzelius-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=03:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/split50_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/split50_%j.err
# Doc-disjoint 50/50 split of the labelled AV pool:
#   half A (12,500 docs) -> AR-SFT (critic)
#   half B (12,500 docs) -> AV-SFT (actor)
# eval stays the AR pool's 914 docs; RL stays the RL pool's 24,981 docs.
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
export HF_HOME=$PROJ/cache/huggingface
export TORCHINDUCTOR_CACHE_DIR=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/torchinductor   # else torch defaults to /tmp/torchinductor_$USER
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJ"
UFW="$PROJ/data/ufw_8b"
S="$PROJ/data/split50"

echo "=== 1. doc-level split of the labelled AV pool ==="
python -m nla.datagen.split_labeled_halves \
    --explained "$UFW/av_sft_explained_filtered.parquet" \
    --vectors   "$UFW/av_sft_explained_vectors.parquet" \
    --out-prefix-a "$S/halfA" \
    --out-prefix-b "$S/halfB" \
    --frac-a 0.5 --seed 42

echo ""
echo "=== 2. half A -> AR-SFT (critic training data) ==="
python -m nla.training.build_training_data \
    --explained "$S/halfA_explained.parquet" \
    --vectors   "$S/halfA_vectors.parquet" \
    --tokenizer Qwen/Qwen3-8B --split-type ar_sft \
    --output "$S/ar_sft_train.parquet"

echo ""
echo "=== 3. half B -> AV-SFT (actor training data) ==="
python -m nla.training.build_training_data \
    --explained "$S/halfB_explained.parquet" \
    --vectors   "$S/halfB_vectors.parquet" \
    --tokenizer Qwen/Qwen3-8B --split-type av_sft \
    --output "$S/av_sft_train.parquet"

echo ""
echo "=== 4. VERIFY ==="
python - <<'PY'
import pyarrow.parquet as pq, numpy as np

def docs(f):
    return set(pq.read_table(f, columns=["doc_id"]).column("doc_id").to_pylist())

ar = docs("data/split50/ar_sft_train.parquet")
av = docs("data/split50/av_sft_train.parquet")
ev = docs("data/ar_eval_8b_ufw_joined.parquet")
rl = docs("data/rl_train_8b_ufw_joined.parquet")

print(f"docs -> AR(critic)={len(ar):,}  AV(actor)={len(av):,}  eval={len(ev):,}  RL={len(rl):,}")
print("--- DOC-LEVEL OVERLAP (all must be 0) ---")
for a, b, na, nb in ((ar, av, "AR", "AV"), (ar, ev, "AR", "eval"), (av, ev, "AV", "eval"),
                     (ar, rl, "AR", "RL"), (av, rl, "AV", "RL"), (ev, rl, "eval", "RL")):
    n = len(a & b)
    print(f"  {na} ∩ {nb} = {n}")
    assert n == 0, f"{na}/{nb} share {n} documents"

for built, src in (("data/split50/ar_sft_train.parquet", "data/split50/halfA_vectors.parquet"),
                   ("data/split50/av_sft_train.parquet", "data/split50/halfB_vectors.parquet")):
    b = pq.read_table(built); v = pq.read_table(src)
    vk = list(zip(v.column('doc_id').to_pylist(), v.column('n_raw_tokens').to_pylist()))
    vpos = {k: i for i, k in enumerate(vk)}
    d = v.column('activation_vector').combine_chunks().type.list_size
    V = v.column('activation_vector').combine_chunks().values.to_numpy().reshape(-1, d)
    B = b.column('activation_vector').combine_chunks().values.to_numpy().reshape(-1, d)
    bk = list(zip(b.column('doc_id').to_pylist(), b.column('n_raw_tokens').to_pylist()))
    idx = np.random.default_rng(0).choice(len(bk), size=min(4000, len(bk)), replace=False)
    bad = sum(not np.array_equal(B[i], V[vpos[bk[i]]]) for i in idx)
    print(f"  {built}: {b.num_rows:,} rows, probed {len(idx)}, MISMATCHES={bad}")
    assert bad == 0

print("ALL CHECKS PASSED")
PY
echo "=== SPLIT50 BUILD DONE ==="
