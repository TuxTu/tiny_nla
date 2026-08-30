#!/bin/bash
#SBATCH --job-name=nla-build-avrl
#SBATCH --partition=berzelius-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/build_avrl_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/build_avrl_%j.err

# Build AV-SFT (train + held-out eval) and RL parquets from the UFW pool with the
# FIXED (doc_id, n_raw_tokens) join. The previous av_sft_*/rl_* files were built by
# the positional-concat code path -> the actor was learning to describe vectors it
# was never shown.
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

echo "=== AV-SFT train (249,449) — same pool the critic trained on ==="
python -m nla.training.build_training_data \
    --explained "$UFW/av_sft_explained_filtered.parquet" \
    --vectors   "$UFW/av_sft_explained_vectors.parquet" \
    --tokenizer Qwen/Qwen3-8B --split-type av_sft \
    --output "$PROJ/data/av_sft_train_8b_ufw_joined.parquet"

echo "=== AV-SFT held-out eval (9,042) — the same rows as the AR held-out set ==="
python -m nla.training.build_training_data \
    --explained "$UFW/ar_sft_explained_filt.parquet" \
    --vectors   "$UFW/ar_sft_explained_vectors.parquet" \
    --tokenizer Qwen/Qwen3-8B --split-type av_sft \
    --output "$PROJ/data/av_eval_8b_ufw_joined.parquet"

echo "=== RL prompts (249,751) ==="
python -m nla.training.build_training_data \
    --explained "$UFW/rl.parquet" \
    --vectors   "$UFW/rl_vectors.parquet" \
    --tokenizer Qwen/Qwen3-8B --split-type rl \
    --output "$PROJ/data/rl_train_8b_ufw_joined.parquet"

echo "=== VERIFY: each built row's vector must equal the vectors-parquet entry for its key ==="
python - <<'PY'
import pyarrow.parquet as pq, numpy as np
def check(built, vecs_path, n_probe=4000):
    b = pq.read_table(built)
    v = pq.read_table(vecs_path)
    vk = list(zip(v.column('doc_id').to_pylist(), v.column('n_raw_tokens').to_pylist()))
    vpos = {k: i for i, k in enumerate(vk)}
    d = v.column('activation_vector').combine_chunks().type.list_size
    V = v.column('activation_vector').combine_chunks().values.to_numpy().reshape(-1, d)
    B = b.column('activation_vector').combine_chunks().values.to_numpy().reshape(-1, d)
    bk = list(zip(b.column('doc_id').to_pylist(), b.column('n_raw_tokens').to_pylist()))
    idx = np.random.default_rng(0).choice(len(bk), size=min(n_probe, len(bk)), replace=False)
    bad = sum(not np.array_equal(B[i], V[vpos[bk[i]]]) for i in idx)
    print(f"  {built}: {b.num_rows} rows, probed {len(idx)}, MISMATCHES={bad}")
    assert bad == 0, "join still wrong"
check("data/av_sft_train_8b_ufw_joined.parquet", "data/ufw_8b/av_sft_explained_vectors.parquet")
check("data/av_eval_8b_ufw_joined.parquet",      "data/ufw_8b/ar_sft_explained_vectors.parquet")
check("data/rl_train_8b_ufw_joined.parquet",     "data/ufw_8b/rl_vectors.parquet")
print("JOIN VERIFIED")

# Cross-check: the AV train set must carry the SAME (key -> vector) pairs the
# critic was trained on, otherwise actor and critic disagree about the space.
a = pq.read_table("data/av_sft_train_8b_ufw_joined.parquet")
c = pq.read_table("data/ar_sft_train_8b_ufw_joined.parquet")
ak = list(zip(a.column('doc_id').to_pylist(), a.column('n_raw_tokens').to_pylist()))
ck = list(zip(c.column('doc_id').to_pylist(), c.column('n_raw_tokens').to_pylist()))
print(f"AV/AR train key sets identical: {set(ak) == set(ck)}  ({len(ak)} vs {len(ck)} rows)")
assert set(ak) == set(ck)

# Held-out sets must not leak into training.
e = pq.read_table("data/av_eval_8b_ufw_joined.parquet")
ek = set(zip(e.column('doc_id').to_pylist(), e.column('n_raw_tokens').to_pylist()))
print(f"held-out rows leaking into AV train: {len(ek & set(ak))} / {len(ek)}")
assert not (ek & set(ak))
print("ALL CHECKS PASSED")
PY

echo "=== TOKEN LENGTHS (pick --max-length) ==="
python - <<'PY'
import pyarrow.parquet as pq, numpy as np, os
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
t = pq.read_table("data/av_sft_train_8b_ufw_joined.parquet",
                  columns=["prompt", "response"]).slice(0, 5000)
prompts = t.column("prompt").to_pylist()
resps = t.column("response").to_pylist()
lens = []
for msgs, r in zip(prompts, resps):
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + r
    lens.append(len(tok(s)["input_ids"]))
lens = np.array(lens)
print(f"  actor seq len over {len(lens)} rows: mean={lens.mean():.1f} p50={np.percentile(lens,50):.0f} "
      f"p99={np.percentile(lens,99):.0f} p99.9={np.percentile(lens,99.9):.0f} max={lens.max()}")
for L in (384, 512, 768, 1024):
    print(f"    truncated at {L}: {(lens > L).mean()*100:.3f}% of rows")
PY
echo "=== BUILD DONE ==="
