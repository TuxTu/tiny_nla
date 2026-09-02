#!/bin/bash
#SBATCH --job-name=nla-critic-ufw
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=08:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/critic_ufw_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/critic_ufw_%j.err

# Critic SFT on UltraFineWeb AV-labeled data reformatted as AR (249k rows).
# SFT-only test: does UltraFineWeb + DeepSeek labels reach original repo's 37.5% FVE?
# Document overlap with future RL is irrelevant here (no RL phase in this test).

set -euo pipefail

CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export HF_HOME=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/huggingface
export TORCHINDUCTOR_CACHE_DIR=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/torchinductor   # else torch defaults to /tmp/torchinductor_$USER
mkdir -p "$HF_HOME"

PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"

UFW="$PROJ/data/ufw_8b"
AR_TRAIN="$PROJ/data/ar_sft_train_8b_ufw.parquet"

echo "=== Step 1: Build AR critic training parquet from AV UltraFineWeb data ==="
if [ ! -f "$AR_TRAIN" ]; then
    # Vector extraction skipped 1 position (token mismatch), so explained has
    # 1 extra row. Pre-filter explained to the intersection of (doc_id, n_raw_tokens)
    # keys present in vectors, so build_training_data's exact-match assert passes.
    FILTERED="$UFW/av_sft_explained_filtered.parquet"
    python3 << PYEOF
import pyarrow.parquet as pq, pyarrow as pa, pyarrow.compute as pc
expl = pq.read_table("$UFW/av_sft_explained.parquet")
vecs = pq.read_table("$UFW/av_sft_explained_vectors.parquet")
vk = set(zip(vecs.column('doc_id').to_pylist(), vecs.column('n_raw_tokens').to_pylist()))
ek = list(zip(expl.column('doc_id').to_pylist(), expl.column('n_raw_tokens').to_pylist()))
mask = pa.array([k in vk for k in ek], type=pa.bool_())
filtered = expl.filter(mask)
pq.write_table(filtered, "$FILTERED")
print(f"filtered explained: {expl.num_rows} -> {filtered.num_rows} rows (matches {vecs.num_rows} vectors)")
PYEOF
    python -m nla.training.build_training_data \
        --explained "$FILTERED" \
        --vectors "$UFW/av_sft_explained_vectors.parquet" \
        --tokenizer Qwen/Qwen3-8B \
        --output "$AR_TRAIN" \
        --split-type ar_sft
else
    echo "  $AR_TRAIN already exists"
fi

N_ROWS=$(python -c "import pyarrow.parquet as pq; print(pq.ParquetFile('$AR_TRAIN').metadata.num_rows)")
echo "  AR training rows: $N_ROWS"

echo ""
echo "=== Step 2: Prepare critic checkpoint (identity init) ==="
CRITIC_INIT="$PROJ/checkpoints/critic_init_8b_ufw"
PREP="$PROJ/scripts/_prep_ufw_critic.py"
cat > "$PREP" << 'PYEOF'
import sys, torch
from nla.training.models import NLACriticModel
from transformers import AutoTokenizer
init_dir = sys.argv[1]
print(f'Truncating Qwen3-8B to 24 layers...')
model = NLACriticModel.from_pretrained('Qwen/Qwen3-8B', nla_num_layers=24, torch_dtype=torch.bfloat16)
with torch.no_grad():
    model.value_head.weight.copy_(torch.eye(model.config.hidden_size, dtype=torch.bfloat16))
model.save_pretrained(init_dir)
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B', trust_remote_code=True)
tok.padding_side = 'right'
tok.save_pretrained(init_dir)
print('Done.')
PYEOF
python "$PREP" "$CRITIC_INIT"
rm "$PREP"

GLOBAL_BATCH=256
NUM_STEPS=$((N_ROWS / GLOBAL_BATCH))
echo ""
echo "=== Step 3: Train critic ($N_ROWS rows, $NUM_STEPS steps) ==="

torchrun --nproc_per_node=2 --master_port=29537 \
    -m nla.training.train_critic_sft \
    --data "$AR_TRAIN" \
    --model-name "$CRITIC_INIT" \
    --output-dir "$PROJ/checkpoints/critic_sft_8b_ufw" \
    --micro-batch-size 2 --global-batch-size $GLOBAL_BATCH \
    --lr 2e-5 --min-lr 2e-6 --lr-warmup-iters 50 --lr-decay-style cosine \
    --num-steps $NUM_STEPS --max-length 512 --value-head-depth 1 --save-every 200 \
    --fsdp

echo "=== Done ==="
