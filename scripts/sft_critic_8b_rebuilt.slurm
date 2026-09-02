#!/bin/bash
#SBATCH --job-name=nla-critic-rebuilt
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=08:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/critic_rebuilt_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/critic_rebuilt_%j.err

# Critic SFT on REBUILT AR data (document-level split, 125k rows, 10 pos/doc, zero overlap)
# Fresh training from scratch with identity value_head init (matching original repo).

set -euo pipefail

CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export HF_HOME=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/huggingface
export TORCHINDUCTOR_CACHE_DIR=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/torchinductor   # else torch defaults to /tmp/torchinductor_$USER
mkdir -p "$HF_HOME"

PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"

echo "=== Critic SFT 8B on REBUILT AR data ==="
echo "Data: $PROJ/data/ar_sft_train_8b.parquet"

# Prepare critic checkpoint via separate Python script (avoids bash quoting issues)
CRITIC_INIT="$PROJ/checkpoints/critic_init_8b_rebuilt"
PREP_SCRIPT="$PROJ/scripts/_prep_critic_init.py"

cat > "$PREP_SCRIPT" << 'PYEOF'
import sys, os, torch
from nla.training.models import NLACriticModel
from transformers import AutoTokenizer

init_dir = sys.argv[1]
print(f'Loading Qwen/Qwen3-8B, truncating to 24 layers...')
model = NLACriticModel.from_pretrained(
    'Qwen/Qwen3-8B', nla_num_layers=24, torch_dtype=torch.bfloat16,
)
with torch.no_grad():
    model.value_head.weight.copy_(torch.eye(model.config.hidden_size, dtype=torch.bfloat16))
print(f'Saving to {init_dir}...')
model.save_pretrained(init_dir)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B', trust_remote_code=True)
tokenizer.padding_side = 'right'
tokenizer.save_pretrained(init_dir)
print('Done.')
PYEOF

echo "=== Step 0: Prepare critic checkpoint (identity init) ==="
python "$PREP_SCRIPT" "$CRITIC_INIT"
rm "$PREP_SCRIPT"

# Count rows for step calculation
N_ROWS=$(python -c "import pyarrow.parquet as pq; print(pq.ParquetFile('$PROJ/data/ar_sft_train_8b.parquet').metadata.num_rows)")
GLOBAL_BATCH=256
NUM_STEPS=$((N_ROWS / GLOBAL_BATCH))
echo "=== Training: $N_ROWS rows, global_batch=$GLOBAL_BATCH, $NUM_STEPS steps ==="

torchrun --nproc_per_node=4 --master_port=29535 \
    -m nla.training.train_critic_sft \
    --data "$PROJ/data/ar_sft_train_8b.parquet" \
    --model-name "$CRITIC_INIT" \
    --output-dir "$PROJ/checkpoints/critic_sft_8b_rebuilt" \
    --micro-batch-size 2 --global-batch-size $GLOBAL_BATCH \
    --lr 2e-5 --min-lr 2e-6 --lr-warmup-iters 50 --lr-decay-style cosine \
    --num-steps $NUM_STEPS --max-length 512 --value-head-depth 1 --save-every 200 \
    --fsdp

echo "=== Done ==="
