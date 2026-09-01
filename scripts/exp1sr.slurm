#!/bin/bash
#SBATCH --job-name=exp1sr
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:4
#SBATCH --ntasks=1 --cpus-per-task=32 --mem=400G
#SBATCH --time=03:30:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/exp1sr_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/exp1sr_%j.err
#
# Best shot at a NON-ZERO reconstruction rate. Three changes, each measured:
#
#  1. SHORT programs only (<=15 words, mean 13.5). tok-sim falls monotonically
#     with length (0.344 at <15 words -> 0.217 at 30-50), and the bit budget
#     says this is the regime reconstruction is affordable in. Never trained
#     on it alone.
#  2. MEAN-CENTRED vectors. The constant offset is 82% of a typical vector's
#     magnitude and carries no program identity; removing it drops
#     nearest-neighbour cosine 0.90 -> 0.69 and near-twins 13.3% -> 1.8%.
#     Invertible, so nothing is lost.
#  3. BEST-OF-32 at eval, not just greedy. Every previous number is pass@1.
#
# 35,014 rows / 256 = 137 steps per epoch, so 600 steps = 4.4 epochs. exp2
# collapsed at 4.9 epochs on 13k rows -- watch held-out CE for the same turn.
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
export HF_HOME=$PROJ/cache/huggingface
export TORCHINDUCTOR_CACHE_DIR=$PROJ/cache/torchinductor
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJ"
D=$PROJ/data/coderec
OUT=$PROJ/checkpoints/code_exp1sr

torchrun --nproc_per_node=4 --master_port=$((20000 + RANDOM % 10000)) \
  -m nla.training.train_actor_sft \
    --data "$D/exp1sr_train.parquet" --eval-data "$D/exp1sr_eval.parquet" \
    --model-name Qwen/Qwen3-8B --output-dir "$OUT" \
    --injection-scale 300 \
    --global-batch-size 256 --micro-batch-size 4 \
    --num-steps 600 --lr 2e-5 \
    --eval-every 50 --eval-samples 512 --eval-batch-size 8 --eval-gap \
    --save-every 300 --fsdp

echo "=== greedy generation eval ==="
python scripts/eval_code_recon.py --actor-ckpt "$OUT" --data "$D/exp1sr_eval.parquet" \
    --n-samples 300 --out "$PROJ/logs/exp1sr_gen.jsonl" || echo "(failed)"
python scripts/eval_code_recon.py --actor-ckpt "$OUT" --data "$D/exp1sr_eval.parquet" \
    --n-samples 300 --shuffle-control --out "$PROJ/logs/exp1sr_gen_shuf.jsonl" || echo "(failed)"

echo "=== round-trip ==="
python scripts/eval_code_roundtrip.py --gen "$PROJ/logs/exp1sr_gen.jsonl" \
    --gen-shuf "$PROJ/logs/exp1sr_gen_shuf.jsonl" \
    --out "$PROJ/logs/roundtrip_exp1sr.json" || echo "(failed)"

echo "=== BEST-OF-32 (the reconstruction-rate shot) ==="
python scripts/eval_best_of_n.py --actor-ckpt "$OUT" --data "$D/exp1sr_eval.parquet" \
    --n-samples 150 --n-candidates 32 --temperature 0.8 \
    --out "$PROJ/logs/exp1sr_bestofn.json" || echo "(failed)"
