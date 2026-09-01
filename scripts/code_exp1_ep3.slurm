#!/bin/bash
#SBATCH --job-name=exp1ep3
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:4
#SBATCH --ntasks=1 --cpus-per-task=32 --mem=400G
#SBATCH --time=03:30:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/exp1ep3_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/exp1ep3_%j.err
#
# exp1 to 3 epochs (1710 steps total), resuming from the step-1000 checkpoint.
#
# Why: at step 1000 = 1.75 epochs the gap was +0.7995 and STILL CLIMBING
# (+0.018 over the last interval), and round-trip FVE had only just crossed
# zero (+0.018). The text actor needed 3 full epochs to reach its +0.4593.
# 145,935 rows / 256 = 570 steps per epoch, so 3 epochs = 1710.
#
# --resume loads weights from --output-dir but NOT optimizer state or the step
# counter, so this is "step-1000 weights + 710 fresh steps". Adam moments reset
# (brief transient) but the LR schedule is constant, so nothing else drifts.
# Eval labels will read 50..710 and correspond to global steps 1050..1710.
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

torchrun --nproc_per_node=4 --master_port=$((20000 + RANDOM % 10000)) \
  -m nla.training.train_actor_sft \
    --data "$D/exp1_train.parquet" --eval-data "$D/exp1_eval.parquet" \
    --model-name Qwen/Qwen3-8B \
    --output-dir "$PROJ/checkpoints/code_exp1_long" --resume \
    --injection-scale 300 \
    --global-batch-size 256 --micro-batch-size 4 \
    --num-steps 710 --lr 2e-5 \
    --eval-every 50 --eval-samples 512 --eval-batch-size 8 --eval-gap \
    --save-every 355 --fsdp

echo "=== generation eval @ 3 epochs ==="
python scripts/eval_code_recon.py --actor-ckpt "$PROJ/checkpoints/code_exp1_long" \
    --data "$D/exp1_eval.parquet" --n-samples 300 \
    --out "$PROJ/logs/code_exp1_ep3_correct.jsonl" || echo "(gen eval failed)"
python scripts/eval_code_recon.py --actor-ckpt "$PROJ/checkpoints/code_exp1_long" \
    --data "$D/exp1_eval.parquet" --n-samples 300 --shuffle-control \
    --out "$PROJ/logs/code_exp1_ep3_shuf.jsonl" || echo "(shuf eval failed)"

echo "=== round-trip @ 3 epochs ==="
python scripts/eval_code_roundtrip.py \
    --gen "$PROJ/logs/code_exp1_ep3_correct.jsonl" \
    --gen-shuf "$PROJ/logs/code_exp1_ep3_shuf.jsonl" \
    --out "$PROJ/logs/roundtrip_exp1_ep3.json" || echo "(roundtrip failed)"
