#!/bin/bash
#SBATCH --job-name=nla-rlprobe
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=01:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/rlprobe_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/rlprobe_%j.err
# How large can --rollout-batch go now that the reference model is FSDP-sharded?
#
# run1 used rollout-batch 1 = ONE prompt per step, so every GRPO update was
# driven by a single activation vector (300 prompts total = 0.12% of the pool).
# That is the dominant reason steps 50-300 looked like drift.
#
# Static memory per GPU: 59.2 -> 47.2 GB after sharding the reference model
# (16 GB unsharded bf16 copy on every rank -> ~4 GB). ~32 GB now free for
# rollout activations, up from ~20.
#
# Must reach step 2: AdamW allocates exp_avg/exp_avg_sq lazily on the first
# .step(), which is what killed four earlier attempts AT step 1, not step 0.
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
export HF_HOME=$PROJ/cache/huggingface
export TORCHINDUCTOR_CACHE_DIR=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/torchinductor   # else torch defaults to /tmp/torchinductor_$USER
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd "$PROJ"
B=${B:-4}
RESP=${RESP:-150}
MAXLEN=${MAXLEN:-320}
rm -rf "$PROJ/checkpoints/rl_probe_b${B}_${RESP}"
echo "########## probing B=$B x8 = $((B*8)) gens/step, resp=$RESP, max_len=$MAXLEN ##########"
torchrun --nproc_per_node=4 --master_port=29631 \
    -m nla.training.train_rl \
    --data "$PROJ/data/rl_train_8b_ufw_joined.parquet" \
    --model-name Qwen/Qwen3-8B \
    --actor-ckpt  "$PROJ/checkpoints/actor_sft_8b_s50fix" \
    --critic-ckpt "$PROJ/checkpoints/critic_sft_8b_s50" \
    --output-dir  "$PROJ/checkpoints/rl_probe_b${B}_${RESP}" \
    --injection-scale 300 --freeze-critic \
    --n-samples 8 --rollout-batch $B --micro-batch-size 1 \
    --lr-actor 1.5e-6 --lr-critic 1.5e-6 \
    --kl-coef 0.01 --kl-type per_seq \
    --max-response-len $RESP --max-length $MAXLEN \
    --num-steps 5 --save-every 10000 \
    --param-dtype float32 --fsdp
echo "=== PROBE B=$B SURVIVED ==="
rm -rf "$PROJ/checkpoints/rl_probe_b${B}_${RESP}"
