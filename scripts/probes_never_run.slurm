#!/bin/bash
#SBATCH --job-name=nla-probes
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/probes_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/probes_%j.err
#
# The two probes written in a5a1ed7 that were never executed:
#   nla_multilingual_probe  -- does the actor describe ACTUAL Chinese content, or
#                              hallucinate? Paired zh/en near-translations, so if
#                              explanations match the vector encodes meaning not
#                              surface form. Actor saw only UltraFineWeb split:en.
#   nla_chat_probe          -- supplement queries: after the explanation, ask
#                              follow-up questions with the vector still in
#                              context. Control: every question repeated with a
#                              ZERO vector, since a fluent wrong answer looks
#                              identical to a fluent right one.
#
# Run on the best SFT actor. Both are inference-only, no critic needed.
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
CKPT=$PROJ/checkpoints/actor_sft_8b_s50fix
SIDE=$PROJ/data/av_eval_8b_ufw_joined.parquet.nla_meta.yaml

echo "############ CROSS-LINGUISTIC PROBE ############"
python scripts/nla_multilingual_probe.py --actor-ckpt "$CKPT" --sidecar "$SIDE" || echo "(multilingual failed)"
echo; echo "############ SUPPLEMENT-QUERY (CHAT) PROBE ############"
python scripts/nla_chat_probe.py --actor-ckpt "$CKPT" --sidecar "$SIDE" || echo "(chat failed)"
