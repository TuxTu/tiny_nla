#!/bin/bash
#SBATCH --job-name=nla-verify-v2
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/verify_v2_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/verify_v2_%j.err

# Independent confirmation of the v2 critic run:
#   A. did the backbone actually move? (bit-level diff vs the identity-init ckpt)
#   B. does the standalone evaluator reproduce the in-training held-out FVE?
#      -- separate code path, full 9042-row eval set, not the 2048-row subset.

set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
export HF_HOME=$PROJ/cache/huggingface
export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJ"

CKPT=$PROJ/checkpoints/critic_sft_8b_v3
INIT=$PROJ/checkpoints/critic_init_8b_ufw

echo "=============== A. backbone weight movement ==============="
python - <<'PY'
from safetensors import safe_open
import torch, json
init, ckpt = "checkpoints/critic_init_8b_ufw", "checkpoints/critic_sft_8b_v3"
print("init  num_hidden_layers =", json.load(open(f"{init}/config.json"))["num_hidden_layers"])
print("v2    num_hidden_layers =", json.load(open(f"{ckpt}/config.json"))["num_hidden_layers"], "(expect 25)")
a, b = safe_open(f"{init}/model.safetensors", "pt"), safe_open(f"{ckpt}/model.safetensors", "pt")
ka, kb = set(a.keys()), set(b.keys())
print(f"\n{'tensor':46s} {'frac_changed':>13s} {'rel_delta':>11s}")
print("-" * 74)
for k in ["model.layers.0.self_attn.q_proj.weight", "model.layers.8.mlp.down_proj.weight",
          "model.layers.16.mlp.gate_proj.weight", "model.layers.24.mlp.gate_proj.weight",
          "model.embed_tokens.weight"]:
    if k in ka and k in kb:
        x = a.get_tensor(k).float(); y = b.get_tensor(k).float(); d = y - x
        print(f"{k:46s} {(d != 0).float().mean():13.4f} {(d.norm() / x.norm()):11.2e}")
    else:
        print(f"{k:46s} {'MISSING':>25s}")
print("\n(old bf16-master run: frac_changed 0.045-0.100 == backbone frozen)")
va = safe_open(f"{init}/value_head.safetensors", "pt").get_tensor("weight").float()
vb = safe_open(f"{ckpt}/value_head.safetensors", "pt").get_tensor("weight").float()
print(f"value_head: diag {va.diag().mean():.4f} -> {vb.diag().mean():.4f}   rel change {(vb-va).norm()/va.norm():.3e}")
PY

echo
echo "=============== B. standalone held-out FVE (all 9042 rows) ==============="
python scripts/nla_eval_fve.py \
    --critic-only \
    --critic-ckpt "$CKPT" \
    --data "$PROJ/data/ar_eval_8b_ufw_joined.parquet" \
    --sidecar "$PROJ/data/ar_eval_8b_ufw_joined.parquet.nla_meta.yaml" \
    --n-samples 9042 \
    --model-name Qwen/Qwen3-8B

echo "=== VERIFY DONE ==="
