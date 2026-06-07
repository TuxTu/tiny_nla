#!/bin/bash
#SBATCH --job-name=nla-test-norm
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=00:30:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/test_norm_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/test_norm_%j.err
# Quick SFT with normalized injection (injection_scale = sqrt(d_model) = 50.60)
# Then test if model generates <explanation> tags
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export HF_HOME=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/huggingface
mkdir -p "$HF_HOME"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla

echo "=== SFT with normalized injection (50 steps, 4-GPU DDP) ==="
torchrun --nproc_per_node=4 --master_port=29535 \
    -m nla.training.train_actor_sft \
    --data "$PROJ/data/av_sft_train.parquet" \
    --model-name Qwen/Qwen3-4B \
    --output-dir "$PROJ/checkpoints/test_norm" \
    --micro-batch-size 4 --global-batch-size 256 \
    --lr 2e-5 --min-lr 2e-6 --lr-warmup-iters 10 --lr-decay-style cosine \
    --num-steps 50 --max-length 2048 --save-every 50 --ddp

echo ""
echo "=== Testing generation from normalized SFT ==="
python3 << 'PYEOF'
import torch, numpy as np, pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForCausalLM

PROJ = "/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla"
CKPT = f"{PROJ}/checkpoints/test_norm"

model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.bfloat16,
    device_map="cuda:0", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(CKPT)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model.eval()

from nla.training.sidecar import read_sidecar
from nla.training.schema import INJECT_PLACEHOLDER
from nla.training.injection import inject_at_marked_positions
import math

sidecar = read_sidecar(f"{PROJ}/data/rl_train.parquet")
tokens = sidecar["tokens"]
inj_char = tokens["injection_char"]
inj_id = tokens["injection_token_id"]
left_id = tokens["injection_left_neighbor_id"]
right_id = tokens["injection_right_neighbor_id"]
d_model = sidecar["extraction"]["d_model"]
injection_scale = math.sqrt(d_model)

data = pq.read_table(f"{PROJ}/data/rl_train.parquet")
prompts_raw = data.column("prompt").to_pylist()
col = data.column("activation_vector")
vectors_np = np.array([v.as_py() for v in col], dtype=np.float32)

N_TEST = 16
success = 0
for i in range(N_TEST):
    msgs = []
    for msg in prompts_raw[i]:
        msgs.append({"role": msg["role"],
                      "content": msg["content"].replace(INJECT_PLACEHOLDER, inj_char)})
    prompt_str = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    prompt_t = torch.tensor([prompt_ids], device="cuda:0")

    # Normalize vector to sqrt(d_model)
    raw_vec = torch.from_numpy(vectors_np[i]).float()
    vec_norm = raw_vec.norm()
    vec = (raw_vec / (vec_norm / injection_scale)).to(torch.bfloat16).unsqueeze(0).to("cuda:0")

    def _make_gen_hook(_vec, _inj_id, _left_id, _right_id):
        def _hook(module, args, output):
            actual_ids = args[0]
            if output.shape[1] > 1:
                return inject_at_marked_positions(
                    actual_ids, output, _vec.repeat(output.shape[0], 1),
                    _inj_id, _left_id, _right_id,
                )
            return output
        return _hook

    embed = model.get_input_embeddings()
    handle = embed.register_forward_hook(_make_gen_hook(vec, inj_id, left_id, right_id))
    with torch.no_grad():
        out_ids = model.generate(prompt_t, max_new_tokens=200, do_sample=True,
                                  temperature=0.8, top_p=0.95,
                                  pad_token_id=tokenizer.pad_token_id)
    handle.remove()
    resp_text = tokenizer.decode(out_ids[0, len(prompt_ids):], skip_special_tokens=True)

    has_exp = "<explanation>" in resp_text and "</explanation>" in resp_text
    if has_exp:
        success += 1
    if i < 3:
        print(f"\n--- Sample {i} ---")
        print(f"Response: {resp_text[:300]}...")
        print(f"Tags: <explanation>={has_exp}")

print(f"\n=== {success}/{N_TEST} generated valid <explanation> tags ===")
PYEOF
echo "=== test done ==="
