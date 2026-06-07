#!/bin/bash
#SBATCH --job-name=nla-diag
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/diag_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/diag_%j.err
# Diagnostic: compare generation with vs without injection hook
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export HF_HOME=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/huggingface
mkdir -p "$HF_HOME"

python3 << 'PYEOF'
import torch, numpy as np, pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForCausalLM

PROJ = "/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla"

# Load base model (not SFT) to test injection
print("=== Loading base Qwen3-4B ===")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", torch_dtype=torch.bfloat16,
    device_map="cuda:0", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model.eval()

# Load data + sidecar
from nla.training.sidecar import read_sidecar
from nla.training.schema import INJECT_PLACEHOLDER
from nla.training.injection import inject_at_marked_positions

data = pq.read_table(f"{PROJ}/data/rl_train.parquet")
col = data.column("activation_vector")
vectors_np = np.array([v.as_py() for v in col], dtype=np.float32)

sidecar = read_sidecar(f"{PROJ}/data/rl_train.parquet")
tokens = sidecar["tokens"]
inj_char = tokens["injection_char"]
inj_id = tokens["injection_token_id"]
left_id = tokens["injection_left_neighbor_id"]
right_id = tokens["injection_right_neighbor_id"]
print(f"injection: char={inj_char!r} id={inj_id} neighbors=({left_id},{right_id})")

prompts_raw = data.column("prompt").to_pylist()

# Test 3 samples, each with and without injection
for i in range(3):
    msgs = []
    for msg in prompts_raw[i]:
        msgs.append({"role": msg["role"],
                      "content": msg["content"].replace(INJECT_PLACEHOLDER, inj_char)})
    prompt_str = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)

    # Check if injection char is in the prompt
    has_inj = inj_id in prompt_ids
    print(f"\n=== Sample {i}: inj_char in prompt = {has_inj} ===")

    prompt_t = torch.tensor([prompt_ids], device="cuda:0")

    # --- WITHOUT injection ---
    with torch.no_grad():
        out_no_inj = model.generate(prompt_t, max_new_tokens=100, do_sample=True,
                                     temperature=0.8, top_p=0.95,
                                     pad_token_id=tokenizer.pad_token_id)
    resp_no_inj = tokenizer.decode(out_no_inj[0, len(prompt_ids):], skip_special_tokens=True)

    # --- WITH injection ---
    vec = torch.from_numpy(vectors_np[i]).to("cuda:0").unsqueeze(0)
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
        out_with_inj = model.generate(prompt_t, max_new_tokens=100, do_sample=True,
                                       temperature=0.8, top_p=0.95,
                                       pad_token_id=tokenizer.pad_token_id)
    handle.remove()
    resp_with_inj = tokenizer.decode(out_with_inj[0, len(prompt_ids):], skip_special_tokens=True)

    print(f"  WITHOUT injection: {resp_no_inj[:150]}...")
    print(f"  WITH injection:    {resp_with_inj[:150]}...")
    same = resp_no_inj == resp_with_inj
    print(f"  Same output: {same}")

PYEOF
echo "=== diag done ==="
