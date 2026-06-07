#!/bin/bash
#SBATCH --job-name=nla-test-gen
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/test_gen_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/test_gen_%j.err
# Test if SFT actor generates valid <explanation> tags
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export HF_HOME=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/huggingface
mkdir -p "$HF_HOME"

python3 << 'PYEOF'
import torch, re, pyarrow.parquet as pq, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

PROJ = "/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla"
CKPT = f"{PROJ}/checkpoints/actor_sft"

# Load model + tokenizer
print("Loading SFT actor ...")
model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.bfloat16,
    device_map="cuda:0", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(CKPT)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model.eval()

# Load sidecar for injection info
from nla.training.sidecar import read_sidecar
sidecar = read_sidecar(f"{PROJ}/data/rl_train.parquet")
tokens = sidecar["tokens"]
inj_char = tokens["injection_char"]
inj_id = tokens["injection_token_id"]
left_id = tokens["injection_left_neighbor_id"]
right_id = tokens["injection_right_neighbor_id"]
print(f"injection: char={inj_char!r} id={inj_id} neighbors=({left_id}, {right_id})")

# Load a few real prompts
data = pq.read_table(f"{PROJ}/data/rl_train.parquet")
from nla.training.schema import INJECT_PLACEHOLDER
prompts_raw = data.column("prompt").to_pylist()
# Extract vectors as numpy array
col = data.column("activation_vector")
vectors_np = np.array([v.as_py() for v in col], dtype=np.float32)

# Test on 16 samples
from nla.training.injection import inject_at_marked_positions
N_TEST = 16
success = 0
for i in range(N_TEST):
    msgs = []
    for msg in prompts_raw[i]:
        msgs.append({"role": msg["role"],
                      "content": msg["content"].replace(INJECT_PLACEHOLDER, inj_char)})

    # Tokenize prompt
    prompt_str = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    prompt_t = torch.tensor([prompt_ids], device="cuda:0")

    # Inject activation vector via hook (uses actual input_ids from args[0])
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
    handle = embed.register_forward_hook(
        _make_gen_hook(vec, inj_id, left_id, right_id)
    )

    with torch.no_grad():
        out_ids = model.generate(prompt_t, max_new_tokens=200, do_sample=True,
                                  temperature=0.8, top_p=0.95, pad_token_id=tokenizer.pad_token_id)
    handle.remove()

    resp_ids = out_ids[0, len(prompt_ids):]
    resp_text = tokenizer.decode(resp_ids, skip_special_tokens=True)

    has_open = "<explanation>" in resp_text
    has_close = "</explanation>" in resp_text
    if has_open and has_close:
        success += 1

    if i < 3:
        print(f"\n--- Sample {i} ---")
        print(f"Response: {resp_text[:300]}...")
        print(f"Tags: open={has_open} close={has_close}")

print(f"\n=== Result: {success}/{N_TEST} generated valid <explanation> tags ===")
PYEOF
echo "=== test done ==="
