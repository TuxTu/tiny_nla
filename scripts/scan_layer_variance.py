"""Quick scan: compute mean-predictor baselines for candidate extraction layers."""
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model_name = 'Qwen/Qwen3-8B'
print(f'Loading {model_name}...')
tok = AutoTokenizer.from_pretrained(model_name)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
tok.padding_side = 'right'

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map='auto'
).eval()

layers = model.model.layers
num_layers = len(layers)
d_model = model.config.hidden_size
mse_scale = float(np.sqrt(d_model))

# Load 200 short texts (~30s)
ds = load_dataset('HuggingFaceFW/fineweb', split='train', streaming=True)
texts = []
for doc in ds:
    t = doc['text'][:2000]
    if len(t) > 200:
        texts.append(t)
    if len(texts) >= 200:
        break
print(f'Sampled {len(texts)} texts')

def normalize(v, scale):
    return torch.nn.functional.normalize(v.float(), p=2, dim=-1) * scale

# Test layers at various depth fractions
candidates = sorted(set(int(num_layers * p) for p in [0.17, 0.25, 0.33, 0.42, 0.50, 0.58, 0.67, 0.75, 0.83]))
print(f'Layers: {num_layers}, candidates: {candidates}')

for layer_idx in candidates:
    captured = {}

    def make_hook(cap_dict):
        def hook(m, args, output):
            h = output[0] if isinstance(output, tuple) else output
            cap_dict['h'] = h.detach().clone()
        return hook

    handle = layers[layer_idx].register_forward_hook(make_hook(captured))

    vecs = []
    bs = 8
    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        enc = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
        input_ids = enc['input_ids'].to(model.device)
        attn = enc['attention_mask'].to(model.device)
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attn)
        lengths = attn.sum(dim=1)
        h = captured['h']
        for j, l in enumerate(lengths):
            vecs.append(h[j, l-1].float().cpu().numpy())
    handle.remove()

    v = torch.from_numpy(np.stack(vecs))
    v_norm = normalize(v, mse_scale)
    mu = v_norm.mean(dim=0, keepdim=True)
    mu_normed = normalize(mu, mse_scale)
    meannorm = ((v_norm - mu_normed) ** 2).mean().item()
    rawvar = ((v_norm - mu) ** 2).mean().item()

    # For reference: original Qwen2.5-7B layer 20 has meannorm=0.938, rawvar=0.72
    print(f'  L{layer_idx:2d}/{num_layers} ({layer_idx/num_layers:.0%}): '
          f'meannorm={meannorm:.4f}  rawvar={rawvar:.4f}  '
          f'{"★ HIGH" if meannorm > 0.85 else ""}')

print(f'\nOriginal Qwen2.5-7B L20/28: meannorm=0.938  rawvar=0.72')
print('Aim for meannorm > 0.85 to give the critic room to learn.')
