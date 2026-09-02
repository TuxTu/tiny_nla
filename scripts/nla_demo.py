#!/usr/bin/env python3
"""tiny_nla demo — one entry point, no model to choose.

    pip install torch transformers huggingface_hub numpy pyyaml
    hf auth login
    hf download TuHan/tiny-nla nla_demo.py --local-dir .

    python nla_demo.py 'def gcd(a, b):
        while b: a, b = b, a % b
        return a'                              # -> reconstructs the function

    python nla_demo.py 'The Federal Reserve announced yesterday that it would'
                                               # -> explains the activation

    python nla_demo.py --file mymodule.py --control

Compresses the input to ONE 4096-float activation vector from a frozen
Qwen3-8B, then asks a trained decoder to write it back out. Valid Python is
routed to the code decoder; anything else to the text explainer. The model is
always whatever currently sits at the pinned Hub paths -- there is no flag to
pick an older one.

Needs one GPU with >=24 GB. Models load sequentially, not together.
"""
import argparse, ast, builtins, difflib, keyword, re, sys
from pathlib import Path
import numpy as np, torch, yaml
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "TuHan/tiny-nla"
CODE_SUB, TEXT_SUB = "code-decoder", "nla/actor"
BASE = "Qwen/Qwen3-8B"
CODE_RE = re.compile(r"<code>\s*(.*?)\s*</code>", re.DOTALL)
EXPL_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)
STOP = set(keyword.kwlist) | set(dir(builtins)) | {"self", "cls"}


def inject_at_marked_positions(input_ids, embeddings, vectors, inj_id, left_id, right_id):
    """Verbatim from nla/training/injection.py. The neighbour check prevents
    false positives; if injection silently fails the model emits Chinese."""
    out = embeddings.clone()
    vectors = vectors.to(out.device, out.dtype)
    n, k = input_ids.shape[-1], 0
    for b, p in (input_ids == inj_id).nonzero(as_tuple=False).tolist():
        if p == 0 or p == n - 1:
            continue
        if input_ids[b, p - 1] != left_id or input_ids[b, p + 1] != right_id:
            continue
        out[b, p] = vectors[k]; k += 1
    if k != vectors.shape[0]:
        raise RuntimeError(f"found {k} injection sites, expected {vectors.shape[0]}")
    return out


def normalize_activation(v, scale):
    return v / (v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12) / scale).to(v.dtype)


def is_python(s):
    """Route by trying to parse. A bare sentence is not valid Python."""
    try:
        t = ast.parse(s)
        return bool(t.body) and not (len(t.body) == 1 and isinstance(t.body[0], ast.Expr)
                                     and isinstance(getattr(t.body[0], "value", None), ast.Constant))
    except SyntaxError:
        return False


def idents(src):
    try:
        t = ast.parse(src)
    except SyntaxError:
        return set()
    return {v for n in ast.walk(t) for a in ("id", "arg", "name", "attr")
            if isinstance(v := getattr(n, a, None), str) and v not in STOP and not v.startswith("__")}


def main():
    p = argparse.ArgumentParser(description="tiny_nla demo — compress to one vector, decode it back")
    p.add_argument("input", nargs="?", help="Python function, or any text")
    p.add_argument("--file", help="read the input from a file instead")
    p.add_argument("--control", action="store_true",
                   help="also decode a RANDOM vector — what the model produces with no information")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if not a.input and not a.file:
        p.error("give some input, or --file")
    raw = Path(a.file).read_text() if a.file else a.input

    code_mode = is_python(raw)
    if code_mode:
        raw = ast.unparse(ast.parse(raw))          # canonicalise so formatting is not counted
    sub = CODE_SUB if code_mode else TEXT_SUB
    print(f"mode: {'CODE reconstruction' if code_mode else 'TEXT explanation'}   ({REPO}/{sub})")

    meta = yaml.safe_load(open(hf_hub_download(REPO, f"{sub}/nla_meta.yaml")))["tokens"]
    inj, L, R, ch = (meta["injection_token_id"], meta["injection_left_neighbor_id"],
                     meta["injection_right_neighbor_id"], meta["injection_char"])
    try:      # a centred decoder given a raw vector yields fluent, unrelated output
        mu = np.load(hf_hub_download(REPO, f"{sub}/centre_mean.npy")).reshape(-1)
    except Exception:
        mu = None

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map={"": a.device}).eval()
    layer = (2 * len(base.model.layers)) // 3
    cap = {}
    h = base.model.layers[layer].register_forward_hook(
        lambda _m, _i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
    with torch.no_grad():
        e = tok(raw, return_tensors="pt", truncation=True, max_length=1024)
        ids, m = e["input_ids"].to(a.device), e["attention_mask"].to(a.device)
        base(input_ids=ids, attention_mask=m, use_cache=False)
    h.remove()
    v = cap["h"][0, int(m.sum()) - 1].float().cpu().numpy()
    print(f"encoded {int(m.sum())} tokens at layer {layer}   ‖v‖={np.linalg.norm(v):.1f}"
          + (f"   centred (‖μ‖={np.linalg.norm(mu):.1f})" if mu is not None else ""))
    if mu is not None:
        v = v - mu
    del base; torch.cuda.empty_cache()

    actor = AutoModelForCausalLM.from_pretrained(
        REPO, subfolder=sub, torch_dtype=torch.bfloat16, device_map={"": a.device}).eval()
    embed = actor.get_input_embeddings()

    def gen(vec):
        bv = normalize_activation(torch.tensor(vec).unsqueeze(0), 300.0).to(a.device)
        s = tok.apply_chat_template([{"role": "user", "content": f"<concept>{ch}</concept>"}],
                                    tokenize=False, add_generation_prompt=True)
        enc = tok(s, return_tensors="pt", add_special_tokens=False)
        i2, m2 = enc["input_ids"].to(a.device), enc["attention_mask"].to(a.device)
        hk = embed.register_forward_hook(
            lambda _m, i, o: o if o.shape[1] <= 1 else inject_at_marked_positions(i[0], o, bv, inj, L, R))
        try:
            with torch.no_grad():
                out = actor.generate(i2, attention_mask=m2, max_new_tokens=a.max_new_tokens,
                                     do_sample=False, pad_token_id=tok.pad_token_id,
                                     eos_token_id=tok.eos_token_id)
        finally:
            hk.remove()
        txt = tok.decode(out[0][i2.shape[1]:], skip_special_tokens=True)
        mm = (CODE_RE if code_mode else EXPL_RE).search(txt)
        return mm.group(1) if mm else txt.strip()[:600]

    bar = "=" * 70
    pred = gen(v)
    print(f"\n{bar}\nINPUT\n{bar}\n{raw[:600]}")
    print(f"{bar}\n{'RECONSTRUCTED' if code_mode else 'EXPLANATION'}\n{bar}\n{pred}")
    if code_mode and pred:
        pu, cu = idents(pred), idents(raw)
        sk = difflib.SequenceMatcher(None,
                                     [type(n).__name__ for n in ast.walk(ast.parse(pred))] if is_python(pred) else [],
                                     [type(n).__name__ for n in ast.walk(ast.parse(raw))]).ratio()
        print(bar)
        print(f"exact match       : {pred.strip() == raw.strip()}")
        print(f"AST-skeleton sim  : {sk:.3f}   (~0.52 for two arbitrary functions)")
        print(f"identifier recall : {len(pu & cu)/max(len(cu),1):.2f}   shared: {sorted(pu & cu)[:8]}")
    if a.control:
        rng = np.random.default_rng(0)
        print(f"{bar}\nCONTROL — random vector, i.e. the model's prior with no information\n{bar}")
        print(gen(rng.normal(0, np.std(v), size=v.shape).astype(np.float32)))


if __name__ == "__main__":
    main()
