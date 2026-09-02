#!/usr/bin/env python3
"""Standalone demo for the tiny_nla code decoder. No repo checkout needed.

    pip install torch transformers huggingface_hub numpy pyyaml
    hf auth login
    hf download ASSERT-KTH/tiny-nla nla_code_demo.py --local-dir .
    python nla_code_demo.py --code 'def is_palindrome(s):
        s = s.lower()
        return s == s[::-1]'

Encodes a Python function to ONE 4096-float activation vector with the frozen
base model, injects that vector into the trained decoder, and asks it to write
the function back out.

This is not a lossless codec. On a 300-function held-out set: 1% exact, 7.7%
structurally near-identical, 71% retrieval@1 (is the output nearest its own
target among 300). The typical error is a SIBLING -- ask for send_video and get
send_document, same signature, wrong entity. Always run with --control, which
generates from a random vector so you can see what the code prior alone gives.

Needs one GPU with >=24 GB. The two models load sequentially, not together.
"""
import argparse, ast, builtins, difflib, keyword, re, sys
from pathlib import Path
import numpy as np, torch, yaml
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

CODE_RE = re.compile(r"<code>\s*(.*?)\s*</code>", re.DOTALL)
STOP = set(keyword.kwlist) | set(dir(builtins)) | {"self", "cls"}


def inject_at_marked_positions(input_ids, embeddings, vectors, inj_id, left_id, right_id):
    """Overwrite embedding rows at injection markers. Copied verbatim from
    nla/training/injection.py -- the most correctness-critical path: if
    injection fails the model sees the literal marker char and emits Chinese.
    The neighbour check prevents false positives from the marker appearing in
    generated text."""
    out = embeddings.clone()
    vectors = vectors.to(out.device, out.dtype)
    seq_len = input_ids.shape[-1]
    vec_idx = 0
    for b, p in (input_ids == inj_id).nonzero(as_tuple=False).tolist():
        if p == 0 or p == seq_len - 1:
            continue
        if input_ids[b, p - 1] != left_id or input_ids[b, p + 1] != right_id:
            continue
        out[b, p] = vectors[vec_idx]
        vec_idx += 1
    if vec_idx != vectors.shape[0]:
        raise RuntimeError(f"found {vec_idx} injection sites, expected {vectors.shape[0]}")
    return out


def normalize_activation(v, target_scale):
    if target_scale is None:
        return v
    return v / (v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12) / target_scale).to(v.dtype)


def canon(src):
    try:
        return ast.unparse(ast.parse(src))
    except SyntaxError as e:
        sys.exit(f"input is not valid Python: {e}")


def skeleton(src):
    try:
        return [type(n).__name__ for n in ast.walk(ast.parse(src))]
    except SyntaxError:
        return []


def idents(src):
    try:
        t = ast.parse(src)
    except SyntaxError:
        return set()
    o = set()
    for n in ast.walk(t):
        for a in ("id", "arg", "name", "attr"):
            v = getattr(n, a, None)
            if isinstance(v, str) and v not in STOP and not v.startswith("__"):
                o.add(v)
    return o


def main():
    p = argparse.ArgumentParser(description="reconstruct a Python function from its activation vector")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--code", help="function source")
    g.add_argument("--file", help="path to a .py file")
    p.add_argument("--repo", default="ASSERT-KTH/tiny-nla")
    p.add_argument("--subfolder", default="code-decoder")
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    p.add_argument("--injection-scale", type=float, default=300.0)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--control", action="store_true",
                   help="also generate from a random vector -- the code prior with no information")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    code = canon(Path(a.file).read_text() if a.file else a.code)

    meta = yaml.safe_load(open(hf_hub_download(a.repo, f"{a.subfolder}/nla_meta.yaml")))["tokens"]
    inj, L, R, ch = (meta["injection_token_id"], meta["injection_left_neighbor_id"],
                     meta["injection_right_neighbor_id"], meta["injection_char"])
    # A centred decoder given a raw vector produces fluent, plausible, entirely
    # unrelated code with no error. Fetch the mean rather than trust the caller.
    try:
        mu = np.load(hf_hub_download(a.repo, f"{a.subfolder}/centre_mean.npy")).reshape(-1)
    except Exception:
        mu = None
        print("note: no centre_mean.npy in repo -- treating decoder as raw-vector")

    tok = AutoTokenizer.from_pretrained(a.base_model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(
        a.base_model, torch_dtype=torch.bfloat16, device_map={"": a.device}).eval()
    layer = (2 * len(base.model.layers)) // 3
    cap = {}
    h = base.model.layers[layer].register_forward_hook(
        lambda _m, _i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
    with torch.no_grad():
        e = tok(code, return_tensors="pt", truncation=True, max_length=1024)
        ids, m = e["input_ids"].to(a.device), e["attention_mask"].to(a.device)
        base(input_ids=ids, attention_mask=m, use_cache=False)
    h.remove()
    v = cap["h"][0, int(m.sum()) - 1].float().cpu().numpy()
    print(f"encoded {int(m.sum())} tokens at layer {layer}   ‖v‖ = {np.linalg.norm(v):.1f}")
    if mu is not None:
        v = v - mu
        print(f"centred (‖μ‖ = {np.linalg.norm(mu):.1f})")
    del base
    torch.cuda.empty_cache()

    actor = AutoModelForCausalLM.from_pretrained(
        a.repo, subfolder=a.subfolder, torch_dtype=torch.bfloat16,
        device_map={"": a.device}).eval()
    embed = actor.get_input_embeddings()

    def gen(vec):
        bv = normalize_activation(torch.tensor(vec).unsqueeze(0), a.injection_scale).to(a.device)
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
        mm = CODE_RE.search(tok.decode(out[0][i2.shape[1]:], skip_special_tokens=True))
        return mm.group(1) if mm else None

    pred = gen(v)
    bar = "=" * 70
    print(f"\n{bar}\nORIGINAL\n{bar}\n{code}")
    print(f"{bar}\nRECONSTRUCTED\n{bar}\n{pred or '(no <code> block emitted)'}")
    if pred:
        pu, cu = idents(pred), idents(code)
        print(bar)
        print(f"exact match       : {pred.strip() == code.strip()}")
        print(f"AST-skeleton sim  : {difflib.SequenceMatcher(None, skeleton(pred), skeleton(code)).ratio():.3f}"
              f"   (~0.52 for two arbitrary functions)")
        print(f"identifier recall : {len(pu & cu) / max(len(cu), 1):.2f}   shared: {sorted(pu & cu)[:8]}")
    if a.control:
        rng = np.random.default_rng(0)
        print(f"{bar}\nCONTROL (random vector -- the code prior with no information)\n{bar}")
        print(gen(rng.normal(0, np.std(v), size=v.shape).astype(np.float32)) or "(none)")


if __name__ == "__main__":
    main()
