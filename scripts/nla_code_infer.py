#!/usr/bin/env python3
"""Code decoder: Python function -> activation vector -> reconstructed function.

The code-side counterpart to nla_infer.py. Encodes a function with the frozen
base model, injects that single vector into the trained code actor, and asks it
to write the function back out. Reports how close it got.

  # reconstruct one function
  python scripts/nla_code_infer.py --code "def gcd(a,b):
      while b: a,b = b, a%b
      return a"

  # from a file, and compare against a shuffled-vector control
  python scripts/nla_code_infer.py --file mymodule.py --control

Expect the SHAPE and DOMAIN of the function back, not the exact tokens. On a
300-row held-out set: 1% exact, 7.7% structurally near-identical, 71% retrieval
at rank 1. Identifier names are the usual failure -- see the report.

NOTE the checkpoint and the --centre-mean must match. Models trained on centred
vectors need the same mean subtracted here, or the injected vector is off by a
constant with ~82% of the typical vector's magnitude.
"""
import argparse, ast, difflib, re, sys
from pathlib import Path
import numpy as np, torch

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))
_CODE_RE = re.compile(r"<code>\s*(.*?)\s*</code>", re.DOTALL)


def canon(src):
    """ast.unparse round-trip, so formatting differences are not counted as errors."""
    try:
        return ast.unparse(ast.parse(src))
    except SyntaxError as e:
        sys.exit(f"input is not valid Python: {e}")


def skeleton(src):
    try:
        return [type(n).__name__ for n in ast.walk(ast.parse(src))]
    except SyntaxError:
        return []


def user_idents(src):
    import builtins, keyword
    stop = set(keyword.kwlist) | set(dir(builtins)) | {"self", "cls"}
    try:
        t = ast.parse(src)
    except SyntaxError:
        return set()
    out = set()
    for n in ast.walk(t):
        for a in ("id", "arg", "name", "attr"):
            v = getattr(n, a, None)
            if isinstance(v, str) and v not in stop and not v.startswith("__"):
                out.add(v)
    return out


def main():
    p = argparse.ArgumentParser(description="reconstruct a Python function from its activation vector")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--code", help="function source as a string")
    src.add_argument("--file", help="path to a .py file")
    p.add_argument("--actor-ckpt", default=str(PROJ / "checkpoints/code_exp1s"),
                   help="local path OR a Hub repo id, e.g. ASSERT-KTH/tiny-nla")
    p.add_argument("--subfolder", default=None,
                   help="folder inside a Hub repo, e.g. code-decoder")
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    p.add_argument("--sidecar", default=str(PROJ / "data/coderec/exp1s_eval.parquet"),
                   help="parquet path OR its .nla_meta.yaml; either is accepted")
    p.add_argument("--centre-mean", default=str(PROJ / "data/coderec/exp1s_mean.npy"),
                   help="npy train mean; pass '' if the actor was trained on raw vectors")
    p.add_argument("--layer", type=int, default=None, help="default 2/3 depth")
    p.add_argument("--injection-scale", type=float, default=300.0)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--control", action="store_true",
                   help="also generate from a RANDOM vector, to show what the code prior alone produces")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from nla.training.injection import inject_at_marked_positions
    from nla.training.schema import normalize_activation
    from nla.training.sidecar import read_sidecar

    code = canon(Path(args.file).read_text() if args.file else args.code)

    # When --actor-ckpt is a Hub repo, pull the sidecar and centring mean from
    # the same folder rather than requiring local copies. A centred actor is
    # unusable without its mean, so this must not be left to the caller.
    if args.subfolder and not Path(args.sidecar).exists():
        from huggingface_hub import hf_hub_download
        dl = hf_hub_download(args.actor_ckpt, f"{args.subfolder}/nla_meta.yaml")
        # give it the name read_sidecar expects: <stem>.nla_meta.yaml
        stem = Path(dl).with_name("hub_ckpt")
        Path(str(stem) + ".nla_meta.yaml").write_text(Path(dl).read_text())
        args.sidecar = str(stem)
        try:
            args.centre_mean = hf_hub_download(args.actor_ckpt, f"{args.subfolder}/centre_mean.npy")
        except Exception:
            print("note: no centre_mean.npy in the repo — treating actor as raw-vector")
            args.centre_mean = ""
    # read_sidecar() appends .nla_meta.yaml itself and returns {} when the file
    # is missing -- a silent empty dict that only surfaces as a KeyError later.
    # Accept either form and fail loudly here instead.
    sc_path = re.sub(r"\.nla_meta\.yaml$", "", str(args.sidecar))
    sc = read_sidecar(sc_path)
    if not sc.get("tokens"):
        sys.exit(f"no injection metadata found for --sidecar {args.sidecar}\n"
                 f"  (looked for {sc_path}.nla_meta.yaml)")
    tk = sc["tokens"]
    inj, L, R = tk["injection_token_id"], tk["injection_left_neighbor_id"], tk["injection_right_neighbor_id"]
    inj_char = tk["injection_char"]

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # ---- encode: frozen base model, last token of the function ---------------
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map={"": args.device}).eval()
    layer = args.layer if args.layer is not None else (2 * len(base.model.layers)) // 3
    cap = {}
    h = base.model.layers[layer].register_forward_hook(
        lambda _m, _a, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
    with torch.no_grad():
        e = tok(code, return_tensors="pt", truncation=True, max_length=1024)
        ids, m = e["input_ids"].to(args.device), e["attention_mask"].to(args.device)
        base(input_ids=ids, attention_mask=m, use_cache=False)
    h.remove()
    v = cap["h"][0, int(m.sum()) - 1].float().cpu().numpy()
    print(f"encoded {int(m.sum())} tokens at layer {layer}   ‖v‖ = {np.linalg.norm(v):.1f}")
    if args.centre_mean:
        mu = np.load(args.centre_mean).reshape(-1)
        v = v - mu
        print(f"centred (mean ‖μ‖ = {np.linalg.norm(mu):.1f})")
    del base
    torch.cuda.empty_cache()

    # ---- decode: inject into the trained code actor --------------------------
    kw = {"subfolder": args.subfolder} if args.subfolder else {}
    actor = AutoModelForCausalLM.from_pretrained(
        args.actor_ckpt, torch_dtype=torch.bfloat16,
        device_map={"": args.device}, **kw).eval()
    embed = actor.get_input_embeddings()

    def generate(vec):
        bv = normalize_activation(torch.tensor(vec).unsqueeze(0), args.injection_scale).to(args.device)
        s = tok.apply_chat_template([{"role": "user", "content": f"<concept>{inj_char}</concept>"}],
                                    tokenize=False, add_generation_prompt=True)
        enc = tok(s, return_tensors="pt", add_special_tokens=False)
        i2, m2 = enc["input_ids"].to(args.device), enc["attention_mask"].to(args.device)
        hk = embed.register_forward_hook(
            lambda _m, a, o: o if o.shape[1] <= 1 else inject_at_marked_positions(a[0], o, bv, inj, L, R))
        try:
            with torch.no_grad():
                g = actor.generate(i2, attention_mask=m2, max_new_tokens=args.max_new_tokens,
                                   do_sample=False, pad_token_id=tok.pad_token_id,
                                   eos_token_id=tok.eos_token_id)
        finally:
            hk.remove()
        out = tok.decode(g[0][i2.shape[1]:], skip_special_tokens=True)
        mm = _CODE_RE.search(out)
        return mm.group(1) if mm else None

    pred = generate(v)
    print("\n" + "=" * 70 + "\nORIGINAL\n" + "=" * 70); print(code)
    print("=" * 70 + "\nRECONSTRUCTED\n" + "=" * 70); print(pred or "(no <code> block emitted)")

    if pred:
        sk = difflib.SequenceMatcher(None, skeleton(pred), skeleton(code)).ratio()
        pu, cu = user_idents(pred), user_idents(code)
        print("=" * 70)
        print(f"exact match        : {pred.strip() == code.strip()}")
        print(f"AST-skeleton sim   : {sk:.3f}   (~0.52 for two arbitrary functions)")
        print(f"identifier recall  : {len(pu & cu) / max(len(cu), 1):.2f}   shared: {sorted(pu & cu)[:8]}")

    if args.control:
        rng = np.random.default_rng(0)
        ctrl = generate(rng.normal(0, np.std(v), size=v.shape).astype(np.float32))
        print("=" * 70 + "\nCONTROL (random vector — what the code prior alone gives)\n" + "=" * 70)
        print(ctrl or "(none)")


if __name__ == "__main__":
    main()
