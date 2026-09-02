#!/usr/bin/env python3
"""Generate code from injected vectors and score it.

The conditioning gap (logged during training by --eval-gap) says whether the
model READS the vector. This says whether what it reads is ENOUGH -- which is
the actual bit-budget question.

Metrics, in increasing order of leniency:
  exact       character-identical to the target (both sides are ast.unparse'd,
              so this is not fighting formatting variance)
  ast_equal   same AST after re-parse -- ignores whitespace only
  tok_sim     1 - normalised token edit distance; shows whether failure is a
              cliff or a graceful degradation
  parse_ok    generated text is syntactically valid Python at all

Always run with --shuffle-control too: high exact-match with a shuffled vector
means the model memorised the corpus, not that the channel works.
"""

import argparse
import ast
import difflib
import json
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

PROJ = Path(__file__).resolve().parent.parent
_CODE_RE = re.compile(r"<code>\s*(.*?)\s*</code>", re.DOTALL)


def extract_code(s):
    m = _CODE_RE.search(s)
    return m.group(1) if m else None


def ast_equal(a, b):
    """NB: ast.dump includes every identifier and literal, so with both sides
    already ast.unparse'd this is nearly as strict as exact match. It is NOT a
    structural metric -- see norm_ast for that."""
    try:
        return ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))
    except SyntaxError:
        return False


def _parse(s):
    try:
        return ast.parse(s or "")
    except SyntaxError:
        return None


def skeleton(t):
    """Control flow only: node types, no names or literals. Note the baseline is
    high (~0.44 between two arbitrary Python functions), so read it against the
    shuffled control, never on its own."""
    return [type(n).__name__ for n in ast.walk(t)] if t else None


def norm_ast(t):
    """Structure + literals with identifiers canonicalised -- 'same code modulo
    renaming'. This is the genuinely lenient structural metric."""
    if t is None:
        return None
    t = ast.parse(ast.unparse(t))
    m = {}
    for n in ast.walk(t):
        for attr in ("id", "arg", "name", "attr"):
            v = getattr(n, attr, None)
            if isinstance(v, str):
                m.setdefault(v, f"V{len(m)}")
                setattr(n, attr, m[v])
    return ast.dump(t)


def arity(t):
    if t is None:
        return -1
    return next((len(n.args.args) for n in ast.walk(t)
                 if isinstance(n, ast.FunctionDef)), -1)


def idents(s):
    return set(re.findall(r"[A-Za-z_]\w*", s or ""))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--actor-ckpt", required=True)
    p.add_argument("--data", required=True, help="eval parquet from build_code_sft.py")
    p.add_argument("--model-name", default="Qwen/Qwen3-8B")
    p.add_argument("--n-samples", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--shuffle-control", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    import sys
    sys.path.insert(0, str(PROJ))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from nla.training.injection import inject_at_marked_positions
    from nla.training.schema import normalize_activation
    from nla.training.sidecar import read_sidecar

    sc = read_sidecar(args.data)
    tokens = sc["tokens"]
    inj_id = tokens["injection_token_id"]
    left_id, right_id = tokens["injection_left_neighbor_id"], tokens["injection_right_neighbor_id"]

    t = pq.read_table(args.data)
    n = min(args.n_samples, t.num_rows)
    prompts = [m[0]["content"] for m in t.column("prompt").to_pylist()[:n]]
    targets = [extract_code(r) for r in t.column("response").to_pylist()[:n]]
    vecs = torch.tensor(np.array(t.column("activation_vector").to_pylist()[:n], dtype=np.float32))

    inj_vecs = vecs[torch.randperm(len(vecs), generator=torch.Generator().manual_seed(0))] \
        if args.shuffle_control else vecs
    if args.shuffle_control:
        print("SHUFFLE CONTROL: permuted vectors injected, scored against true targets")

    tok = AutoTokenizer.from_pretrained(args.actor_ckpt)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.actor_ckpt, torch_dtype=torch.bfloat16, device_map={"": args.device}).eval()

    d_model = model.config.hidden_size
    scale = 2.5 * (d_model ** 0.5)
    embed = model.get_input_embeddings()

    rows = []
    with torch.no_grad():
        for i in range(0, n, args.batch_size):
            bp = prompts[i:i + args.batch_size]
            bv = normalize_activation(inj_vecs[i:i + args.batch_size], scale).to(args.device)
            strs = [tok.apply_chat_template([{"role": "user", "content": c}],
                                            tokenize=False, add_generation_prompt=True) for c in bp]
            enc = tok(strs, return_tensors="pt", padding=True, add_special_tokens=False)
            ids, mask = enc["input_ids"].to(args.device), enc["attention_mask"].to(args.device)

            def _hook(_m, a, out):
                if out.shape[1] <= 1:
                    return out
                return inject_at_marked_positions(a[0], out, bv, inj_id, left_id, right_id)

            h = embed.register_forward_hook(_hook)
            try:
                gen = model.generate(ids, attention_mask=mask, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=tok.pad_token_id,
                                     eos_token_id=tok.eos_token_id)
            finally:
                h.remove()
            for j in range(len(bp)):
                txt = tok.decode(gen[j][ids.shape[1]:], skip_special_tokens=True)
                rows.append({"idx": i + j, "gen": txt, "pred": extract_code(txt),
                             "target": targets[i + j]})
            if (i // args.batch_size) % 10 == 0:
                print(f"  {i + len(bp)}/{n}", flush=True)

    ok = [r for r in rows if r["pred"] and r["target"]]
    exact = sum(r["pred"].strip() == r["target"].strip() for r in ok)
    aeq = sum(ast_equal(r["pred"], r["target"]) for r in ok)
    parse = sum(_safe_parse(r["pred"]) for r in ok)
    sims = [difflib.SequenceMatcher(None, r["pred"].split(), r["target"].split()).ratio() for r in ok]

    # Graded ladder. Exact match sits at 0 long before conditioning is dead, so
    # it is useless for tracking progress; token overlap is worse than useless
    # because two arbitrary Python functions score ~0.15 on shared boilerplate.
    # Retrieval is the metric this project already validated on text (68% vs 5%).
    pts = [(_parse(r["pred"]), _parse(r["target"])) for r in ok]
    skel = [difflib.SequenceMatcher(None, skeleton(p), skeleton(t)).ratio()
            for p, t in pts if p is not None and t is not None]
    nast = sum(norm_ast(p) == norm_ast(t) for p, t in pts
               if p is not None and t is not None)
    nar = sum(arity(p) == arity(t) for p, t in pts if p is not None and t is not None)
    jac = [len(idents(r["pred"]) & idents(r["target"])) /
           max(len(idents(r["pred"]) | idents(r["target"])), 1) for r in ok]
    tgts = [r["target"] for r in ok]
    r1 = 0
    for i, r in enumerate(ok):
        sc = [difflib.SequenceMatcher(None, (r["pred"] or "").split(), t.split()).ratio()
              for t in tgts]
        if sc and int(np.argmax(sc)) == i:
            r1 += 1

    print("\n" + "=" * 62)
    print(f"n={n}  extracted={len(ok)} ({len(ok)/max(n,1):.0%})")
    print(f"exact match   : {exact}/{len(ok)}  ({exact/max(len(ok),1):.1%})")
    print(f"AST-equal     : {aeq}/{len(ok)}  ({aeq/max(len(ok),1):.1%})")
    print(f"parses        : {parse}/{len(ok)}  ({parse/max(len(ok),1):.1%})")
    print(f"token overlap : mean={np.mean(sims):.3f}  median={np.median(sims):.3f}")
    print(f"norm-AST equal: {nast}/{len(ok)}   (same code modulo renaming)")
    print(f"skeleton sim  : {np.mean(skel):.3f}  (~0.44 baseline; compare vs shuffled)")
    print(f"identifier Jac: {np.mean(jac):.3f}")
    print(f"arity match   : {nar}/{len(ok)}")
    print(f"RETRIEVAL rank-1: {r1}/{len(ok)}  ({100*r1/max(len(ok),1):.1f}%)  "
          f"chance {100/max(len(ok),1):.2f}%")
    print("=" * 62)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            f.write(json.dumps({"kind": "summary", "n": n, "extracted": len(ok),
                                "exact": exact, "ast_equal": aeq, "parses": parse,
                                "tok_sim_mean": float(np.mean(sims)) if sims else None,
                                "norm_ast": nast, "skeleton": float(np.mean(skel)) if skel else None,
                                "ident_jaccard": float(np.mean(jac)) if jac else None,
                                "arity": nar, "retrieval_rank1": r1,
                                "shuffle_control": args.shuffle_control,
                                "ckpt": args.actor_ckpt, "data": args.data}) + "\n")
            for r in rows:
                f.write(json.dumps({"kind": "sample", **r}) + "\n")
        print(f"wrote {args.out}")


def _safe_parse(s):
    try:
        ast.parse(s); return True
    except SyntaxError:
        return False


if __name__ == "__main__":
    main()
