#!/usr/bin/env python3
"""What does the actor actually SAY now?

Three questions the CE gap cannot answer:
  1. Does greedy decoding still collapse onto one modal explanation?
     (old actor: 25 unique / 500 -- see logs/e2e_v2_17359072.out)
  2. Is a generated explanation related to the GOLD explanation for ITS OWN
     vector, or is it generic text that would fit any vector equally well?
  3. What do they look like side by side?

Q2 is the one that matters and it needs a control. Raw overlap-with-gold is
uninterpretable on its own: two explanations of ANY activation vectors share
"the vector", "text", "activations" etc. So we score each generation against
its own gold AND against other rows' golds, and compare. Only the DIFFERENCE
is evidence of vector-specific content.
"""
import argparse, json, math, random, re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJ = Path(__file__).resolve().parent.parent

_WORD = re.compile(r"[a-z0-9']+")
# Function words dominate token overlap between any two English sentences and
# would flatten the own-vs-other contrast we are trying to measure.
_STOP = set("""a an the and or but if then than that this these those of in on at to for from by with
without about into over under is are was were be been being it its as not no so such can may might will
would should could have has had do does did i you he she they we them their his her our your my me us
which who whom whose what when where why how all any both each few more most other some only own same
too very s t just don now here there""".split())


def toks(s: str) -> Counter:
    return Counter(w for w in _WORD.findall(s.lower()) if w not in _STOP and len(w) > 2)


def f1(a: Counter, b: Counter) -> float:
    """Token-overlap F1 between two bags of content words."""
    if not a or not b:
        return 0.0
    overlap = sum((a & b).values())
    if overlap == 0:
        return 0.0
    p, r = overlap / sum(a.values()), overlap / sum(b.values())
    return 2 * p * r / (p + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor-ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--injection-scale", type=float, required=True)
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--n-show", type=int, default=8)
    ap.add_argument("--jsonl", type=str, default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from nla.training.injection import inject_at_marked_positions
    from nla.training.schema import normalize_activation, INJECT_PLACEHOLDER, extract_explanation
    from nla.training.sidecar import read_sidecar

    dev = "cuda"
    sc = args.sidecar[:-len(".nla_meta.yaml")] if args.sidecar.endswith(".nla_meta.yaml") else args.sidecar
    side = read_sidecar(sc)
    tm = side["tokens"]
    inj_id, left_id, right_id = (tm["injection_token_id"],
                                 tm["injection_left_neighbor_id"],
                                 tm["injection_right_neighbor_id"])
    inj_char = tm["injection_char"]

    tok = AutoTokenizer.from_pretrained(args.actor_ckpt)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    # Decoder-only batched generation requires LEFT padding, otherwise the
    # continuation starts after a run of pads. inject_at_marked_positions finds
    # the site by token id, so the shifted position is harmless.
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.actor_ckpt, torch_dtype=torch.bfloat16, device_map={"": dev})
    model.eval()

    df = pd.read_parquet(args.data)
    if len(df) > args.n_samples:
        df = df.sample(n=args.n_samples, random_state=42).reset_index(drop=True)
    n = len(df)
    vecs = torch.tensor(np.stack(df["activation_vector"].values), dtype=torch.float32)
    prompts = [[{"role": m["role"], "content": m["content"].replace(INJECT_PLACEHOLDER, inj_char)}
                for m in msgs] for msgs in df["prompt"].tolist()]
    golds = [extract_explanation(r) or r for r in df["response"].astype(str).tolist()]

    mode = "greedy" if args.greedy else f"sampled T={args.temperature}"
    print(f"{n} rows  scale={args.injection_scale}  decoding={mode}")

    gens, n_fail = [], 0
    for i in range(0, n, args.batch_size):
        sl = slice(i, min(i + args.batch_size, n))
        texts = [tok.apply_chat_template(list(m), tokenize=False, add_generation_prompt=True)
                 for m in prompts[sl]]
        enc = tok(texts, padding=True, return_tensors="pt")
        ids, am = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
        v = normalize_activation(vecs[sl].to(dev), args.injection_scale)

        embed = model.get_input_embeddings()

        def _hook(_m, a, out):
            # generate() runs prefill with the full prompt, then feeds ONE token
            # per decode step (KV cache). Inject only on the pass that actually
            # contains the marker; closing over the prompt ids would blow up on
            # every decode step with a [B,1] mismatch.
            cur = a[0]
            if cur.dim() != 2 or cur.shape[1] < 3 or not (cur == inj_id).any():
                return out
            return inject_at_marked_positions(cur, out, v.to(out.dtype),
                                              inj_id, left_id, right_id)

        h = embed.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                out = model.generate(
                    input_ids=ids, attention_mask=am,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=not args.greedy,
                    temperature=None if args.greedy else args.temperature,
                    top_p=None if args.greedy else 0.95,
                    pad_token_id=tok.pad_token_id)
        finally:
            h.remove()
        for row in out[:, ids.shape[1]:]:
            full = tok.decode(row, skip_special_tokens=True)
            e = extract_explanation(full)
            if e is None:
                n_fail += 1
                e = full.strip()
            gens.append(e)
        print(f"  {min(i+args.batch_size, n)}/{n}", flush=True)

    # ---- Q1: collapse ------------------------------------------------------
    uniq = len(set(g.strip() for g in gens))
    print("\n" + "=" * 60)
    print(f"UNIQUE explanations : {uniq}/{n}  ({100*uniq/n:.1f}%)")
    print(f"extraction failures : {n_fail}/{n}  ({100*n_fail/n:.1f}%)")
    top = Counter(g.strip() for g in gens).most_common(3)
    print(f"most common output  : {top[0][1]}x  ({100*top[0][1]/n:.1f}% of rows)")

    # ---- Q2: related to ITS OWN gold, vs a control -------------------------
    gt, ct = [toks(g) for g in gens], [toks(g) for g in golds]
    rng = random.Random(0)
    own = [f1(gt[i], ct[i]) for i in range(n)]
    # control: same generation vs 5 OTHER rows' golds
    oth = [np.mean([f1(gt[i], ct[j]) for j in rng.sample([k for k in range(n) if k != i], k=min(5, n-1))])
           for i in range(n)]
    own_m, oth_m = float(np.mean(own)), float(np.mean(oth))
    # retrieval: does its own gold rank #1 against 19 distractors?
    hits = 0
    for i in range(n):
        pool = rng.sample([k for k in range(n) if k != i], k=min(19, n - 1))
        if f1(gt[i], ct[i]) > max(f1(gt[i], ct[j]) for j in pool):
            hits += 1
    print("-" * 60)
    print(f"content-word F1 vs OWN gold    : {own_m:.4f}")
    print(f"content-word F1 vs OTHER golds : {oth_m:.4f}   (control)")
    print(f"lift                           : {own_m - oth_m:+.4f}  ({own_m/max(oth_m,1e-9):.2f}x)")
    print(f"retrieval acc (own gold beats 19 distractors): {100*hits/n:.1f}%   (chance = 5.0%)")
    print("=" * 60)

    # ---- Q3: eyeball -------------------------------------------------------
    for i in range(min(args.n_show, n)):
        print(f"\n--- row {i}  (F1 vs own gold {own[i]:.3f}) ---")
        print(f"GOLD: {golds[i][:340]}")
        print(f"GEN : {gens[i][:340]}")

    if args.jsonl:
        with open(args.jsonl, "w") as f:
            for i in range(n):
                f.write(json.dumps({"idx": i, "gold": golds[i], "gen": gens[i],
                                    "f1_own": own[i], "f1_other": float(oth[i])}) + "\n")
        print(f"\nper-sample → {args.jsonl}")


if __name__ == "__main__":
    main()
