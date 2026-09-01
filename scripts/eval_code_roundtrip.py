#!/usr/bin/env python3
"""Round-trip metric for code reconstruction: v -> AV -> code' -> E(code') -> v'.

This is the direct analogue of the text pipeline's FVE (vector -> explanation ->
critic -> vector). For code no critic is needed: the same frozen encoder maps
generated code straight back into the vector space.

Strictly better than the lexical difflib similarity used in eval_code_recon.py,
because two arbitrary Python functions share ~0.15 token overlap on boilerplate
alone (def/return/if/self), which inflates every lexical number and forces the
shuffled control to carry all the interpretive weight. Cosine in activation
space has no such floor.

Encoding uses the FROZEN BASE model at layer 24 -- the same E that produced the
targets. Using the finetuned actor as encoder would compare against a different
space.

  python scripts/eval_code_roundtrip.py --gen logs/code_exp1_gen.jsonl \
      --gen-shuf logs/code_exp1_gen_shuf.jsonl
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

PROJ = Path(__file__).resolve().parent.parent


@torch.no_grad()
def encode(texts, model, tok, layer, device, bs=16, max_len=1024):
    cap = {}
    h = model.model.layers[layer].register_forward_hook(
        lambda _m, _a, out: cap.__setitem__("h", (out[0] if isinstance(out, tuple) else out).detach()))
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    try:
        for i in range(0, len(texts), bs):
            ch = [t if t and t.strip() else "pass" for t in texts[i:i + bs]]
            enc = tok(ch, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
            ids, m = enc["input_ids"].to(device), enc["attention_mask"].to(device)
            model(input_ids=ids, attention_mask=m, use_cache=False)
            out[i:i + len(ch)] = cap["h"][torch.arange(len(ch)), m.sum(1) - 1].float().cpu().numpy()
    finally:
        h.remove()
    return out


def report(name, P, T):
    Pn = P / np.linalg.norm(P, axis=1, keepdims=True).clip(1e-9)
    Tn = T / np.linalg.norm(T, axis=1, keepdims=True).clip(1e-9)
    cos = (Pn * Tn).sum(1)
    S = Pn @ Tn.T                       # [n_pred, n_target]
    r1 = (S.argmax(1) == np.arange(len(S))).mean()
    rank = (S > S[np.arange(len(S)), np.arange(len(S))][:, None]).sum(1) + 1
    # FVE on L2-NORMALISED vectors, matching nla_eval_fve.py. On raw vectors a
    # single generation with an outsized norm dominates the MSE -- the first run
    # reported FVE=-2303 for the shuffled control for exactly that reason.
    scale = float(np.sqrt(P.shape[1]))
    Ps, Ts = Pn * scale, Tn * scale
    mu = Ts.mean(0, keepdims=True)
    fve = 1.0 - ((Ps - Ts) ** 2).mean() / ((Ts - mu) ** 2).mean()
    print(f"  {name:<10} cos={cos.mean():.4f}  FVE={fve:+.4f}  "
          f"retrieval@1={r1:.1%} (chance {1/len(S):.2%})  median rank={np.median(rank):.0f}/{len(S)}")
    return dict(cos=float(cos.mean()), fve=float(fve), r1=float(r1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen", required=True, help="generations jsonl (correct vectors)")
    p.add_argument("--gen-shuf", default=None, help="generations jsonl (shuffled control)")
    p.add_argument("--model-name", default="Qwen/Qwen3-8B")
    p.add_argument("--layer-index", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    def load(f):
        return [json.loads(l) for l in open(f) if json.loads(l)["kind"] == "sample"]

    cor = [r for r in load(args.gen) if r["pred"] and r["target"]]
    print(f"{len(cor)} generations with correct vectors")

    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map={"": args.device}).eval()
    layer = args.layer_index if args.layer_index is not None else (2 * len(model.model.layers)) // 3
    print(f"encoding with frozen {args.model_name} at layer {layer}\n")

    T = encode([r["target"] for r in cor], model, tok, layer, args.device)
    res = {"correct": report("correct", encode([r["pred"] for r in cor], model, tok, layer, args.device), T)}

    if args.gen_shuf:
        sm = {r["idx"]: r for r in load(args.gen_shuf)}
        preds = [sm[r["idx"]]["pred"] if r["idx"] in sm and sm[r["idx"]]["pred"] else "" for r in cor]
        res["shuffled"] = report("shuffled", encode(preds, model, tok, layer, args.device), T)
        d = res["correct"]["cos"] - res["shuffled"]["cos"]
        print(f"\n  cosine lift over shuffled control: {d:+.4f}")

    # Ceiling: identical code round-trips to cos=1.0 by construction, so the
    # meaningful reference is the shuffled floor, not 1.0.
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
