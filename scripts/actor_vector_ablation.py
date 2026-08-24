#!/usr/bin/env python3
"""Does the actor actually read the injected vector?

Teacher-forced CE on the SAME held-out rows under three injection conditions:
  correct  — the row's own activation vector
  shuffled — another row's vector (same distribution, wrong pairing)
  zeros    — no information at all

If correct ~= shuffled, the actor has learned p(explanation), not
p(explanation | vector), and free-running generation will collapse onto the
modal explanation no matter how good the teacher-forced CE looks.
"""
import argparse, math
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

PROJ = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor-ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--injection-scale", type=str, default=None,
                    help="L2 norm used at TRAINING time for this checkpoint.")
    ap.add_argument("--n-samples", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()

    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from nla.training.injection import inject_at_marked_positions
    from nla.training.schema import normalize_activation, INJECT_PLACEHOLDER
    from nla.training.sidecar import read_sidecar

    dev = "cuda"
    sc = args.sidecar[:-len(".nla_meta.yaml")] if args.sidecar.endswith(".nla_meta.yaml") else args.sidecar
    side = read_sidecar(sc)
    tok_meta = side["tokens"]
    d_model = side["extraction"]["d_model"]
    inj_id = tok_meta["injection_token_id"]
    left_id = tok_meta["injection_left_neighbor_id"]
    right_id = tok_meta["injection_right_neighbor_id"]
    # MUST match what the checkpoint was trained at -- evaluating a scale-300
    # actor with vectors injected at 160 is a train/eval mismatch and the CE
    # gap it reports is meaningless.
    if args.injection_scale is None:
        raise SystemExit("--injection-scale is required (must match training)")
    scale = float(args.injection_scale)

    tokenizer = AutoTokenizer.from_pretrained(args.actor_ckpt)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.actor_ckpt, torch_dtype=torch.bfloat16, device_map={"": dev})
    model.eval()

    df = pd.read_parquet(args.data)
    if len(df) > args.n_samples:
        df = df.sample(n=args.n_samples, random_state=42)
    vecs = torch.tensor(np.stack(df["activation_vector"].values), dtype=torch.float32)
    # The parquet stores the <INJECT> placeholder; ActorDataset swaps it for the
    # real injection char at load time. Skip this and there is no injection site.
    inj_char = tok_meta["injection_char"]
    prompts = [
        [{"role": m["role"], "content": m["content"].replace(INJECT_PLACEHOLDER, inj_char)}
         for m in msgs]
        for msgs in df["prompt"].tolist()
    ]
    responses = df["response"].astype(str).tolist()
    n = len(df)
    print(f"{n} rows  d_model={d_model}  injection_scale={scale:.1f}")

    # a fixed derangement so no row keeps its own vector
    perm = (np.arange(n) + 1) % n

    def ce(mode: str) -> float:
        tot, nb = 0.0, 0
        for i in range(0, n, args.batch_size):
            sl = slice(i, min(i + args.batch_size, n))
            msgs = prompts[sl]
            resp = responses[sl]
            texts, ponly = [], []
            for m, r in zip(msgs, resp):
                p = tokenizer.apply_chat_template(list(m), tokenize=False,
                                                  add_generation_prompt=True)
                ponly.append(p)
                texts.append(p + r)
            pe = tokenizer(ponly, padding=True, truncation=True,
                           max_length=args.max_length, return_tensors="pt")
            fe = tokenizer(texts, padding=True, truncation=True,
                           max_length=args.max_length, return_tensors="pt")
            ids = fe["input_ids"].to(dev)
            am = fe["attention_mask"].to(dev)
            labels = ids.clone()
            plens = pe["attention_mask"].sum(dim=1)
            for b in range(len(plens)):
                labels[b, :plens[b]] = -100
            labels[am == 0] = -100

            if mode == "correct":
                v = vecs[sl]
            elif mode == "shuffled":
                v = vecs[perm[np.arange(n)[sl]]]
            else:
                v = torch.zeros(ids.shape[0], d_model)
            v = normalize_activation(v.to(dev), scale) if mode != "zeros" else v.to(dev)

            embed = model.get_input_embeddings()
            def _hook(_m, _a, out):
                return inject_at_marked_positions(
                    ids, out, v.to(out.dtype), inj_id, left_id, right_id)
            h = embed.register_forward_hook(_hook)
            try:
                with torch.no_grad():
                    logits = model(input_ids=ids, attention_mask=am).logits
            finally:
                h.remove()
            sl_lg = logits[:, :-1].float()
            sl_lb = labels[:, 1:]
            loss = F.cross_entropy(sl_lg.reshape(-1, sl_lg.size(-1)),
                                   sl_lb.reshape(-1), ignore_index=-100)
            tot += loss.item(); nb += 1
        return tot / nb

    res = {m: ce(m) for m in ("correct", "shuffled", "zeros")}
    print("\n" + "=" * 56)
    for m, v in res.items():
        print(f"  {m:9s} CE = {v:.4f}   (ppl {math.exp(min(v,20)):.2f})")
    gap = res["shuffled"] - res["correct"]
    print("-" * 56)
    print(f"  shuffled - correct = {gap:+.4f} nats/token")
    print(f"  zeros    - correct = {res['zeros']-res['correct']:+.4f} nats/token")
    print("=" * 56)
    if gap < 0.05:
        print("VERDICT: actor is IGNORING the vector (gap < 0.05).")
    elif gap < 0.25:
        print("VERDICT: actor uses the vector WEAKLY.")
    else:
        print("VERDICT: actor genuinely conditions on the vector.")


if __name__ == "__main__":
    main()
