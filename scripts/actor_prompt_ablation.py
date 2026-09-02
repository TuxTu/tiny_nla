#!/usr/bin/env python3
"""Does the trained actor still work when the prompt preamble is stripped?

The actor was SFT'd with a ~90-token "meticulous AI researcher" preamble that is
IDENTICAL on every training example.  A constant prefix carries no information
that discriminates between examples, so post-SFT it can only be acting as a
fixed mode-selector.  This sweeps a ladder of shorter prompts on the ALREADY
TRAINED actor and reports how much end-to-end FVE survives.

What this measures and what it does NOT: the actor never saw these prompts, so a
drop confounds "the preamble was load-bearing" with "the actor is brittle to
prompt distribution shift".  Only a retrain separates those.  But if FVE holds
up under stripping, the preamble is inert and the retrain is worth doing.

Scoring mirrors nla_eval_fve.py exactly (subset-matched baselines included).

  python scripts/actor_prompt_ablation.py \
      --model-name Qwen/Qwen3-8B \
      --actor-ckpt checkpoints/actor_sft_8b_s50fix \
      --critic-ckpt checkpoints/critic_sft_8b_s50 \
      --data data/av_eval_8b_ufw_joined.parquet \
      --n-samples 200 --greedy
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from scripts.nla_eval_fve import compute_fve_baselines, _normalize  # noqa: E402


# ---------------------------------------------------------------------------
# Prompt ladder.  {inj} is replaced by the injection char.
#
# chat=True  -> wrapped in the Qwen chat template (as at training time)
# prefill    -> appended AFTER the assistant header, so the model continues
#               from inside it rather than having to open the tag itself
#
# Variants that keep "<concept>{inj}</concept>" verbatim preserve the canonical
# left/right neighbour tokens and need no injection changes.  The bare ones do
# not, so they run through inject_relaxed().
# ---------------------------------------------------------------------------
_FULL = (
    "You are a meticulous AI researcher conducting an important investigation "
    "into activation vectors from a language model. Your overall task is to describe "
    "the semantic content of that activation vector.\n\n"
    "We will pass the vector enclosed in <concept> tags into your context. You must "
    "then produce an explanation for the vector, enclosed within <explanation> tags. "
    "The explanation consists of 2-3 text snippets describing that vector.\n\n"
    "Here is the vector:\n\n"
    "<concept>{inj}</concept>\n\n"
    "Please provide an explanation."
)

VARIANTS = [
    # name              text                                              chat  prefill        strict
    ("full",            _FULL,                                            True, "",            True),
    ("no_persona",      "Describe the semantic content of this activation "
                        "vector, enclosed within <explanation> tags.\n\n"
                        "<concept>{inj}</concept>",                       True, "",            True),
    ("terse",           "Here is the vector:\n\n<concept>{inj}</concept>"
                        "\n\nPlease provide an explanation.",             True, "",            True),
    ("tags_only",       "<concept>{inj}</concept>",                       True, "",            True),
    ("tags_prefill",    "<concept>{inj}</concept>",                       True, "<explanation>", True),
    ("bare_chat",       "{inj}",                                          True, "<explanation>", False),
    ("bare_raw",        "{inj}",                                          False, "<explanation>", False),
]


def inject_relaxed(input_ids, embeddings, vectors, inj_id):
    """Overwrite every injection-marker row. No neighbour check, no bounds check.

    inject_at_marked_positions() requires canonical left/right neighbours AND
    skips p==0 / p==seq_len-1, so it cannot serve the bare-marker variants
    (where the marker may be the entire sequence).
    """
    out = embeddings.clone()
    vectors = vectors.to(out.device, out.dtype)
    matches = (input_ids == inj_id).nonzero(as_tuple=False)
    if matches.shape[0] != vectors.shape[0]:
        raise RuntimeError(
            f"found {matches.shape[0]} markers, expected {vectors.shape[0]} "
            "(one per row) — check padding or prompt template"
        )
    for k, (b, p) in enumerate(matches.tolist()):
        out[b, p] = vectors[k]
    return out


def build_prompt(text, inj_char, chat, prefill, tokenizer):
    body = text.format(inj=inj_char)
    if chat:
        s = tokenizer.apply_chat_template(
            [{"role": "user", "content": body}],
            tokenize=False, add_generation_prompt=True,
        )
    else:
        s = body
    return s + prefill


@torch.no_grad()
def run_variant(name, text, chat, prefill, strict, *, vecs, gold_vecs, actor, critic,
                tokenizer, tokens, d_model, mse_scale, max_new_tokens,
                batch_size, greedy, device):
    from nla.training.injection import inject_at_marked_positions
    from nla.training.schema import extract_explanation

    inj_id = tokens["injection_token_id"]
    left_id = tokens["injection_left_neighbor_id"]
    right_id = tokens["injection_right_neighbor_id"]
    inj_char = tokens["injection_char"]

    prompt_str = build_prompt(text, inj_char, chat, prefill, tokenizer)
    n_prompt_tok = len(tokenizer(prompt_str, add_special_tokens=False)["input_ids"])

    embed_layer = actor.get_input_embeddings()
    tokenizer.padding_side = "left"

    rows, preds = [], []
    for start in range(0, len(vecs), batch_size):
        batch = vecs[start:start + batch_size]
        enc = tokenizer([prompt_str] * len(batch), return_tensors="pt",
                        padding=True, add_special_tokens=not chat)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        bvec = torch.stack([v for v in batch]).to(device)

        def _hook(_module, args, output):
            actual = args[0]
            if output.shape[1] <= 1:            # decode step: KV-cached, skip
                return output
            if strict:
                return inject_at_marked_positions(
                    actual, output, bvec, inj_id, left_id, right_id)
            return inject_relaxed(actual, output, bvec, inj_id)

        h = embed_layer.register_forward_hook(_hook)
        try:
            out = actor.generate(
                ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                do_sample=not greedy,
                temperature=None if greedy else 1.0,
                top_p=None if greedy else 1.0,
                top_k=None if greedy else 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        finally:
            h.remove()

        for j in range(len(batch)):
            gen = tokenizer.decode(out[j][ids.shape[1]:], skip_special_tokens=True)
            # The opening tag lives in the prompt for prefill variants, so the
            # continuation alone would never match the extraction regex.
            expl = extract_explanation(prefill + gen)
            rows.append({"idx": start + j, "gen": gen, "expl": expl,
                         "gen_tok": len(tokenizer(gen, add_special_tokens=False)["input_ids"])})

    # ---- critic: explanation -> vector ------------------------------------
    ok = [r for r in rows if r["expl"]]
    for i in range(0, len(ok), batch_size):
        chunk = ok[i:i + batch_size]
        cp = [f"Summary of the following text: <text>{r['expl']}</text> <summary>"
              for r in chunk]
        tokenizer.padding_side = "right"
        ce = tokenizer(cp, return_tensors="pt", padding=True,
                       truncation=True, max_length=512)
        cid, cm = ce["input_ids"].to(device), ce["attention_mask"].to(device)
        co = critic(input_ids=cid, attention_mask=cm)
        last = cm.sum(dim=1) - 1
        preds.append(co.values[torch.arange(len(chunk)), last].float().cpu())

    n_ok = len(ok)
    metrics = {
        "variant": name, "prompt_tokens": n_prompt_tok, "n": len(rows),
        "parsed": n_ok, "parse_rate": n_ok / max(len(rows), 1),
    }
    if n_ok == 0:
        metrics.update(fve_nrm=float("nan"), fve_meannorm=float("nan"),
                       gen_tok=float("nan"), uniq=float("nan"), nonascii=float("nan"))
        return metrics, rows

    pred_t = torch.cat(preds)
    # MUST index gold_vecs, not vecs: under --shuffle-control those differ, and
    # scoring the injected vector against itself just relabels the same pairs,
    # giving MSE identical to the unshuffled run.
    gold_sub = torch.stack([gold_vecs[r["idx"]] for r in ok]).float()
    mse = ((_normalize(pred_t, mse_scale) - _normalize(gold_sub, mse_scale)) ** 2).mean().item()
    # Subset-matched: variants fail on different rows, so a shared baseline
    # would make their FVEs incomparable.
    b_mean, b_raw = compute_fve_baselines(gold_sub, mse_scale)

    texts = [r["expl"] for r in ok]
    metrics.update(
        fve_nrm=1.0 - mse / b_raw,
        fve_meannorm=1.0 - mse / b_mean,
        gen_tok=float(np.mean([r["gen_tok"] for r in rows])),
        uniq=len(set(texts)) / len(texts),
        nonascii=float(np.mean([sum(ord(c) > 127 for c in t) / max(len(t), 1) for t in texts])),
    )
    return metrics, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen3-8B")
    p.add_argument("--actor-ckpt", default=str(PROJ / "checkpoints/actor_sft_8b_s50fix"))
    p.add_argument("--critic-ckpt", default=str(PROJ / "checkpoints/critic_sft_8b_s50"))
    p.add_argument("--data", default=str(PROJ / "data/av_eval_8b_ufw_joined.parquet"))
    p.add_argument("--sidecar", default=None, help="defaults to <data>.nla_meta.yaml")
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--shuffle-control", action="store_true",
                   help="permute vectors before injection; FVE should collapse")
    p.add_argument("--only", help="comma-separated variant names to run")
    p.add_argument("--jsonl", default=str(PROJ / "logs/actor_prompt_ablation.jsonl"))
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    import pyarrow.parquet as pq
    from scripts.nla_eval_fve import load_models
    from nla.training.schema import normalize_activation

    sidecar = args.sidecar or (args.data + ".nla_meta.yaml")
    actor, critic, tokenizer, tokens, d_model, mse_scale = load_models(
        args.model_name, args.actor_ckpt, args.critic_ckpt, sidecar, args.device)
    mse_scale = float(mse_scale)

    tbl = pq.read_table(args.data, columns=["activation_vector"])
    raw = np.stack(tbl.column("activation_vector").to_pylist()[: args.n_samples])
    gold = torch.from_numpy(raw.astype(np.float32))
    inj_scale = 2.5 * math.sqrt(d_model)
    vecs = [normalize_activation(gold[i:i + 1], inj_scale).squeeze(0) for i in range(len(gold))]

    if args.shuffle_control:
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(len(vecs), generator=g)
        inj_vecs = [vecs[i] for i in perm.tolist()]
        print("SHUFFLE CONTROL: injecting permuted vectors, scoring against true vectors")
    else:
        inj_vecs = vecs

    wanted = set(args.only.split(",")) if args.only else None
    print(f"\nactor={args.actor_ckpt}\ncritic={args.critic_ckpt}\n"
          f"n={len(vecs)}  greedy={args.greedy}  d_model={d_model}  mse_scale={mse_scale:.2f}\n")

    all_metrics, dump = [], []
    for name, text, chat, prefill, strict in VARIANTS:
        if wanted and name not in wanted:
            continue
        print(f"── {name} ...", flush=True)
        # Injected vectors may be permuted; gold for scoring is always true order.
        m, rows = run_variant(
            name, text, chat, prefill, strict,
            vecs=inj_vecs, gold_vecs=vecs, actor=actor, critic=critic, tokenizer=tokenizer,
            tokens=tokens, d_model=d_model, mse_scale=mse_scale,
            max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
            greedy=args.greedy, device=args.device)
        if args.shuffle_control:
            m["variant"] = name + "/shuf"
        all_metrics.append(m)
        for r in rows[:5]:
            dump.append({"variant": name, **r})
        print(f"   prompt={m['prompt_tokens']:>3}tok  parsed={m['parse_rate']:.0%}  "
              f"FVE_nrm={m['fve_nrm']:.4f}  FVE_mn={m['fve_meannorm']:.4f}")

    print("\n" + "=" * 88)
    print(f"{'variant':<16}{'prompt':>7}{'parsed':>8}{'FVE_nrm':>10}{'FVE_mn':>10}"
          f"{'gen_tok':>9}{'uniq':>7}{'non-ascii':>11}")
    print("-" * 88)
    for m in all_metrics:
        print(f"{m['variant']:<16}{m['prompt_tokens']:>7}{m['parse_rate']:>7.0%}"
              f"{m['fve_nrm']:>10.4f}{m['fve_meannorm']:>10.4f}"
              f"{m['gen_tok']:>9.1f}{m['uniq']:>7.2f}{m['nonascii']:>11.3f}")
    print("=" * 88)
    print("FVE baselines are subset-matched per variant: a variant that parses only")
    print("half its samples is NOT directly comparable on FVE alone — read parsed too.")

    Path(args.jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.jsonl, "w") as f:
        for m in all_metrics:
            f.write(json.dumps({"kind": "metrics", **m}) + "\n")
        for d in dump:
            f.write(json.dumps({"kind": "sample", **d}) + "\n")
    print(f"\nwrote {args.jsonl}")


if __name__ == "__main__":
    main()
