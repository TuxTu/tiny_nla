#!/usr/bin/env python3
"""Does the AV actor work on non-English activations?

Training was UltraFineWeb split:en -- the actor has only ever seen English
activations paired with English explanations. Chinese is out-of-distribution on
both sides, so this asks two separable questions:
  1. Does the actor still produce coherent, on-topic output at all?
  2. Does it describe the ACTUAL Chinese content, or hallucinate generic text?

Paired design: each Chinese text has an English near-translation. Same meaning,
different tokens/activations. If the explanations describe the same content, the
vector encodes meaning rather than surface form.
"""
import argparse, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

PAIRS = [
    ("zh-hist", "中国的首都是北京，它有着三千多年的建城历史，是明清两代的皇都。"),
    ("en-hist", "The capital of China is Beijing, which has over three thousand years of history as a city and served as the imperial capital of the Ming and Qing dynasties."),
    ("zh-sci",  "光合作用是植物利用阳光将二氧化碳和水转化为葡萄糖和氧气的过程。"),
    ("en-sci",  "Photosynthesis is the process by which plants use sunlight to convert carbon dioxide and water into glucose and oxygen."),
    ("zh-code", "def 计算平均值(数列):\n    return sum(数列) / len(数列)\n\n结果 = 计算平均值([1, 2, 3"),
    ("zh-news", "据新华社报道，国务院昨日发布了关于进一步深化医疗改革的指导意见，要求各省在年底前"),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--actor-ckpt", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--injection-scale", type=float, default=300.0)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    args = ap.parse_args()

    from nla.training.injection import inject_at_marked_positions
    from nla.training.schema import normalize_activation, INJECT_PLACEHOLDER, extract_explanation
    from nla.training.sidecar import read_sidecar

    dev = "cuda"
    sc = args.sidecar[:-len(".nla_meta.yaml")] if args.sidecar.endswith(".nla_meta.yaml") else args.sidecar
    side = read_sidecar(sc)
    tm = side["tokens"]
    inj_id, left_id, right_id = (tm["injection_token_id"], tm["injection_left_neighbor_id"],
                                 tm["injection_right_neighbor_id"])
    inj_char = tm["injection_char"]
    tpl = side["prompt_templates"]["actor"] if "actor" in side.get("prompt_templates", {}) else None

    tok = AutoTokenizer.from_pretrained(args.actor_ckpt)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # ---- 1. extract activation vectors from the BASE model at the last token ----
    print(f"extracting activations from {args.base_model} @ layer {args.layer} ...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map={"": dev})
    base.eval()
    vecs, metas = [], []
    for tag, text in PAIRS:
        ids = tok(text, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            out = base(input_ids=ids, output_hidden_states=True)
        v = out.hidden_states[args.layer][0, -1].float().cpu()   # last token
        vecs.append(v); metas.append((tag, text, ids.shape[1], v.norm().item()))
        print(f"  {tag:8s} {ids.shape[1]:3d} tok  ‖v‖={v.norm().item():7.2f}  {text[:40]}...")
    del base; torch.cuda.empty_cache()

    # ---- 2. run them through the actor -----------------------------------------
    print(f"\nloading actor {args.actor_ckpt} ...")
    actor = AutoModelForCausalLM.from_pretrained(
        args.actor_ckpt, torch_dtype=torch.bfloat16, device_map={"": dev})
    actor.eval()

    for (tag, text, ntok, nrm), v in zip(metas, vecs):
        # The sidecar template carries a PYTHON FORMAT FIELD {injection_char}
        # (see build_training_data.py:35,140) -- not the <INJECT> placeholder.
        # .replace(INJECT_PLACEHOLDER, ...) is a no-op on it, which leaves no
        # marker token, so the hook never fires and the actor just describes the
        # literal prompt -- identical output for every input.
        assert tpl and "{injection_char}" in tpl, "unexpected actor template"
        msgs = [{"role": "user", "content": tpl.format(injection_char=inj_char)}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt")
        ids, am = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
        vv = normalize_activation(v.unsqueeze(0).to(dev), args.injection_scale)

        embed = actor.get_input_embeddings()
        def _hook(_m, a, out):
            cur = a[0]
            if cur.dim() != 2 or cur.shape[1] < 3 or not (cur == inj_id).any():
                return out
            return inject_at_marked_positions(cur, out, vv.to(out.dtype), inj_id, left_id, right_id)
        h = embed.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                o = actor.generate(input_ids=ids, attention_mask=am,
                                   max_new_tokens=args.max_new_tokens,
                                   do_sample=False, pad_token_id=tok.pad_token_id)
        finally:
            h.remove()
        full = tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)
        expl = extract_explanation(full) or f"[NO TAG] {full.strip()}"
        print("\n" + "=" * 72)
        print(f"[{tag}]  {ntok} tok  ‖v‖={nrm:.1f}")
        print(f"INPUT : {text[:120]}")
        print(f"OUTPUT: {expl[:700]}")

if __name__ == "__main__":
    main()
