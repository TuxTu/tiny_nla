#!/usr/bin/env python3
"""Can the actor debug code it never saw, from one activation vector?

Pipeline: buggy snippet -> base model activation at the LAST token -> inject into
actor -> explanation -> interrogate for the bug and a fix.

This is doubly out of distribution. The actor was trained on UltraFineWeb prose to
describe WHAT TOKEN COMES NEXT -- not on code, and never on debugging. A negative
result says little; a positive one says the residual stream carries more than
next-token constraint.

THREE conditions per snippet, which is what makes it interpretable:
  BUGGY   - vector from the broken version
  CORRECT - vector from the fixed version (same code otherwise)
  ZERO    - no vector at all
If BUGGY and CORRECT produce the same answer, the vector is not carrying the bug.
If ZERO produces the same answer, nothing is being read at all.

Each pair differs only near the final token, so the difference is local to where
the activation is taken.
"""
import argparse, torch

# (name, buggy, correct, what the bug is)
CASES = [
    ("off-by-one",
     "def last_element(arr):\n    # return the final item\n    return arr[len(arr)]",
     "def last_element(arr):\n    # return the final item\n    return arr[len(arr) - 1]",
     "arr[len(arr)] is out of range; should be len(arr)-1"),
    ("wrong-var",
     "def mean(xs, ys):\n    # average of xs\n    return sum(xs) / len(ys)",
     "def mean(xs, ys):\n    # average of xs\n    return sum(xs) / len(xs)",
     "divides by len(ys) instead of len(xs)"),
    ("mutable-default",
     "def append_item(item, acc=[]):\n    acc.append(item)\n    return acc",
     "def append_item(item, acc=None):\n    acc = [] if acc is None else acc\n    return acc",
     "mutable default argument shared across calls"),
    ("bsearch-bound",
     "def bsearch(a, t):\n    lo, hi = 0, len(a) - 1\n    while lo < hi:\n        mid = (lo + hi) // 2",
     "def bsearch(a, t):\n    lo, hi = 0, len(a) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2",
     "while lo < hi misses the final element; needs <="),
]
QUESTIONS = [
    ("Does the code contain a bug? Answer yes or no, then name it.",
     "Looking at the code, the answer is"),
    ("Write only the corrected line of code. No explanation, no prose.",
     "```python\n"),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--actor-ckpt", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--injection-scale", type=float, default=300.0)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from nla.training.injection import inject_at_marked_positions
    from nla.training.schema import normalize_activation
    from nla.training.sidecar import read_sidecar

    dev = "cuda"
    sc = args.sidecar[:-len(".nla_meta.yaml")] if args.sidecar.endswith(".nla_meta.yaml") else args.sidecar
    side = read_sidecar(sc); tm = side["tokens"]
    inj_id, left_id, right_id = (tm["injection_token_id"], tm["injection_left_neighbor_id"],
                                 tm["injection_right_neighbor_id"])
    inj_char = tm["injection_char"]; tpl = side["prompt_templates"]["actor"]

    tok = AutoTokenizer.from_pretrained(args.actor_ckpt)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    print(f"extracting activations @ layer {args.layer} ...")
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16,
                                                device_map={"": dev}).eval()
    vecs = {}
    for name, buggy, correct, _ in CASES:
        for cond, code in (("BUGGY", buggy), ("CORRECT", correct)):
            ids = tok(code, return_tensors="pt").input_ids.to(dev)
            with torch.no_grad():
                o = base(input_ids=ids, output_hidden_states=True)
            vecs[(name, cond)] = o.hidden_states[args.layer][0, -1].float().cpu()
    del base; torch.cuda.empty_cache()

    actor = AutoModelForCausalLM.from_pretrained(args.actor_ckpt, torch_dtype=torch.bfloat16,
                                                 device_map={"": dev}).eval()

    def gen(messages, vec, max_new=140, prefill=None):
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if prefill: prompt += prefill
        enc = tok(prompt, return_tensors="pt")
        ids, am = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
        v = vec.unsqueeze(0).to(dev)
        v = normalize_activation(v, args.injection_scale) if vec.abs().sum() > 0 else v
        def _hook(_m, a, out):
            cur = a[0]
            if cur.dim() != 2 or cur.shape[1] < 3 or not (cur == inj_id).any():
                return out
            return inject_at_marked_positions(cur, out, v.to(out.dtype), inj_id, left_id, right_id)
        h = actor.get_input_embeddings().register_forward_hook(_hook)
        try:
            with torch.no_grad():
                o = actor.generate(input_ids=ids, attention_mask=am, max_new_tokens=max_new,
                                   do_sample=False, pad_token_id=tok.pad_token_id)
        finally:
            h.remove()
        return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).strip()

    for name, buggy, correct, bugdesc in CASES:
        print("\n" + "#" * 76)
        print(f"# {name}   BUG: {bugdesc}")
        print(f"# buggy tail  : ...{buggy[-46:]!r}")
        print(f"# correct tail: ...{correct[-46:]!r}")
        print("#" * 76)
        for cond in ("BUGGY", "CORRECT"):
            vec = (torch.zeros_like(vecs[(name, "BUGGY")]) if cond == "ZERO"
                   else vecs[(name, cond)])
            msgs = [{"role": "user", "content": tpl.format(injection_char=inj_char)}]
            expl = gen(msgs, vec, max_new=170)
            print(f"\n  [{cond}] EXPLANATION: {expl[:260]}")
            msgs = msgs + [{"role": "assistant", "content": expl}]
            for q, pre in QUESTIONS:
                ans = gen(msgs + [{"role": "user", "content": q}], vec, max_new=100, prefill=pre)
                print(f"  [{cond}] Q: {q[:44]}...")
                print(f"  [{cond}] A: {pre}{ans[:260]}")

if __name__ == "__main__":
    main()
