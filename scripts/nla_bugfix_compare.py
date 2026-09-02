#!/usr/bin/env python3
"""SFT vs RL actor on the same buggy snippets -- semantic correctness, not FVE.

Tests BOTH decodings because they measure different things here: GRPO rollouts
are sampled at T=1.0, so RL reshaped the SAMPLED distribution and never
optimised greedy. Judging an RL actor greedily measures the path it never
trained on.

Sampling is stochastic, so K draws per condition; a single sample proves
nothing either way.
"""
import argparse, torch

CASES = [
    ("off-by-one",
     "def last_element(arr):\n    # return the final item\n    return arr[len(arr)]",
     "out-of-bounds index; needs len(arr)-1"),
    ("wrong-var",
     "def mean(xs, ys):\n    # average of xs\n    return sum(xs) / len(ys)",
     "divides by len(ys) not len(xs)"),
    ("mutable-default",
     "def append_item(item, acc=[]):\n    acc.append(item)\n    return acc",
     "mutable default argument"),
    ("bsearch-bound",
     "def bsearch(a, t):\n    lo, hi = 0, len(a) - 1\n    while lo < hi:\n        mid = (lo + hi) // 2",
     "while lo < hi misses the last element"),
]
Q, PREFILL = ("Does the code contain a bug? Answer yes or no, then name it.",
              "Looking at the code, the answer is")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--actor-ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--injection-scale", type=float, default=300.0)
    ap.add_argument("--k", type=int, default=3, help="sampled draws per case")
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
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16,
                                                device_map={"": dev}).eval()
    V = {}
    for name, code, _ in CASES:
        ids = tok(code, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            o = base(input_ids=ids, output_hidden_states=True)
        V[name] = o.hidden_states[args.layer][0, -1].float().cpu()
    del base; torch.cuda.empty_cache()

    actor = AutoModelForCausalLM.from_pretrained(args.actor_ckpt, torch_dtype=torch.bfloat16,
                                                 device_map={"": dev}).eval()

    def gen(messages, vec, greedy, max_new=150, prefill=None, seed=None):
        if seed is not None: torch.manual_seed(seed)
        p = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if prefill: p += prefill
        enc = tok(p, return_tensors="pt")
        ids, am = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
        v = normalize_activation(vec.unsqueeze(0).to(dev), args.injection_scale)
        def _hook(_m, a, out):
            cur = a[0]
            if cur.dim() != 2 or cur.shape[1] < 3 or not (cur == inj_id).any(): return out
            return inject_at_marked_positions(cur, out, v.to(out.dtype), inj_id, left_id, right_id)
        h = actor.get_input_embeddings().register_forward_hook(_hook)
        try:
            with torch.no_grad():
                o = actor.generate(input_ids=ids, attention_mask=am, max_new_tokens=max_new,
                                   do_sample=not greedy,
                                   temperature=None if greedy else 1.0,
                                   top_p=None if greedy else 1.0,
                                   top_k=None if greedy else 0,
                                   pad_token_id=tok.pad_token_id)
        finally: h.remove()
        return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).strip()

    print(f"\n{'='*78}\nACTOR: {args.label}\n{'='*78}")
    for name, code, bug in CASES:
        print(f"\n### {name}  (true bug: {bug})")
        for mode in ("greedy", "sampled"):
            draws = 1 if mode == "greedy" else args.k
            for k in range(draws):
                msgs = [{"role": "user", "content": tpl.format(injection_char=inj_char)}]
                expl = gen(msgs, V[name], mode == "greedy", 170, seed=1000+k)
                msgs = msgs + [{"role": "assistant", "content": expl}]
                ans = gen(msgs + [{"role": "user", "content": Q}], V[name],
                          mode == "greedy", 90, prefill=PREFILL, seed=2000+k)
                tag = mode if draws == 1 else f"{mode}[{k}]"
                print(f"  [{tag:10s}] {PREFILL}{ans[:220]}")

if __name__ == "__main__":
    main()
