#!/usr/bin/env python3
"""Ask ONLY for corrected code -- no prose -- and diff it against the original.

Restricted to off-by-one, the single case the actor identified correctly. The
question is whether "it knows the bug" survives being asked for the artefact
rather than a description: a correct diagnosis in prose is much cheaper than
emitting the right line.

BUGGY / CORRECT / ZERO as before. If the BUGGY condition emits the fix and the
CORRECT condition emits something else, the vector carries the defect.
"""
import argparse, torch

ORIGINAL_BUGGY   = "def last_element(arr):\n    # return the final item\n    return arr[len(arr)]"
ORIGINAL_CORRECT = "def last_element(arr):\n    # return the final item\n    return arr[len(arr) - 1]"
TRUE_FIX         = "return arr[len(arr) - 1]"

ASKS = [
    ("Write only the corrected line of code. No explanation, no prose.",
     "```python\n"),
    ("Rewrite the whole function correctly. Output only code.",
     "```python\ndef "),
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
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16,
                                                device_map={"": dev}).eval()
    V = {}
    for cond, code in (("BUGGY", ORIGINAL_BUGGY), ("CORRECT", ORIGINAL_CORRECT)):
        ids = tok(code, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            o = base(input_ids=ids, output_hidden_states=True)
        V[cond] = o.hidden_states[args.layer][0, -1].float().cpu()
    del base; torch.cuda.empty_cache()

    actor = AutoModelForCausalLM.from_pretrained(args.actor_ckpt, torch_dtype=torch.bfloat16,
                                                 device_map={"": dev}).eval()

    def gen(messages, vec, max_new=120, prefill=None):
        p = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if prefill: p += prefill
        enc = tok(p, return_tensors="pt")
        ids, am = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
        v = vec.unsqueeze(0).to(dev)
        v = normalize_activation(v, args.injection_scale) if vec.abs().sum() > 0 else v
        def _hook(_m, a, out):
            cur = a[0]
            if cur.dim() != 2 or cur.shape[1] < 3 or not (cur == inj_id).any(): return out
            return inject_at_marked_positions(cur, out, v.to(out.dtype), inj_id, left_id, right_id)
        h = actor.get_input_embeddings().register_forward_hook(_hook)
        try:
            with torch.no_grad():
                o = actor.generate(input_ids=ids, attention_mask=am, max_new_tokens=max_new,
                                   do_sample=False, pad_token_id=tok.pad_token_id)
        finally: h.remove()
        return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)

    print("=" * 78)
    print("ORIGINAL (buggy)  :", repr(ORIGINAL_BUGGY))
    print("GROUND-TRUTH FIX  :", repr(TRUE_FIX))
    print("=" * 78)

    for cond in ("BUGGY", "CORRECT", "ZERO"):
        vec = torch.zeros_like(V["BUGGY"]) if cond == "ZERO" else V[cond]
        msgs = [{"role": "user", "content": tpl.format(injection_char=inj_char)}]
        expl = gen(msgs, vec, max_new=170)
        msgs = msgs + [{"role": "assistant", "content": expl}]
        print(f"\n{'#'*32} {cond} {'#'*32}")
        for q, pre in ASKS:
            out = gen(msgs + [{"role": "user", "content": q}], vec, max_new=110, prefill=pre)
            body = (pre + out).split("```")[1] if "```" in (pre + out) else (pre + out)
            print(f"\n  ASK: {q}")
            print("  --- model output ---")
            for line in body.strip().splitlines()[:8]:
                print("   ", line)

if __name__ == "__main__":
    main()
