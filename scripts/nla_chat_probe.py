#!/usr/bin/env python3
"""Can we interrogate the activation vector conversationally?

Turn 1: inject the vector, actor writes its <explanation>.
Turn 2+: append user questions about the ORIGINAL text and let the actor answer.
The injected vector stays in context, so the model can attend back to it on
every later turn -- the question is whether it holds information the single
explanation did not surface.

CONTROL: every question is asked twice, once with the real vector and once with
a ZERO vector, same prompts. Without that we cannot tell reading from
confabulation -- a fluent wrong answer looks identical to a fluent right one.

CAVEAT: the actor was SFT'd on SINGLE-turn data only. Multi-turn is
out-of-distribution, so poor instruction-following here is expected and is not
evidence about the vector.
"""
import argparse, torch

TEXTS = [
    ("en-news", "The Federal Reserve announced yesterday that it would raise interest rates by 0.25 percentage points, citing persistent inflation in the services sector and a labour market that remains"),
    ("zh-news", "据新华社报道，国务院昨日发布了关于进一步深化医疗改革的指导意见，要求各省在年底前"),
    ("en-code", "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) //"),
]
# (question, assistant PREFILL). The prefill matters: the actor was SFT'd on
# 124k single-turn examples of exactly one shape (vector in, <explanation> out)
# for 3 epochs and has largely lost general instruction-following -- asked a
# follow-up it just re-emits its turn-1 explanation. Seeding the assistant turn
# mid-sentence makes restarting that template the unlikely continuation.
QUESTIONS = [
    ("What language is the original text written in?",
     "The original text is written in"),
    ("What is the topic of the original text?",
     "The original text is about"),
    ("Quote the final few words of the original text as exactly as you can.",
     "The final words of the original text are:"),
    ("What kind of document is the original text from?",
     "The original text appears to come from"),
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
    inj_char = tm["injection_char"]
    tpl = side["prompt_templates"]["actor"]
    assert "{injection_char}" in tpl

    tok = AutoTokenizer.from_pretrained(args.actor_ckpt)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    print(f"extracting activations @ layer {args.layer} ...")
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16,
                                                device_map={"": dev}).eval()
    vecs = {}
    for tag, text in TEXTS:
        ids = tok(text, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            out = base(input_ids=ids, output_hidden_states=True)
        vecs[tag] = out.hidden_states[args.layer][0, -1].float().cpu()
    del base; torch.cuda.empty_cache()

    actor = AutoModelForCausalLM.from_pretrained(args.actor_ckpt, torch_dtype=torch.bfloat16,
                                                 device_map={"": dev}).eval()

    def gen(messages, vec, max_new=160, prefill=None):
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if prefill:
            prompt = prompt + prefill
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

    for tag, text in TEXTS:
        print("\n" + "#" * 74)
        print(f"# {tag}   SOURCE: {text[:100]}")
        print("#" * 74)
        for cond in ("REAL", "ZERO"):
            vec = vecs[tag] if cond == "REAL" else torch.zeros_like(vecs[tag])
            msgs = [{"role": "user", "content": tpl.format(injection_char=inj_char)}]
            expl = gen(msgs, vec, max_new=200)
            print(f"\n--- [{cond}] turn 1 (explanation) ---")
            print(expl[:400])
            msgs = msgs + [{"role": "assistant", "content": expl}]
            for q, prefill in QUESTIONS:
                turn = msgs + [{"role": "user", "content": q}]
                ans = gen(turn, vec, max_new=70, prefill=prefill)
                print(f"\n  [{cond}] Q: {q}")
                print(f"  [{cond}] A: {prefill}{ans[:260]}")

if __name__ == "__main__":
    main()
