#!/usr/bin/env python3
"""Best-of-n reranking by embedding distance -- is the ceiling search or capacity?

Vec2Text (arXiv 2310.06816) shows single-pass inversion of an embedding is the
hard way: one forward pass gets ~30 BLEU, iterative correction reaches 92% exact
match. Our exp1 IS the single-pass baseline.

Before building a corrector, test the cheaper half of the idea: sample n
candidates, re-encode each with the FROZEN base model, and keep the one closest
to the target vector. No new model, no training.

  greedy      what we measured (0/290 exact)
  best-of-n   rerank by cos(v_target, E(candidate))   <- can the model already
                                                         produce better output?
  oracle      rerank by true similarity to the target code -- an upper bound on
              what ANY reranker over these samples could achieve

If best-of-n moves exact match off zero, the ceiling is search. If oracle is
also ~0, the samples simply do not contain the right program and the ceiling is
capacity.
"""
import argparse, ast, difflib, json, re
def _norm_ast(src):
    """Exact-modulo-renaming: canonical identifiers, literals KEPT."""
    try: t = ast.parse(src or "")
    except SyntaxError: return None
    m = {}
    for n in ast.walk(t):
        for a in ("id","arg","name","attr"):
            v = getattr(n, a, None)
            if isinstance(v, str):
                m.setdefault(v, f"V{len(m)}"); setattr(n, a, m[v])
    try: return ast.dump(t)
    except Exception: return None
def _skel(src):
    """Graded shape: node TYPE sequence, identifiers and literals discarded."""
    try: return [type(n).__name__ for n in ast.walk(ast.parse(src or ""))]
    except SyntaxError: return []
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, torch

PROJ = Path(__file__).resolve().parent.parent
_CODE_RE = re.compile(r"<code>\s*(.*?)\s*</code>", re.DOTALL)

def extract(s):
    m = _CODE_RE.search(s or ""); return m.group(1) if m else None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--actor-ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--model-name", default="Qwen/Qwen3-8B")
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--n-candidates", type=int, default=16)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--centre-mean", default=None,
                   help="npy of the train mean. REQUIRED when --data holds centred "
                        "vectors: candidates are encoded raw, so without this the "
                        "cosine compares centred targets to raw encodings.")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    import sys; sys.path.insert(0, str(PROJ))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from nla.training.injection import inject_at_marked_positions
    from nla.training.schema import normalize_activation
    from nla.training.sidecar import read_sidecar

    tk = read_sidecar(args.data)["tokens"]
    inj, L, R = tk["injection_token_id"], tk["injection_left_neighbor_id"], tk["injection_right_neighbor_id"]
    t = pq.read_table(args.data)
    n = min(args.n_samples, t.num_rows)
    prompts = [m[0]["content"] for m in t.column("prompt").to_pylist()[:n]]
    targets = [extract(r) for r in t.column("response").to_pylist()[:n]]
    V = torch.tensor(np.array(t.column("activation_vector").to_pylist()[:n], dtype=np.float32))

    tok = AutoTokenizer.from_pretrained(args.actor_ckpt)
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    actor = AutoModelForCausalLM.from_pretrained(
        args.actor_ckpt, torch_dtype=torch.bfloat16, device_map={"": args.device}).eval()
    d = actor.config.hidden_size; scale = 2.5 * d ** 0.5
    embed = actor.get_input_embeddings()

    # candidates: 1 greedy + (n_candidates-1) sampled, per row
    cands = [[] for _ in range(n)]
    with torch.no_grad():
        for k in range(args.n_candidates):
            greedy = (k == 0)
            for i in range(0, n, 8):
                bp = prompts[i:i+8]
                bv = normalize_activation(V[i:i+8], scale).to(args.device)
                strs = [tok.apply_chat_template([{"role":"user","content":c}], tokenize=False,
                                                add_generation_prompt=True) for c in bp]
                enc = tok(strs, return_tensors="pt", padding=True, add_special_tokens=False)
                ids, m = enc["input_ids"].to(args.device), enc["attention_mask"].to(args.device)
                def _h(_m, a, out):
                    return out if out.shape[1] <= 1 else inject_at_marked_positions(a[0], out, bv, inj, L, R)
                h = embed.register_forward_hook(_h)
                try:
                    g = actor.generate(ids, attention_mask=m, max_new_tokens=args.max_new_tokens,
                                       do_sample=not greedy,
                                       temperature=None if greedy else args.temperature,
                                       top_p=None if greedy else 0.95,
                                       pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
                finally: h.remove()
                for j in range(len(bp)):
                    cands[i+j].append(extract(tok.decode(g[j][ids.shape[1]:], skip_special_tokens=True)))
            print(f"  candidate {k+1}/{args.n_candidates} done", flush=True)
    del actor; torch.cuda.empty_cache()

    # re-encode every candidate with the FROZEN base model at layer 24
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map={"": args.device}).eval()
    layer = (2 * len(base.model.layers)) // 3
    cap = {}
    hh = base.model.layers[layer].register_forward_hook(
        lambda _m,_a,o: cap.__setitem__("h", (o[0] if isinstance(o,tuple) else o).detach()))
    def enc_all(texts):
        out = np.zeros((len(texts), d), np.float32)
        with torch.no_grad():
            for i in range(0, len(texts), 16):
                ch = [x if x and x.strip() else "pass" for x in texts[i:i+16]]
                e = tok(ch, return_tensors="pt", padding=True, truncation=True, max_length=1024)
                ii, mm = e["input_ids"].to(args.device), e["attention_mask"].to(args.device)
                base(input_ids=ii, attention_mask=mm, use_cache=False)
                out[i:i+len(ch)] = cap["h"][torch.arange(len(ch)), mm.sum(1)-1].float().cpu().numpy()
        return out
    tok.padding_side = "right"
    flat = [c for row in cands for c in row]
    E = enc_all(flat).reshape(n, args.n_candidates, d)
    hh.remove()

    Ev = E
    if args.centre_mean:
        mu = np.load(args.centre_mean).reshape(1,1,-1)
        Ev = E - mu
        print(f"centred candidate encodings using {args.centre_mean}")
    Vn = (V.numpy() / np.linalg.norm(V.numpy(),axis=1,keepdims=True).clip(1e-9))
    En = Ev / np.linalg.norm(Ev,axis=2,keepdims=True).clip(1e-9)
    cos = np.einsum('nd,nkd->nk', Vn, En)

    def score(sel, label):
        ex = na = sk9 = 0; sks = []
        for i,k in enumerate(sel):
            c, t = cands[i][k], targets[i]
            if not c or not t: continue
            if c.strip() == t.strip(): ex += 1
            pn, tn = _norm_ast(c), _norm_ast(t)
            if pn is not None and pn == tn: na += 1
            r = difflib.SequenceMatcher(None, _skel(c), _skel(t)).ratio()
            sks.append(r)
            if r > 0.9: sk9 += 1
        cs = np.mean([cos[i,k] for i,k in enumerate(sel)])
        sim = np.mean([difflib.SequenceMatcher(None,(cands[i][k] or '').split(),
                                               (targets[i] or '').split()).ratio() for i,k in enumerate(sel)])
        print(f"  {label:<22} exact={ex}/{n} ({100*ex/n:.1f}%)  norm-AST={na}/{n}  "
              f"skel>0.9={sk9}/{n}  skel={np.mean(sks):.3f}  cos={cs:.4f}  tok-sim={sim:.3f}")
        return ex

    print(f"\n{'='*66}\nn={n} rows x {args.n_candidates} candidates (1 greedy + {args.n_candidates-1} sampled @ T={args.temperature})\n{'='*66}")
    score([0]*n, "greedy (candidate 0)")
    score(list(cos.argmax(1)), f"best-of-{args.n_candidates} by cos")
    orc = [int(np.argmax([difflib.SequenceMatcher(None,(c or '').split(),
            (targets[i] or '').split()).ratio() for c in cands[i]])) for i in range(n)]
    score(orc, f"ORACLE best-of-{args.n_candidates}")
    print("="*66)
    if args.out:
        json.dump({"n":n,"k":args.n_candidates,
                   "cands":cands[:20],"targets":targets[:20]}, open(args.out,"w"))
        print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
