#!/usr/bin/env python3
"""tiny_nla demo — one entry point, no model to choose.

    pip install torch transformers accelerate huggingface_hub numpy pyyaml
    hf auth login
    hf download TuHan/tiny-nla nla_demo.py --local-dir .

    python nla_demo.py --mode code 'def gcd(a, b):
        while b: a, b = b, a % b
        return a'                              # -> reconstructs the function

    python nla_demo.py --mode text 'The Federal Reserve announced yesterday'
                                               # -> explains the activation

    python nla_demo.py --mode code --file mymodule.py --control

Compresses the input to ONE 4096-float activation vector from a frozen
Qwen3-8B, then asks a trained decoder to write it back out. YOU pick the mode;
the CHECKPOINT is chosen for you -- always whatever currently sits at the
pinned Hub paths, with no flag to select an older one.

Needs one GPU with >=24 GB. Models load sequentially, not together.
"""
import argparse, ast, builtins, difflib, keyword, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, torch, transformers, yaml
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "TuHan/tiny-nla"
CODE_SUB, TEXT_SUB = "code-decoder", "nla/actor"
BASE = "Qwen/Qwen3-8B"
# Verified per mode by A/B (job 17444219) -- do NOT "unify" these.
#   text + minimal  -> emits <answer>, drifts off-topic (regex misses, raw falls through)
#   code + sidecar  -> ident recall 0.00: writes a function ABOUT activation vectors,
#                      following the preamble's subject instead of the injected vector
#   the winners     -> code 1.00 recall / AST 0.862 ; text = correct 2-3 snippet format
# Each checkpoint wants the prompt IT was trained on. code-decoder's sidecar
# prompt_templates is a stale dataset-builder default and does not record that.
PROMPT_MODE = {"code": "minimal", "text": "sidecar"}

# transformers 5.x renamed torch_dtype -> dtype and warns on the old name; 4.x
# does not know the new one. An unpinned `pip install transformers` gets 5.x.
_DT = ({"dtype": torch.bfloat16}
       if int(transformers.__version__.split(".")[0]) >= 5
       else {"torch_dtype": torch.bfloat16})
CODE_RE = re.compile(r"<code>\s*(.*?)\s*</code>", re.DOTALL)
EXPL_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)
STOP = set(keyword.kwlist) | set(dir(builtins)) | {"self", "cls"}


def inject_at_marked_positions(input_ids, embeddings, vectors, inj_id, left_id, right_id):
    """Verbatim from nla/training/injection.py. The neighbour check prevents
    false positives; if injection silently fails the model emits Chinese."""
    out = embeddings.clone()
    vectors = vectors.to(out.device, out.dtype)
    n, k = input_ids.shape[-1], 0
    for b, p in (input_ids == inj_id).nonzero(as_tuple=False).tolist():
        if p == 0 or p == n - 1:
            continue
        if input_ids[b, p - 1] != left_id or input_ids[b, p + 1] != right_id:
            continue
        out[b, p] = vectors[k]; k += 1
    if k != vectors.shape[0]:
        raise RuntimeError(f"found {k} injection sites, expected {vectors.shape[0]}")
    return out


def normalize_activation(v, scale):
    return v / (v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12) / scale).to(v.dtype)


def scratch_dir():
    """A writable scratch directory that is never /tmp.

    /tmp is off limits on this cluster and is node-local anyway, but both
    tempfile and triton default there. Honour an explicit TMPDIR when it does
    not point into /tmp, else fall back to the user's cache dir.
    """
    for cand in (os.environ.get("TMPDIR"), os.environ.get("XDG_CACHE_HOME")):
        if cand and not os.path.realpath(cand).startswith("/tmp"):
            d = os.path.join(cand, "tiny_nla")
            try:
                os.makedirs(d, exist_ok=True)
                return d
            except OSError:
                pass
    d = os.path.join(os.path.expanduser("~"), ".cache", "tiny_nla")
    os.makedirs(d, exist_ok=True)
    return d


def ensure_compiler():
    """Pick a C compiler that actually works, before torch needs one.

    Triton JIT-compiles a CUDA helper on first GPU use and shells out to `cc`.
    On NSC/Berzelius the gcc first on PATH is a wrapper that refuses to run
    without a build-env module loaded; the refusal surfaces as a
    CalledProcessError fifteen frames deep inside triton, naming a .c file the
    user never wrote. Probe candidates once and export the first that links.
    Costs one ~100ms compile, and is skipped entirely if CC is already set.
    """
    if os.environ.get("CC"):
        return
    with tempfile.TemporaryDirectory(dir=scratch_dir()) as d:
        src = os.path.join(d, "probe.c")
        with open(src, "w") as f:
            f.write("int main(){return 0;}")
        for cand in (shutil.which("cc"), shutil.which("gcc"), "/usr/bin/gcc"):
            if not cand or not os.path.exists(cand):
                continue
            try:
                r = subprocess.run([cand, "-shared", "-fPIC", "-o",
                                    os.path.join(d, "probe.so"), src],
                                   capture_output=True, timeout=60)
            except Exception:
                continue
            if r.returncode == 0:
                os.environ["CC"] = cand
                return


def parses(s):
    try:
        ast.parse(s); return True
    except SyntaxError:
        return False


def idents(src):
    try:
        t = ast.parse(src)
    except SyntaxError:
        return set()
    return {v for n in ast.walk(t) for a in ("id", "arg", "name", "attr")
            if isinstance(v := getattr(n, a, None), str) and v not in STOP and not v.startswith("__")}


def main():
    p = argparse.ArgumentParser(description="tiny_nla demo — compress to one vector, decode it back")
    p.add_argument("--prompt", choices=["minimal", "sidecar"], default=None,
                   help="override the injection prompt (diagnostic; default is per-mode verified)")
    p.add_argument("--mode", required=True, choices=["code", "text"],
                   help="code = reconstruct a Python function; text = explain the activation")
    p.add_argument("input", nargs="?", help="Python function, or any text")
    p.add_argument("--file", help="read the input from a file instead")
    p.add_argument("--control", action="store_true",
                   help="also decode a RANDOM vector — what the model produces with no information")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if not a.input and not a.file:
        p.error("give some input, or --file")
    raw = Path(a.file).read_text() if a.file else a.input

    code_mode = a.mode == "code"
    if code_mode:
        if not parses(raw):
            sys.exit("--mode code needs valid Python; use --mode text for prose")
        raw = ast.unparse(ast.parse(raw))          # canonicalise so formatting is not counted
    # Keep triton's JIT scratch and every tempfile default off /tmp.
    _sd = scratch_dir()
    if os.path.realpath(os.environ.get("TMPDIR", "/tmp")).startswith("/tmp"):
        os.environ["TMPDIR"] = _sd
    tempfile.tempdir = _sd
    os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_sd, "triton"))
    ensure_compiler()
    sub = CODE_SUB if code_mode else TEXT_SUB
    print(f"mode: {'CODE reconstruction' if code_mode else 'TEXT explanation'}   "
          f"(checkpoint chosen automatically: {REPO}/{sub})")

    try:
        sc = yaml.safe_load(open(hf_hub_download(REPO, f"{sub}/nla_meta.yaml")))
    except Exception as e:
        if "401" in str(e) or "Repository Not Found" in str(e):
            sys.exit(f"cannot read {REPO}: it is private and this machine is not "
                     f"authenticated.\n  run:  hf auth login\n"
                     f"  then ask the repo owner to grant your HF account access.")
        raise
    meta = sc["tokens"]
    inj, L, R, ch = (meta["injection_token_id"], meta["injection_left_neighbor_id"],
                     meta["injection_right_neighbor_id"], meta["injection_char"])
    try:      # a centred decoder given a raw vector yields fluent, unrelated output
        mu = np.load(hf_hub_download(REPO, f"{sub}/centre_mean.npy")).reshape(-1)
    except Exception:
        mu = None

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(
        BASE, **_DT, device_map={"": a.device}).eval()
    layer = (2 * len(base.model.layers)) // 3
    cap = {}
    h = base.model.layers[layer].register_forward_hook(
        lambda _m, _i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
    with torch.no_grad():
        e = tok(raw, return_tensors="pt", truncation=True, max_length=1024)
        ids, m = e["input_ids"].to(a.device), e["attention_mask"].to(a.device)
        base(input_ids=ids, attention_mask=m, use_cache=False)
    h.remove()
    v = cap["h"][0, int(m.sum()) - 1].float().cpu().numpy()
    print(f"encoded {int(m.sum())} tokens at layer {layer}   ‖v‖={np.linalg.norm(v):.1f}"
          + (f"   centred (‖μ‖={np.linalg.norm(mu):.1f})" if mu is not None else ""))
    if mu is not None:
        v = v - mu
    del base; torch.cuda.empty_cache()

    actor = AutoModelForCausalLM.from_pretrained(
        REPO, subfolder=sub, **_DT, device_map={"": a.device}).eval()
    embed = actor.get_input_embeddings()

    # Which prompt a checkpoint wants is a property of how it was TRAINED, and the
    # sidecar's prompt_templates is a dataset-builder default that does not always
    # match. Both were measured; see PROMPT_MODE.
    want = a.prompt or PROMPT_MODE[a.mode]
    tmpl = (sc.get("prompt_templates") or {}).get("actor") if want == "sidecar" else None
    user_msg = tmpl.replace("{injection_char}", ch) if tmpl else f"<concept>{ch}</concept>"

    def gen(vec):
        bv = normalize_activation(torch.tensor(vec).unsqueeze(0), 300.0).to(a.device)
        s = tok.apply_chat_template([{"role": "user", "content": user_msg}],
                                    tokenize=False, add_generation_prompt=True)
        enc = tok(s, return_tensors="pt", add_special_tokens=False)
        i2, m2 = enc["input_ids"].to(a.device), enc["attention_mask"].to(a.device)
        hk = embed.register_forward_hook(
            lambda _m, i, o: o if o.shape[1] <= 1 else inject_at_marked_positions(i[0], o, bv, inj, L, R))
        try:
            with torch.no_grad():
                out = actor.generate(i2, attention_mask=m2, max_new_tokens=a.max_new_tokens,
                                     do_sample=False, pad_token_id=tok.pad_token_id,
                                     eos_token_id=tok.eos_token_id)
        finally:
            hk.remove()
        txt = tok.decode(out[0][i2.shape[1]:], skip_special_tokens=True)
        mm = (CODE_RE if code_mode else EXPL_RE).search(txt)
        return mm.group(1) if mm else txt.strip()[:600]

    bar = "=" * 70
    pred = gen(v)
    print(f"\n{bar}\nINPUT\n{bar}\n{raw[:600]}")
    print(f"{bar}\n{'RECONSTRUCTED' if code_mode else 'EXPLANATION'}\n{bar}\n{pred}")
    if code_mode and pred:
        pu, cu = idents(pred), idents(raw)
        sk = difflib.SequenceMatcher(None,
                                     [type(n).__name__ for n in ast.walk(ast.parse(pred))] if parses(pred) else [],
                                     [type(n).__name__ for n in ast.walk(ast.parse(raw))]).ratio()
        print(bar)
        print(f"exact match       : {pred.strip() == raw.strip()}")
        print(f"AST-skeleton sim  : {sk:.3f}   (~0.52 for two arbitrary functions)")
        print(f"identifier recall : {len(pu & cu)/max(len(cu),1):.2f}   shared: {sorted(pu & cu)[:8]}")
    if a.control:
        rng = np.random.default_rng(0)
        print(f"{bar}\nCONTROL — random vector, i.e. the model's prior with no information\n{bar}")
        print(gen(rng.normal(0, np.std(v), size=v.shape).astype(np.float32)))


if __name__ == "__main__":
    main()
