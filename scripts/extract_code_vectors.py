#!/usr/bin/env python3
"""Extract last-token activation vectors from code (and optionally a spec).

The NLA text pipeline samples MID-document positions with a min_position floor
(extract_positions.py), because it wants a prefix summary. Here we want a
WHOLE-PROGRAM summary, so we take the final token instead.

--mode selects what gets encoded, which is the experimental variable:
  code       E(code)            reconstruction, and the no-spec repair arm
  spec_code  E(spec ++ code)    joint encoding -- the spec x code interaction
                                happens inside the frozen model at full
                                attention, so the vector carries a conclusion
                                rather than two premises
  spec       E(spec)            for the two-vector arm

Layer 24 = (2*36)//3, matching every existing vectors parquet.
"""

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

PROJ = Path(__file__).resolve().parent.parent


def build_text(row, mode):
    if mode == "code":
        return row["code"]
    if mode == "spec":
        return row["spec"]
    if mode == "spec_code":
        # code last: the trailing tokens dominate the residual stream, and the
        # implementation is what the vector must be "about".
        return f"# {row['spec']}\n{row['code']}"
    raise ValueError(mode)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="parquet with a 'code' column (and 'spec' if needed)")
    p.add_argument("--output", required=True)
    p.add_argument("--mode", default="code", choices=["code", "spec_code", "spec"])
    p.add_argument("--model-name", default="Qwen/Qwen3-8B")
    p.add_argument("--layer-index", type=int, default=None, help="default 2/3 depth")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    df = pq.read_table(args.input).to_pandas()
    if args.limit:
        df = df.head(args.limit)
    texts = [build_text(r, args.mode) for _, r in df.iterrows()]
    print(f"{len(texts)} rows  mode={args.mode}")

    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"          # so the last real token is at sum(mask)-1

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map={"": args.device})
    model.eval()
    layers = model.model.layers
    li = args.layer_index if args.layer_index is not None else (2 * len(layers)) // 3
    print(f"layers={len(layers)}  extracting at layer {li}")

    captured = {}

    def _hook(_m, _a, out):
        captured["h"] = (out[0] if isinstance(out, tuple) else out).detach()

    h = layers[li].register_forward_hook(_hook)
    vecs = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    try:
        with torch.no_grad():
            for i in range(0, len(texts), args.batch_size):
                chunk = texts[i:i + args.batch_size]
                enc = tok(chunk, return_tensors="pt", padding=True,
                          truncation=True, max_length=args.max_length)
                ids = enc["input_ids"].to(args.device)
                mask = enc["attention_mask"].to(args.device)
                model(input_ids=ids, attention_mask=mask, use_cache=False)
                last = mask.sum(dim=1) - 1
                v = captured["h"][torch.arange(len(chunk)), last]
                vecs[i:i + len(chunk)] = v.float().cpu().numpy()
                if (i // args.batch_size) % 50 == 0:
                    print(f"  {i + len(chunk)}/{len(texts)}", flush=True)
    finally:
        h.remove()

    out = pa.table({
        "doc_id": pa.array([f"{args.mode}:{i}" for i in range(len(texts))]),
        # fixed_size_list, matching what ActorDataset requires
        "activation_vector": pa.FixedSizeListArray.from_arrays(
            pa.array(vecs.reshape(-1), type=pa.float32()), int(vecs.shape[1])),
        "activation_layer": pa.array([li] * len(texts), type=pa.int64()),
        "n_raw_tokens": pa.array(
            [len(tok(t, add_special_tokens=True)["input_ids"]) for t in texts], type=pa.int64()),
    })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, args.output)
    nt = out.column("n_raw_tokens").to_pylist()
    print(f"wrote {args.output}  {len(texts)} vectors  "
          f"tokens: mean={np.mean(nt):.1f} median={np.median(nt):.0f} p90={np.percentile(nt,90):.0f}")
    norms = np.linalg.norm(vecs, axis=1)
    print(f"vector norms: mean={norms.mean():.2f} std={norms.std():.2f}")


if __name__ == "__main__":
    main()
