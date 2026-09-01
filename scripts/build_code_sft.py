#!/usr/bin/env python3
"""Assemble AV-SFT-shaped training parquets for the code experiments.

  exp1    E(code)          -> code            reconstruction. No spec anywhere,
                                              so the vector is unambiguously the
                                              only channel. Gates everything else.
  exp2a   E(buggy)         -> correct         repair, no spec (target
                                              underdetermined, but tests raw capacity)
  exp2b   E(buggy) + spec  -> correct         spec as PROMPT TEXT. The framework
                             in the prompt    predicts collapse here: Y indep V | Q,
                                              so this is a falsification test.
  exp2d   E(spec++buggy)   -> correct         joint encoding -- recommended form

Prompt is the 15-token minimal template (see commit 2fd78ac); keeping the marker
inside <concept>...</concept> preserves canonical neighbours (29,522) so the
sidecar carries over unchanged.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

PROJ = Path(__file__).resolve().parent.parent
STRUCT = pa.struct([("role", pa.string()), ("content", pa.string())])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="parquet with code/spec columns")
    p.add_argument("--vectors", required=True, help="parquet from extract_code_vectors.py")
    p.add_argument("--sidecar", default=str(PROJ / "data/minprompt/av_sft_train.parquet.nla_meta.yaml"))
    p.add_argument("--exp", required=True, choices=["exp1", "exp2a", "exp2b", "exp2d"])
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--eval-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    meta = yaml.safe_load(open(args.sidecar))
    inj = meta["tokens"]["injection_char"]

    src = pq.read_table(args.source).to_pandas()
    vec_t = pq.read_table(args.vectors)
    vecs = np.array(vec_t.column("activation_vector").to_pylist(), dtype=np.float32)
    assert len(src) == len(vecs), f"source {len(src)} != vectors {len(vecs)} — regenerate one of them"

    target_col = "code" if args.exp == "exp1" else "correct_code"
    targets = [f"<code>\n{c}\n</code>" for c in src[target_col]]

    if args.exp == "exp2b":
        prompts = [[{"role": "user",
                     "content": f"{s}\n<concept>{inj}</concept>"}] for s in src["spec"]]
    else:
        prompts = [[{"role": "user", "content": f"<concept>{inj}</concept>"}]] * len(src)

    # Split so no program appears on both sides. MBPP rows carry task_id; CSN
    # rows are already deduplicated by code, so a row split is doc-disjoint there.
    rng = np.random.default_rng(args.seed)
    if "task_id" in src.columns:
        ids = src["task_id"].unique()
        rng.shuffle(ids)
        n_ev = max(1, int(len(ids) * args.eval_frac))
        ev_ids = set(ids[:n_ev].tolist())
        is_eval = src["task_id"].isin(ev_ids).values
        print(f"split by task_id: {len(ev_ids)}/{len(ids)} problems held out")
    else:
        is_eval = np.zeros(len(src), bool)
        is_eval[rng.permutation(len(src))[:max(1, int(len(src) * args.eval_frac))]] = True
        print("split by row (corpus already deduped by code)")

    for name, sel in (("train", ~is_eval), ("eval", is_eval)):
        idx = np.where(sel)[0]
        tbl = pa.table({
            "prompt": pa.array([prompts[i] for i in idx], type=pa.list_(STRUCT)),
            "response": pa.array([targets[i] for i in idx]),
            # MUST be fixed_size_list: ActorDataset reads chunked.type.list_size
            # and reshapes the flat buffer. A variable-size list has no
            # .list_size and fails at load with an AttributeError.
            # Built from the flat buffer -- list(vecs[idx]) materialises one
            # numpy array per row and takes minutes at 150k rows.
            "activation_vector": pa.FixedSizeListArray.from_arrays(
                pa.array(np.ascontiguousarray(vecs[idx]).reshape(-1), type=pa.float32()),
                int(vecs.shape[1])),
            "doc_id": pa.array([str(src.iloc[i].get("task_id", i)) for i in idx]),
            "n_raw_tokens": pa.array([0] * len(idx), type=pa.int64()),
        })
        out = f"{args.out_prefix}_{name}.parquet"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(tbl, out)
        shutil.copy(args.sidecar, out + ".nla_meta.yaml")
        print(f"  {out}  {len(idx)} rows")

    tl = [len(t.split()) for t in targets]
    print(f"target words: mean={np.mean(tl):.1f} median={np.median(tl):.0f} p90={np.percentile(tl,90):.0f}")


if __name__ == "__main__":
    main()
