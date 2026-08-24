"""Split a labeled pool into two DOC-DISJOINT halves for AV-SFT and AR-SFT.

The design (original repo stage1_split.py, this repo's split_positions.py) puts
AV, AR and RL on disjoint document sets so the actor never learns to describe a
vector whose explanation the critic has already memorised. That split exists in
data/pool_ufw_8b/{av,ar,rl}, but the AR pool was only 1.8% labelled, so every run
so far trained BOTH models on the AV pool. This re-derives the boundary inside
the labelled AV pool: half the DOCUMENTS to the critic, half to the actor.

Splits at the document level, not the row level: stage0 samples ~10 positions per
document, so a row-level split would put the same document's context on both
sides of the AV/AR boundary.
"""

import argparse
import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ACT = "activation_vector"


def _filter(table: pa.Table, mask: np.ndarray) -> pa.Table:
    """Row-select a table whose activation column is a big FixedSizeList.

    numpy reorder rather than Table.filter(): the values buffer is ~4 GiB and
    arrow's take/filter path indexes offsets with uint32.
    """
    cols = {}
    idx = np.flatnonzero(mask)
    for name in table.column_names:
        if name == ACT:
            flat = table.column(name).combine_chunks()
            d = flat.type.list_size
            sub = flat.values.to_numpy().reshape(-1, d)[idx]
            cols[name] = pa.FixedSizeListArray.from_arrays(
                pa.array(sub.reshape(-1), type=pa.float32()), d
            )
        else:
            cols[name] = table.column(name).take(pa.array(idx))
    return pa.table(cols)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--explained", required=True)
    p.add_argument("--vectors", required=True)
    p.add_argument("--out-prefix-a", required=True, help="half A (critic / AR)")
    p.add_argument("--out-prefix-b", required=True, help="half B (actor / AV)")
    p.add_argument("--frac-a", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    expl = pq.read_table(args.explained)
    vecs = pq.read_table(args.vectors)
    print(f"explained: {expl.num_rows:,} rows   vectors: {vecs.num_rows:,} rows")

    e_docs = expl.column("doc_id").to_pylist()
    v_docs = vecs.column("doc_id").to_pylist()

    # sorted() before shuffle so the split is reproducible across runs/hash seeds
    docs = sorted(set(e_docs))
    rng = random.Random(args.seed)
    rng.shuffle(docs)
    n_a = int(len(docs) * args.frac_a)
    set_a, set_b = set(docs[:n_a]), set(docs[n_a:])
    assert not (set_a & set_b)
    print(f"documents: {len(docs):,}  ->  A={len(set_a):,}  B={len(set_b):,}")

    for tag, keep, pref in (("A", set_a, args.out_prefix_a),
                            ("B", set_b, args.out_prefix_b)):
        e_mask = np.fromiter((d in keep for d in e_docs), dtype=bool, count=len(e_docs))
        v_mask = np.fromiter((d in keep for d in v_docs), dtype=bool, count=len(v_docs))
        e_half, v_half = _filter(expl, e_mask), _filter(vecs, v_mask)
        pq.write_table(e_half, f"{pref}_explained.parquet")
        pq.write_table(v_half, f"{pref}_vectors.parquet")
        print(f"  half {tag}: {e_half.num_rows:,} explained  {v_half.num_rows:,} vectors "
              f"-> {pref}_{{explained,vectors}}.parquet")

    print("SPLIT DONE")


if __name__ == "__main__":
    main()
