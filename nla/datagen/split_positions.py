"""Document-level three-way split by doc_id.

Partition at the DOCUMENT level (by doc_id), not the row level. Stage 0
samples ~N positions per document; row-level split would leak the same
document's context across AV/AR/RL subsets, contaminating the SL ↔ RL boundary.

Runs AFTER labeling — takes the fully-labeled positions parquet and splits
into {av_sft,ar_sft,rl}_explained.parquet. Splitting late means you can vary
the split ratio per model scale without re-labeling. The RL subset keeps the
api_explanation column (if present) but it's ignored at build time.

Adapted from nla.datagen.stage1_split — reads only doc_id for the split logic,
passes all other columns through unchanged.
"""

import argparse
import random

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", required=True, help="explained positions parquet from label_positions")
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--av-sft-frac", type=float, default=0.25,
        help="fraction of docs for actor SFT (default: 0.25)",
    )
    p.add_argument(
        "--ar-sft-frac", type=float, default=0.25,
        help="fraction of docs for critic SFT (default: 0.25)",
    )
    p.add_argument(
        "--rl-frac", type=float, default=0.50,
        help="fraction of docs for RL (default: 0.50)",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    fracs = (args.av_sft_frac, args.ar_sft_frac, args.rl_frac)
    assert all(f >= 0 for f in fracs), f"fractions must be non-negative, got {fracs}"
    total = sum(fracs)
    assert abs(total - 1.0) < 1e-6, f"fractions must sum to 1.0, got {total}"

    # Read ONLY doc_id to compute the split — avoids loading heavy columns.
    pf = pq.ParquetFile(args.input)
    doc_id_col = pf.read(columns=["doc_id"]).column("doc_id").to_pylist()

    # sorted() makes the shuffle deterministic across Python versions / hash seeds.
    doc_ids = sorted(set(doc_id_col))
    rng = random.Random(args.seed)
    rng.shuffle(doc_ids)

    n_docs = len(doc_ids)
    n_av = int(n_docs * args.av_sft_frac)
    n_ar = int(n_docs * args.ar_sft_frac)
    buckets = {
        "av_sft": set(doc_ids[:n_av]),
        "ar_sft": set(doc_ids[n_av : n_av + n_ar]),
        "rl": set(doc_ids[n_av + n_ar :]),
    }

    schema = pf.schema_arrow
    out_paths = {
        s: f"{args.output_dir.rstrip('/')}/{s}_explained.parquet" for s in buckets
    }
    writers = {
        s: pq.ParquetWriter(out_paths[s], schema) for s in buckets
    }
    row_counts = {s: 0 for s in buckets}

    # Stream row groups to keep memory bounded.
    for batch in pf.iter_batches(batch_size=65536):
        batch_docs = batch.column("doc_id").to_pylist()
        for stage, bucket_ids in buckets.items():
            mask = pa.array([d in bucket_ids for d in batch_docs], type=pa.bool_())
            subset = batch.filter(mask)
            if subset.num_rows > 0:
                writers[stage].write_table(pa.Table.from_batches([subset]))
                row_counts[stage] += subset.num_rows

    for w in writers.values():
        w.close()

    for stage, bucket_ids in buckets.items():
        print(
            f"{stage}: {len(bucket_ids)} docs"
            f" → {row_counts[stage]} rows → {out_paths[stage]}"
        )


if __name__ == "__main__":
    main()
