"""CPU-only text position extraction — tokenize corpus, sample positions, decode text.

Model-agnostic: loads only the tokenizer (no model weights), samples positions
deterministically per document, and decodes text at each position.

No activation vectors — those are extracted separately per model via
extract_vectors.py (GPU). This decoupling means one positions file serves all
models sharing the same tokenizer.

Output schema:
    doc_id                       string   provenance (corpus:split:doc_index)
    n_raw_tokens                 int64    1-indexed count of tokens up to extraction position
    detokenized_text_truncated   string   decoded text (skip_special_tokens=True) up to position
"""

import argparse
import hashlib
import random

import pyarrow as pa
import pyarrow.parquet as pq
from itertools import islice

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from nla import (
    _TOKENIZER_FINGERPRINT_KEY,
    _TOKENIZER_NAME_KEY,
    compute_tokenizer_fingerprint,
)
from nla.datagen._common import add_config_arg, apply_config

_MIN_POSITION = 50          # need enough left-context for the activation to be meaningful
_CHUNK_SIZE = 256            # docs per parquet write batch


def _sample_positions(
    token_ids: list[int],
    n_positions: int,
    special_ids: set[int],
    doc_id: str,
    seed: int,
    min_position: int = _MIN_POSITION,
) -> list[int]:
    """Sample token positions deterministically per document.

    Uses per-doc keyed RNG so the same (seed, doc_id) produces identical
    positions regardless of chunk boundaries, slice ordering, or process count.
    This is what makes labels reusable across models with the same tokenizer.
    """
    rng = random.Random(hashlib.sha256(f"{seed}|{doc_id}".encode()).digest())
    candidates = [
        i
        for i, tid in enumerate(token_ids)
        if i >= min_position and tid not in special_ids
    ]
    if not candidates:
        return []
    k = min(n_positions, len(candidates))
    return rng.sample(candidates, k=k)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--tokenizer-name", required=True, help="HF tokenizer name, e.g. Qwen/Qwen3-4B"
    )
    p.add_argument(
        "--corpus", required=True, help="HF dataset name, e.g. HuggingFaceFW/fineweb"
    )
    p.add_argument("--corpus-config", default=None, help="HF dataset config name")
    p.add_argument("--corpus-split", default="train")
    p.add_argument("--corpus-start", type=int, default=0)
    p.add_argument(
        "--corpus-length", type=int, required=True, help="number of documents to process"
    )
    p.add_argument("--text-column", default="text")
    p.add_argument(
        "--positions-per-doc", type=int, default=10, help="positions to sample per document"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--min-position",
        type=int,
        default=_MIN_POSITION,
        help="skip first N token positions (need left-context)",
    )
    p.add_argument(
        "--max-length", type=int, default=2048,
        help="max tokens per document — truncate before sampling (default: 2048, matches original)",
    )
    p.add_argument("--output", required=True, help="output parquet path")
    add_config_arg(p)
    args = apply_config(p)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.truncation_side = "right"

    # Compute tokenizer fingerprint — embedded in parquet metadata so downstream
    # training can verify label compatibility before touching GPU.
    fingerprint = compute_tokenizer_fingerprint(tokenizer)
    print(f"tokenizer: {args.tokenizer_name}")
    print(f"  fingerprint: {fingerprint[:16]}…  (vocab={tokenizer.vocab_size}, "
          f"specials={len(tokenizer.all_special_ids)})")

    special_ids = set(tokenizer.all_special_ids)
    pad_id_to_check = (
        tokenizer.pad_token_id
        if (
            tokenizer.pad_token_id is not None
            and tokenizer.pad_token_id != tokenizer.eos_token_id
        )
        else None
    )

    # Streaming: lists shard URLs (fast, no footer scan), opens shards
    # sequentially, .skip() reads through earlier docs, .take() stops after
    # N docs.  os._exit(0) at end of main() is needed because streaming
    # prefetch threads block clean shutdown when crossing shard boundaries.
    ds = load_dataset(
        args.corpus, name=args.corpus_config, split=args.corpus_split,
        streaming=True,
    )
    ds = ds.skip(args.corpus_start)
    if args.corpus_length:
        ds = ds.take(args.corpus_length)

    schema = pa.schema(
        [
            ("doc_id", pa.string()),
            ("n_raw_tokens", pa.int64()),
            ("detokenized_text_truncated", pa.string()),
        ]
    ).with_metadata(
        {
            _TOKENIZER_FINGERPRINT_KEY: fingerprint.encode(),
            _TOKENIZER_NAME_KEY: args.tokenizer_name.encode(),
        }
    )

    n_positions = 0
    n_docs_processed = 0
    n_docs_skipped = 0
    n_docs_short = 0
    total_docs = args.corpus_length

    with pq.ParquetWriter(args.output, schema) as writer:
        it = iter(ds)
        with tqdm(total=total_docs, desc="extracting positions") as pbar:
            while True:
                chunk = list(islice(it, _CHUNK_SIZE))
                if not chunk:
                    break
                texts = [doc[args.text_column] for doc in chunk]

                rows: dict[str, list] = {k: [] for k in schema.names}
                for doc_offset, text in enumerate(texts):
                    doc_idx = args.corpus_start + n_docs_processed + doc_offset
                    doc_id = f"{args.corpus}:{args.corpus_split}:{doc_idx}"

                    token_ids = tokenizer(
                        text,
                        add_special_tokens=True,
                        truncation=True,
                        max_length=args.max_length,
                    )["input_ids"]

                    if pad_id_to_check is not None:
                        assert pad_id_to_check not in token_ids, (
                            f"pad_token_id {pad_id_to_check} found in token_ids for {doc_id}"
                        )

                    positions = _sample_positions(
                        token_ids,
                        args.positions_per_doc,
                        special_ids,
                        doc_id,
                        args.seed,
                        args.min_position,
                    )
                    if not positions:
                        n_docs_skipped += 1
                        continue
                    if len(positions) < args.positions_per_doc:
                        n_docs_short += 1

                    for pos in positions:
                        n_raw_tokens = pos + 1
                        truncated_ids = token_ids[:n_raw_tokens]
                        rows["doc_id"].append(doc_id)
                        rows["n_raw_tokens"].append(n_raw_tokens)
                        rows["detokenized_text_truncated"].append(
                            tokenizer.decode(truncated_ids, skip_special_tokens=True)
                        )

                writer.write_table(pa.Table.from_pydict(rows, schema=schema))
                n_positions += len(rows["doc_id"])
                n_docs_processed += len(texts)
                pbar.update(len(chunk))

    print(f"wrote {n_positions} rows → {args.output}")
    print(
        f"  skipped {n_docs_skipped} docs"
        f" (too short / all-special past position {args.min_position})"
    )
    print(
        f"  short-sampled {n_docs_short} docs"
        f" (fewer than {args.positions_per_doc} valid positions)"
    )


if __name__ == "__main__":
    main()
    # Streaming mode leaves background prefetch threads that block clean
    # shutdown when extraction spans multiple shards.  The parquet file is
    # fully written before main returns, so force-exit is safe.
    import os as _os
    _os._exit(0)
