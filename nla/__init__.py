"""nla — lightweight, model-agnostic NLA pipeline for small HuggingFace models."""

import hashlib
import json

__version__ = "0.1.0"

# Parquet schema metadata keys
_TOKENIZER_FINGERPRINT_KEY = b"nla.tokenizer_fingerprint"
_TOKENIZER_NAME_KEY = b"nla.tokenizer_name"


def compute_tokenizer_fingerprint(tokenizer) -> str:
    """Compute a stable fingerprint of a tokenizer's vocabulary.

    Hashes the sorted vocab (token-string → id) plus special token config.
    Identical vocab + special tokens → identical fingerprint, regardless of
    which specific HF model name was used to load the tokenizer. This is what
    makes labels transferable across a model family.
    """
    vocab = tokenizer.get_vocab()
    # Sort by token id for deterministic ordering
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    payload = json.dumps(
        {
            "vocab": sorted_vocab,
            "all_special_ids": sorted(tokenizer.all_special_ids),
            "all_special_tokens": sorted(tokenizer.all_special_tokens),
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "vocab_size": tokenizer.vocab_size,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def verify_tokenizer_fingerprint(parquet_path: str, tokenizer) -> bool:
    """Check whether a tokenizer is compatible with a labeled parquet file.

    Returns True if the tokenizer produces the same fingerprint embedded in the
    file's metadata. Call this in training-prep code to catch tokenizer mismatches
    early — before wasting GPU time on incompatible labels.
    """
    import pyarrow.parquet as pq

    fingerprint = compute_tokenizer_fingerprint(tokenizer)
    pf = pq.ParquetFile(parquet_path)
    stored = pf.schema_arrow.metadata.get(_TOKENIZER_FINGERPRINT_KEY, b"").decode()
    if not stored:
        raise ValueError(
            f"{parquet_path} has no tokenizer fingerprint — was it produced by an "
            f"older version of extract_positions.py? Re-extract to get compatibility checks."
        )
    return stored == fingerprint
