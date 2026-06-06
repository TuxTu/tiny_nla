"""API labeling — generate natural-language explanations via LLM.

Takes positions.parquet from extract_positions, calls DeepSeek or Anthropic to
explain each text position, and writes an explained parquet with an added
`api_explanation` column.

This is the second and final step of the labeling pipeline. Labeling is
model-agnostic — it only needs decoded text, never activation vectors.
Models sharing a tokenizer produce identical text at identical positions,
so labels are reusable across an entire model family.

Cache: explanations are keyed by sha256(detokenized_text_truncated).
Point --cache-from at existing explained parquets (or a standalone JSON cache)
to skip API calls for texts already labeled.

Standalone cache export (--cache-export): writes {sha256: explanation} as JSON
for distribution (Google Drive, etc.). Colab users download the cache and pass
--cache-from to label without an API key.

Chunked processing with crash-resume: each chunk goes to {output}.chunks/.
Existing chunk files are skipped on restart — no API calls wasted.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from nla import _TOKENIZER_FINGERPRINT_KEY, _TOKENIZER_NAME_KEY
from nla.datagen._common import add_config_arg, apply_config
from nla.datagen.providers import CompletionProvider, resolve_provider

# Proven instruction template from the original NLA paper appendix.
# Shortened from 4-5 → 2-3 features, ~80-100 words — responses reliably
# fit in 300 tokens WITH closing tag.
_DEFAULT_INSTRUCTION = """A language model needs to predict what text comes next after a snippet which will be presented to you shortly. Identify the 2-3 most important features it would use for this prediction.
Focus on what the language model must be "thinking about" at the point where the provided text ends. You should not need to reference the fact that the text is truncated/incomplete/a prefix: the language model is causal, so only sees the prefix to what it predicts and this is implicit.
Order features by what is most important for predicting the next tokens. Each feature should consist of a concise ~10-20 word description. Feel free to include specific textual examples inline.

Feature types to consider (as inspiration, not a rigid checklist):
- Syntactic/structural constraints: "unclosed parenthesis requires matching close"
- Immediate semantic expectations: "list promised three items but only two given"
- Stylistic/register patterns: "formal academic tone maintained throughout"
- Narrative/argumentative momentum: "thesis stated, supporting evidence now expected"
- Domain/genre signals: "medical case history following SOAP format"
- Repetition/continuation patterns: "same phrase structure repeating with variations"

The final feature must describe the very end of the presented sequence: its role, what it's part of, and immediate constraints on what follows.

Format — IMPORTANT: keep to ~80-100 words total and ALWAYS close the tag. Separate each feature with a blank line (press Enter twice between features):
<analysis>
[first feature — include specific examples when relevant]

[second feature]

[final feature: the last token, its role, immediate constraints]
</analysis>

Text to analyze:

<begin_text>{text}<end_text>"""

# Strict: both opening and closing tags MUST be present.
_DEFAULT_RESPONSE_PATTERN = r"<analysis>\s*(.*?)\s*</analysis>"

# Minimum features required — fewer than 2 means the model ignored format.
_MIN_FEATURES = 1

# Prefix stripping — API models use all kinds of list markers.
_LIST_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"[-*•+–—]"              # bullet chars
    r"|\d+[.)]"               # 1. 1)
    r"|\(\d+\)"               # (1)
    r"|[a-zA-Z][.)]"          # a. a)
    r"|\([a-zA-Z]\)"          # (a)
    r"|[ivxIVX]+[.)]"         # i. ii)
    r")\s+"
)
_BOLD_WRAP_RE = re.compile(r"^\*\*(.+?)\*\*\s*")


def _extract_and_clean(raw: str, pattern: str) -> str | None:
    """Extract content inside response tags, strip list formatting.

    Returns None if the pattern doesn't match (truncated, no tags) — caller
    drops the row.
    """
    m = re.search(pattern, raw, flags=re.DOTALL)
    if m is None:
        return None
    content = m.group(1)

    cleaned: list[str] = []
    for line in content.split("\n"):
        line = _LIST_PREFIX_RE.sub("", line)
        line = _BOLD_WRAP_RE.sub(r"\1 ", line)
        line = line.strip().strip("*_")
        if line:
            cleaned.append(line)
    return "\n\n".join(cleaned)


def load_cache(paths: list[str]) -> dict[str, str]:
    """Build {sha256(text): explanation} from parquet or JSON files."""
    cache: dict[str, str] = {}
    for path in paths:
        path = Path(path)
        if path.suffix == ".json":
            with open(path) as f:
                raw = json.load(f)
            for k, v in raw.items():
                cache[k] = v
            print(f"  cache: +{len(raw)} from {path}")
        else:
            t = pq.read_table(
                path, columns=["detokenized_text_truncated", "api_explanation"]
            )
            texts = t.column("detokenized_text_truncated").to_pylist()
            expls = t.column("api_explanation").to_pylist()
            for txt, expl in zip(texts, expls, strict=True):
                cache[hashlib.sha256(txt.encode()).hexdigest()] = expl
            print(f"  cache: +{len(texts)} from {path}")
    print(f"  cache: {len(cache)} unique texts loaded")
    return cache


def lookup(cache: dict[str, str], text: str) -> str | None:
    return cache.get(hashlib.sha256(text.encode()).hexdigest())


def export_cache_json(
    cache: dict[str, str], path: str,
    fingerprint: str = "", tokenizer_name: str = "",
) -> None:
    """Export standalone cache as JSON for distribution.

    Includes tokenizer fingerprint so downstream users can verify the cache
    is compatible with their model's tokenizer before downloading.
    """
    out: dict = {}
    meta: dict = {
        "description": "NLA explanation cache — keyed by sha256(detokenized_text_truncated)",
        "num_entries": len(cache),
    }
    if fingerprint:
        meta["tokenizer_fingerprint"] = fingerprint
        meta["tokenizer_name"] = tokenizer_name
    out["_meta"] = meta
    out.update(cache)
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"cache exported: {len(cache)} entries → {path}")
    if fingerprint:
        print(f"  tokenizer: {tokenizer_name}  fingerprint: {fingerprint[:16]}…")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", required=True, help="positions.parquet from extract_positions")
    p.add_argument("--output", required=True, help="output explained parquet path")
    p.add_argument(
        "--provider", default="deepseek",
        help="provider name: deepseek | anthropic (default: deepseek)",
    )
    p.add_argument(
        "--provider-model", default=None,
        help="override provider's default model",
    )
    p.add_argument(
        "--provider-concurrency", type=int, default=32,
        help="max concurrent API calls (default: 32)",
    )
    p.add_argument(
        "--provider-max-tokens", type=int, default=400,
        help="max tokens per completion (default: 400)",
    )
    p.add_argument(
        "--instruction-template", default=_DEFAULT_INSTRUCTION,
        help="prompt template with {text} placeholder",
    )
    p.add_argument(
        "--response-extract-pattern", default=_DEFAULT_RESPONSE_PATTERN,
        help="regex with one capture group for extracting content from API response",
    )
    p.add_argument("--chunk-size", type=int, default=512,
                   help="rows per provider.complete() call (default: 512)")
    p.add_argument(
        "--cache-from", action="append", default=[],
        help="path(s) to existing *_explained.parquet or .json cache to reuse",
    )
    p.add_argument(
        "--cache-export", default=None,
        help="export standalone cache JSON to this path after labeling",
    )
    p.add_argument(
        "--resume", "--no-resume", action=argparse.BooleanOptionalAction, default=True,
        help="skip already-completed chunks on restart (default: --resume)",
    )
    p.add_argument(
        "--max-rows", type=int, default=None,
        help="only process first N rows (for testing)",
    )
    p.add_argument(
        "--debug-dump", default=None,
        help="save raw API responses to this JSON file for debugging dropped rows",
    )
    add_config_arg(p)
    args = apply_config(p)

    assert "{text}" in args.instruction_template, (
        "instruction-template must contain {text} placeholder"
    )

    # Build provider
    provider_kwargs = {
        "max_tokens": args.provider_max_tokens,
        "concurrency": args.provider_concurrency,
    }
    if args.provider_model:
        provider_kwargs["model"] = args.provider_model
    provider: CompletionProvider = resolve_provider(args.provider, provider_kwargs)

    # Load cache
    cache = load_cache(args.cache_from) if args.cache_from else {}

    table = pq.read_table(args.input)
    if args.max_rows:
        table = table.slice(0, args.max_rows)

    # Propagate tokenizer metadata from input to output
    in_meta = table.schema.metadata or {}
    fingerprint = in_meta.get(_TOKENIZER_FINGERPRINT_KEY, b"").decode()
    tokenizer_name = in_meta.get(_TOKENIZER_NAME_KEY, b"").decode()
    if fingerprint:
        print(f"tokenizer fingerprint: {fingerprint[:16]}…  (from {tokenizer_name})")

    out_schema = table.schema.append(pa.field("api_explanation", pa.string()))
    if fingerprint:
        out_schema = out_schema.with_metadata(
            {_TOKENIZER_FINGERPRINT_KEY: fingerprint.encode()}
        )

    # Per-chunk files for crash-safe resumption
    chunks_dir = Path(f"{args.output}.chunks")
    chunks_dir.mkdir(parents=True, exist_ok=True)

    def _process_chunk(chunk: pa.Table) -> tuple[pa.Table, int, list[dict]]:
        texts = chunk.column("detokenized_text_truncated").to_pylist()
        cached_expls = [lookup(cache, t) for t in texts]
        miss_idx = [i for i, e in enumerate(cached_expls) if e is None]
        miss_prompts = [
            args.instruction_template.format(text=texts[i]) for i in miss_idx
        ]
        raw_completions = provider.complete(miss_prompts) if miss_prompts else []
        assert len(raw_completions) == len(miss_prompts), (
            f"provider returned {len(raw_completions)} completions"
            f" for {len(miss_prompts)} prompts — length mismatch"
        )
        miss_cleaned: dict[int, str | None] = {}
        debug_records: list[dict] = []
        for j, raw in zip(miss_idx, raw_completions, strict=True):
            if raw is None:
                miss_cleaned[j] = None
                debug_records.append(
                    {"chunk_row": j, "text": texts[j], "raw": None, "extracted": None,
                     "dropped": True, "drop_reason": "provider returned None (retries exhausted)"}
                )
                continue
            extracted = _extract_and_clean(raw, args.response_extract_pattern)
            miss_cleaned[j] = extracted
            if extracted is None:
                debug_records.append(
                    {"chunk_row": j, "text": texts[j], "raw": raw, "extracted": None,
                     "dropped": True, "drop_reason": "extract pattern did not match"}
                )
            else:
                n_features = extracted.count("\n\n") + 1
                if n_features < _MIN_FEATURES:
                    debug_records.append(
                        {"chunk_row": j, "text": texts[j], "raw": raw, "extracted": extracted,
                         "dropped": True,
                         "drop_reason": f"too few features ({n_features} < {_MIN_FEATURES})"}
                    )
                else:
                    debug_records.append(
                        {"chunk_row": j, "text": texts[j], "raw": raw, "extracted": extracted,
                         "dropped": False, "drop_reason": None}
                    )

        dropped = 0
        keep_mask: list[bool] = []
        explanations: list[str] = []
        for i, hit in enumerate(cached_expls):
            cleaned = hit if hit is not None else miss_cleaned[i]
            if cleaned is None or cleaned.count("\n\n") + 1 < _MIN_FEATURES:
                dropped += 1
                keep_mask.append(False)
                continue
            keep_mask.append(True)
            explanations.append(cleaned)
        if not all(keep_mask):
            chunk = chunk.filter(pa.array(keep_mask, type=pa.bool_()))
        return (
            chunk.append_column("api_explanation", pa.array(explanations, type=pa.string())),
            dropped,
            debug_records,
        )

    dropped_count = 0
    all_debug: list[dict] = []
    chunk_paths: list[Path] = []
    chunk_starts = list(range(0, table.num_rows, args.chunk_size))
    skipped = 0
    for chunk_start in tqdm(chunk_starts, desc="chunks"):
        chunk_path = chunks_dir / f"chunk_{chunk_start:08d}.parquet"
        chunk_paths.append(chunk_path)
        if args.resume and chunk_path.exists():
            skipped += 1
            continue
        chunk_out, dropped, debug_records = _process_chunk(table.slice(chunk_start, args.chunk_size))
        all_debug.extend(debug_records)
        dropped_count += dropped
        tmp = chunk_path.with_suffix(".tmp")
        pq.write_table(chunk_out, tmp)
        tmp.rename(chunk_path)
    if skipped:
        print(f"  resumed: skipped {skipped}/{len(chunk_starts)} already-completed chunks")

    # Merge chunks into final output via streaming ParquetWriter
    row_count = 0
    with pq.ParquetWriter(args.output, out_schema) as writer:
        for p in chunk_paths:
            t = pq.read_table(p)
            writer.write_table(t)
            row_count += t.num_rows

    print(f"wrote {row_count} rows → {args.output}")
    if dropped_count > 0:
        print(f"  DROPPED {dropped_count} rows (response didn't match extract pattern)")

    assert row_count > 0, (
        f"ALL {dropped_count} rows dropped — either responses didn't match "
        f"--response-extract-pattern={args.response_extract_pattern!r} (truncated?), "
        f"or had fewer than {_MIN_FEATURES} features after cleanup. "
        f"Use --debug-dump debug.json to inspect raw API responses."
    )

    # Export debug dump if requested
    if args.debug_dump:
        with open(args.debug_dump, "w") as f:
            json.dump(all_debug, f, ensure_ascii=False, indent=2)
        n_failed = sum(1 for d in all_debug if d["dropped"])
        print(f"debug dump: {len(all_debug)} records ({n_failed} dropped) → {args.debug_dump}")

    # Export standalone cache if requested
    if args.cache_export:
        # Merge newly generated explanations into cache before export
        final_table = pq.read_table(args.output)
        texts = final_table.column("detokenized_text_truncated").to_pylist()
        expls = final_table.column("api_explanation").to_pylist()
        merged = dict(cache)  # start with incoming cache
        for txt, expl in zip(texts, expls, strict=True):
            merged[hashlib.sha256(txt.encode()).hexdigest()] = expl
        export_cache_json(
            merged, args.cache_export,
            fingerprint=fingerprint, tokenizer_name=tokenizer_name,
        )


if __name__ == "__main__":
    main()
