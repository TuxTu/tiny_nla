"""Pipeline orchestrator — extract → label → split, with a shared growing pool.

Pool architecture
-----------------
Three independent buckets (av, ar, rl) under pool_dir/. Each bucket
contains positions.parquet; av and ar additionally have explained.parquet.
When a model needs more docs in a bucket, only that bucket grows — new docs
are extracted and assigned where there's deficit.  When a model needs fewer
docs, a deterministic seed-based subset is drawn from each bucket without
modifying the pool.

Buckets grow monotonically.  Split assignment is permanent — a doc once in
the AV bucket stays there, so labels are never wasted.

Pool layout::

    pool_dir/
      pool_meta.json              # tokenizer fingerprint, corpus_next_start
      av/
        positions.parquet
        explained.parquet
        explained.parquet.chunks/  # crash-resume state (api_explain)
      ar/
        positions.parquet
        explained.parquet
        explained.parquet.chunks/
      rl/
        positions.parquet          # RL never gets API-labeled

Model output::

    output_dir/
      av_sft_explained.parquet
      ar_sft_explained.parquet
      rl.parquet                   # positions only, no explanations

Each stage can also be run individually (the orchestrator shells out), so
the pipeline is inspectable and resumable at every step.
"""

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
from transformers import AutoTokenizer

from nla import (
    _TOKENIZER_FINGERPRINT_KEY,
    _TOKENIZER_NAME_KEY,
    compute_tokenizer_fingerprint,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def _opt(args: list[str], flag: str, value: Any) -> None:
    if value is not None:
        args.extend([flag, str(value)])


def _paths(pool_dir: str, output_dir: str) -> tuple[dict, dict]:
    """Derive pool bucket paths and model output paths."""
    pool = {
        "meta":     f"{pool_dir}/pool_meta.json",
        "av":       {"positions":  f"{pool_dir}/av/positions.parquet",
                      "explained": f"{pool_dir}/av/explained.parquet"},
        "ar":       {"positions":  f"{pool_dir}/ar/positions.parquet",
                      "explained": f"{pool_dir}/ar/explained.parquet"},
        "rl":       {"positions":  f"{pool_dir}/rl/positions.parquet"},
    }
    out = {
        "av": f"{output_dir}/av_sft_explained.parquet",
        "ar": f"{output_dir}/ar_sft_explained.parquet",
        "rl": f"{output_dir}/rl.parquet",
    }
    return pool, out


def _bucket_doc_count(parquet_path: str) -> int:
    """Number of unique doc_ids in a parquet file.  Returns 0 if missing."""
    if not os.path.exists(parquet_path):
        return 0
    pf = pq.ParquetFile(parquet_path)
    ids = pf.read(columns=["doc_id"]).column("doc_id").to_pylist()
    return len(set(ids))


def _load_pool_meta(pool_dir: str) -> dict:
    path = Path(pool_dir) / "pool_meta.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_pool_meta(pool_dir: str, meta: dict) -> None:
    Path(pool_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(pool_dir) / "pool_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def _verify_fingerprint(pool_dir: str, tokenizer) -> None:
    """Abort if pool was created with a different tokenizer family."""
    meta = _load_pool_meta(pool_dir)
    stored = meta.get("tokenizer_fingerprint")
    if stored is None:
        # First run — record fingerprint
        fingerprint = compute_tokenizer_fingerprint(tokenizer)
        _save_pool_meta(pool_dir, {
            "tokenizer_fingerprint": fingerprint,
            "tokenizer_name": tokenizer.name_or_path,
            "corpus_next_start": 0,
        })
        print(f"tokenizer fingerprint: {fingerprint[:16]}…  ({tokenizer.name_or_path})")
        print("new pool — first run")
        return
    fingerprint = compute_tokenizer_fingerprint(tokenizer)
    if fingerprint != stored:
        raise RuntimeError(
            f"Tokenizer fingerprint mismatch!\n"
            f"  Pool was created with: {meta.get('tokenizer_name', 'unknown')}\n"
            f"  Current tokenizer:     {tokenizer.name_or_path}\n"
            f"  Current fingerprint:   {fingerprint[:16]}...\n"
            f"\nDelete {pool_dir}/pool_meta.json and re-extract,"
            f" or use a matching tokenizer."
        )
    print(f"tokenizer fingerprint verified: {fingerprint[:16]}…")


# ---------------------------------------------------------------------------
# stage: extract new docs and assign to buckets
# ---------------------------------------------------------------------------


def _extract_and_assign(cfg: dict, pool: dict, needs: dict) -> None:
    """Extract new docs from corpus, assign to buckets with deficit.

    Computes deficit per bucket.  If any bucket is short, extracts a batch
    of new documents, deterministically shuffles their doc_ids, and assigns
    them to the buckets that need them.  Extra docs (margin for short/skipped
    documents) go to RL.
    """
    # how many docs are already in each pool bucket
    have = {
        "av": _bucket_doc_count(pool["av"]["positions"]),
        "ar": _bucket_doc_count(pool["ar"]["positions"]),
        "rl": _bucket_doc_count(pool["rl"]["positions"]),
    }
    deficit = {
        k: max(0, needs[k] - have[k])
        for k in ("av", "ar", "rl")
    }
    total_deficit = sum(deficit.values())

    if total_deficit == 0:
        print(f"all buckets filled — skipping extraction")
        print(f"  av={have['av']}/{needs['av']}  ar={have['ar']}/{needs['ar']}  rl={have['rl']}/{needs['rl']}")
        return

    print(f"bucket deficits: av={deficit['av']}  ar={deficit['ar']}  rl={deficit['rl']}  total={total_deficit}")

    # Extract exactly the deficit — no margin.  Short docs get skipped and
    # any resulting shortfall is filled on the next pipeline run (matching
    # the original repo's approach of accepting whatever the corpus yields).
    extract_count = total_deficit

    meta = _load_pool_meta(cfg["pool_dir"])
    corpus_start = meta.get("corpus_next_start", cfg["corpus"].get("start", 0))

    batch_path = f"{cfg['pool_dir']}/_batch.parquet"
    cmd = [
        sys.executable, "-m", "nla.datagen.extract_positions",
        "--tokenizer-name", cfg["tokenizer_name"],
        "--corpus", cfg["corpus"]["name"],
        "--corpus-split", cfg["corpus"].get("split", "train"),
        "--corpus-start", str(corpus_start),
        "--corpus-length", str(extract_count),
        "--text-column", cfg.get("text_column", "text"),
        "--positions-per-doc", str(cfg["positions_per_doc"]),
        "--seed", str(cfg["seed"]),
        "--min-position", str(cfg["min_position"]),
        "--max-length", str(cfg.get("max_length", 2048)),
        "--output", batch_path,
    ]
    _opt(cmd, "--corpus-config", cfg["corpus"].get("config"))
    _run(cmd)

    # update pool meta for next extraction
    meta["corpus_next_start"] = corpus_start + extract_count
    _save_pool_meta(cfg["pool_dir"], meta)

    # read new docs and assign
    batch = pq.read_table(batch_path)
    doc_ids = sorted(set(batch.column("doc_id").to_pylist()))
    rng = random.Random(cfg["seed"])
    rng.shuffle(doc_ids)

    n_actual = len(doc_ids)
    print(f"extracted {n_actual} unique docs (asked for {extract_count})")

    # assign: fill deficits.  Short docs were already skipped by extract_positions,
    # so actual doc count may be slightly less than deficit — remainder is filled
    # on the next pipeline run.
    assigned: dict[str, set[str]] = {"av": set(), "ar": set(), "rl": set()}
    i = 0
    for bucket in ("av", "ar", "rl"):
        take = min(deficit[bucket], len(doc_ids) - i)
        assigned[bucket] = set(doc_ids[i:i + take])
        i += take

    # write each bucket
    for bucket_name, ids in assigned.items():
        if not ids:
            continue
        mask = pc.is_in(
            batch.column("doc_id"),
            value_set=pa.array(sorted(ids), type=pa.string()),
        )
        subset = batch.filter(mask)

        pos_path = pool[bucket_name]["positions"]
        Path(pos_path).parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(pos_path):
            old = pq.read_table(pos_path)
            subset = pa.concat_tables([old, subset])
        pq.write_table(subset, pos_path)
        print(f"  pool/{bucket_name}/positions: +{len(ids)} docs → {len(ids) + have[bucket_name]} total")

    os.remove(batch_path)


# ---------------------------------------------------------------------------
# stage: label AV and AR buckets
# ---------------------------------------------------------------------------


def _label_buckets(cfg: dict, pool: dict) -> None:
    """Label (or resume labeling) AV and AR buckets via api_explain.

    Calls api_explain with --resume so existing chunks are skipped.
    Only new rows trigger API calls.  RL is never labeled.
    """
    provider = cfg.get("provider", {})

    for bucket_name in ("av", "ar"):
        pos_path = pool[bucket_name]["positions"]
        expl_path = pool[bucket_name]["explained"]

        if not os.path.exists(pos_path):
            print(f"pool/{bucket_name}/positions not found — skipping")
            continue

        Path(expl_path).parent.mkdir(parents=True, exist_ok=True)

        n_pos = pq.ParquetFile(pos_path).metadata.num_rows
        n_expl = pq.ParquetFile(expl_path).metadata.num_rows if os.path.exists(expl_path) else 0
        if n_expl >= n_pos:
            print(f"pool/{bucket_name}: {n_expl}/{n_pos} labeled — skipping")
            continue

        print(f"pool/{bucket_name}: {n_expl}/{n_pos} labeled — resuming")
        cmd = [
            sys.executable, "-m", "nla.datagen.api_explain",
            "--input", pos_path,
            "--output", expl_path,
            "--provider", provider.get("name", "deepseek"),
            "--resume",
        ]
        _opt(cmd, "--provider-model", provider.get("model"))
        _opt(cmd, "--provider-max-tokens", provider.get("max_tokens"))
        _opt(cmd, "--provider-concurrency", provider.get("concurrency"))
        _run(cmd)


# ---------------------------------------------------------------------------
# stage: deterministic subsample from pool for model output
# ---------------------------------------------------------------------------


def _model_output(cfg: dict, pool: dict, out: dict, needs: dict) -> None:
    """Write model-specific output from pool via deterministic doc selection.

    If a bucket has more docs than needed, a seed-based subset is drawn.
    If it has exactly what's needed (or fewer), all docs are used.
    """
    seed = cfg["seed"]

    for bucket_name in ("av", "ar", "rl"):
        pos_path = pool[bucket_name]["positions"]
        expl_path = pool[bucket_name].get("explained")
        out_path = out[bucket_name]

        if not os.path.exists(pos_path):
            print(f"pool/{bucket_name}/positions not found — skipping")
            continue

        # source: explained for av/ar, positions for rl
        src_path = expl_path if expl_path and os.path.exists(expl_path) else pos_path
        table = pq.read_table(src_path)
        all_ids = sorted(set(table.column("doc_id").to_pylist()))
        need_count = needs[bucket_name]

        if len(all_ids) <= need_count:
            # use all
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, out_path)
            print(f"  {bucket_name}: {len(all_ids)} docs ({table.num_rows} rows) → {out_path}")
        else:
            # deterministic downsample
            rng = random.Random(seed)
            rng.shuffle(all_ids)
            selected = set(all_ids[:need_count])
            mask = pc.is_in(
                table.column("doc_id"),
                value_set=pa.array(sorted(selected), type=pa.string()),
            )
            subset = table.filter(mask)
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(subset, out_path)
            print(f"  {bucket_name}: sampled {need_count}/{len(all_ids)} docs ({subset.num_rows} rows) → {out_path}")


# ---------------------------------------------------------------------------
# stage: GPU hidden-state extraction for a specific model
# ---------------------------------------------------------------------------


def _extract_vectors(cfg: dict, out: dict) -> None:
    """Extract hidden-state vectors from a model at pre-determined positions.

    Reads the output files from the [output] stage and runs a GPU forward
    pass to grab hidden_states[pos] at each (doc_id, n_raw_tokens) position.
    One forward pass per document — causal attention means all positions in
    the same doc are extracted from the same forward pass.
    """
    model_name = cfg.get("model_name")
    if not model_name:
        print("no model_name in config — skipping vector extraction")
        return

    layer_index = cfg.get("layer_index")  # optional, auto-detected if omitted
    batch_size = cfg.get("batch_size", 8)
    max_length = cfg.get("max_length", 2048)

    for side, out_path in out.items():
        if not os.path.exists(out_path):
            print(f"{out_path} not found — skipping")
            continue

        vec_path = out_path.replace(".parquet", "_vectors.parquet")
        if os.path.exists(vec_path):
            print(f"{vec_path} exists — skipping")
            continue
        cmd = [
            sys.executable, "-m", "nla.datagen.extract_vectors",
            "--input", out_path,
            "--model-name", model_name,
            "--output", vec_path,
            "--batch-size", str(batch_size),
            "--max-length", str(max_length),
        ]
        if layer_index is not None:
            cmd.extend(["--layer-index", str(layer_index)])
        _run(cmd)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

_STAGES = {
    "extract": "extract_and_assign",
    "explain": "label_buckets",
    "output":  "model_output",
    "vectors": "extract_vectors",
}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True, help="YAML pipeline config")
    p.add_argument(
        "--stages", default=None,
        help="comma-separated stages: extract,explain,output  (default: all)",
    )
    p.add_argument(
        "--override", nargs="*", default=[],
        help="dotted-key overrides, e.g. 'num_docs=10000 split.av_sft=0.3'",
    )
    args_in = p.parse_args()

    # ---- load config -------------------------------------------------------
    with open(args_in.config) as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict), f"config must be a mapping, got {type(cfg).__name__}"

    # apply --override
    for ov in args_in.override:
        key, _, val = ov.partition("=")
        d = cfg
        *path, leaf = key.split(".")
        for k in path:
            d = d.setdefault(k, {})
        d[leaf] = yaml.safe_load(val)

    # defaults
    cfg.setdefault("seed", 42)
    cfg.setdefault("min_position", 50)
    cfg.setdefault("positions_per_doc", 10)
    cfg.setdefault("text_column", "text")
    cfg.setdefault("corpus", {}).setdefault("split", "train")
    cfg.setdefault("split", {"av_sft": 0.25, "ar_sft": 0.25, "rl": 0.50})
    cfg.setdefault("provider", {}).setdefault("name", "deepseek")

    # ---- fingerprint check -------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_name"])
    _verify_fingerprint(cfg["pool_dir"], tokenizer)

    # ---- derived values ----------------------------------------------------
    num_docs = cfg["num_docs"]
    split_cfg = cfg["split"]
    needs = {
        "av": int(num_docs * split_cfg["av_sft"]),
        "ar": int(num_docs * split_cfg["ar_sft"]),
        "rl": num_docs
        - int(num_docs * split_cfg["av_sft"])
        - int(num_docs * split_cfg["ar_sft"]),
    }

    pool, out = _paths(cfg["pool_dir"], cfg["output_dir"])

    have_av = _bucket_doc_count(pool["av"]["positions"])
    have_ar = _bucket_doc_count(pool["ar"]["positions"])
    have_rl = _bucket_doc_count(pool["rl"]["positions"])

    print(f"=== pipeline: {args_in.config} ===")
    print(f"  tokenizer: {cfg['tokenizer_name']}")
    print(f"  pool_dir:  {cfg['pool_dir']}")
    print(f"  output:    {cfg['output_dir']}")
    print(f"  pool:  av={have_av}  ar={have_ar}  rl={have_rl}")
    print(f"  needs: av={needs['av']}  ar={needs['ar']}  rl={needs['rl']}  (total={num_docs} docs)")

    # ---- stages ------------------------------------------------------------
    stages = args_in.stages.split(",") if args_in.stages else list(_STAGES)
    for s in stages:
        assert s in _STAGES, f"unknown stage {s!r}, valid: {sorted(_STAGES)}"

    if "extract" in stages:
        print(f"\n{'='*20} EXTRACT {'='*20}")
        _extract_and_assign(cfg, pool, needs)

    if "explain" in stages:
        print(f"\n{'='*20} EXPLAIN {'='*20}")
        _label_buckets(cfg, pool)

    if "output" in stages:
        print(f"\n{'='*20} OUTPUT {'='*20}")
        _model_output(cfg, pool, out, needs)

    if "vectors" in stages:
        print(f"\n{'='*20} VECTORS {'='*20}")
        _extract_vectors(cfg, out)

    print(f"\n=== done ===")
    for bucket_name, path in out.items():
        if os.path.exists(path):
            n = pq.ParquetFile(path).metadata.num_rows
            print(f"  {path}  ({n} rows)")
        else:
            print(f"  {path}  (MISSING)")


if __name__ == "__main__":
    main()
