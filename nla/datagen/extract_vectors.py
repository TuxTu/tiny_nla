"""GPU hidden-state extraction — forward a model, grab vectors at sampled positions.

Model-specific step of the decoupled NLA pipeline.  Reads positions/explained
parquets from the labeling pipeline, loads a model on GPU, and extracts
hidden-state vectors at the pre-determined (doc_id, n_raw_tokens) positions.

One forward pass per document — causal attention means hidden_states[pos]
is identical whether the model sees 100 or 1000 tokens.

Output schema (matches original stage3_build input):

    doc_id              string
    n_raw_tokens        int64        1-indexed position
    activation_vector   fixed_size_list<float32, d_model>
    activation_layer    int64        layer index
"""

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.datagen._common import add_config_arg, apply_config
from nla.training.resolve import resolve_parquet


def _resolve_text_config(config):
    """Unwrap multimodal config wrapper (Gemma-3 etc.) to get text hidden_size."""
    for attr in ("text_config",):
        nested = getattr(config, attr, None)
        if nested is not None:
            return nested
    return config


def _resolve_decoder_layers(model) -> torch.nn.ModuleList:
    """Find the ModuleList of decoder layers, unwrapping multimodal wrappers."""
    if hasattr(model, "language_model"):
        model = model.language_model
    if hasattr(model, "model"):
        layers = model.model.layers
    elif hasattr(model, "transformer"):
        layers = model.transformer.h
    else:
        raise AssertionError(
            f"Cannot find decoder layers in {type(model).__name__}. "
            f"Expected model.model.layers or model.transformer.h."
        )
    assert isinstance(layers, torch.nn.ModuleList), (
        f"Decoder layers must be ModuleList, got {type(layers).__name__}"
    )
    return layers


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True,
                   help="explained/positions parquet from labeling pipeline")
    p.add_argument("--model-name", required=True,
                   help="HF model name, e.g. Qwen/Qwen3-4B")
    p.add_argument("--layer-index", type=int, default=None,
                   help="transformer layer for extraction (default: 2/3 * num_layers)")
    p.add_argument("--output", required=True, help="output vectors parquet path")
    p.add_argument("--batch-size", type=int, default=8,
                   help="docs per forward pass (default: 8)")
    p.add_argument("--max-length", type=int, default=2048,
                   help="max tokens per document (default: 2048)")
    p.add_argument("--device", default=None,
                   help="device override: cuda, mps, cpu (default: auto-detect)")
    p.add_argument("--gpus", type=int, default=None,
                   help="number of GPUs for data-parallel extraction (default: 1)")
    p.add_argument("--shard-id", type=int, default=None,
                   help="shard index for manual splitting (0-indexed)")
    p.add_argument("--num-shards", type=int, default=None,
                   help="total shards for manual splitting")
    add_config_arg(p)
    args = apply_config(p)

    # ---- detect device -----------------------------------------------------
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    use_device_map = device.type == "cuda"
    print(f"device: {device.type}")

    # ---- load model and tokenizer ------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    model_kwargs = dict(dtype=torch.bfloat16)
    if use_device_map:
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, **model_kwargs,
    ).eval()
    if not use_device_map:
        model = model.to(device)

    d_model = _resolve_text_config(model.config).hidden_size
    layers = _resolve_decoder_layers(model)
    num_layers = len(layers)
    layer_index = args.layer_index if args.layer_index is not None else (2 * num_layers) // 3

    assert 0 <= layer_index < num_layers, (
        f"layer_index={layer_index} out of range for model with {num_layers} layers"
    )

    print(f"model: {args.model_name}")
    print(f"  d_model={d_model}  layers={num_layers}  layer_index={layer_index}")

    # ---- read positions ----------------------------------------------------
    table = pq.read_table(resolve_parquet(args.input))
    assert "doc_id" in table.column_names, "input must have doc_id column"
    assert "n_raw_tokens" in table.column_names, "input must have n_raw_tokens column"
    assert "detokenized_text_truncated" in table.column_names, (
        "input must have detokenized_text_truncated column"
    )

    # Group by doc_id: for each doc, keep the longest prefix for the forward
    # pass, then slice at each position.
    docs: dict[str, dict] = {}
    for row in table.to_pylist():
        did = row["doc_id"]
        nrt = row["n_raw_tokens"]
        if did not in docs:
            docs[did] = {"positions": [], "max_nrt": 0, "longest_text": ""}
        docs[did]["positions"].append(nrt)
        if nrt > docs[did]["max_nrt"]:
            docs[did]["max_nrt"] = nrt
            docs[did]["longest_text"] = row["detokenized_text_truncated"]

    doc_items = sorted(docs.items(), key=lambda x: x[0])  # deterministic order
    print(f"positions: {table.num_rows} rows across {len(docs)} unique docs")

    # ---- sharding -----------------------------------------------------------
    num_gpus = getattr(args, "gpus", None) or 1
    if num_gpus > 1 and not getattr(args, "shard_id", None):
        # Multi-GPU: spawn one process per GPU, each handles a slice of docs.
        available = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if available == 0:
            raise RuntimeError("--gpus requires CUDA devices")
        if num_gpus > available:
            print(f"warning: --gpus {num_gpus} > available GPUs ({available}), "
                  f"using {available}")
            args.gpus = available
        _run_multi_gpu(args, doc_items, d_model, layer_index)
        return

    shard_id = getattr(args, "shard_id", None) or 0
    num_shards = getattr(args, "num_shards", None) or 1
    shard_slice = slice(
        shard_id * len(doc_items) // num_shards,
        (shard_id + 1) * len(doc_items) // num_shards if shard_id < num_shards - 1 else len(doc_items),
    )
    if num_shards > 1:
        shard_output = f"{args.output}.shard_{shard_id:03d}"
    else:
        shard_output = args.output

    # ---- run extraction on this shard ---------------------------------------
    norms, n_written, n_skipped = _extract_shard(
        doc_items[shard_slice],
        model=model,
        tokenizer=tokenizer,
        layers=layers,
        layer_index=layer_index,
        d_model=d_model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        output_path=shard_output,
        shard_label=f"gpu {shard_id}" if num_shards > 1 else "extracting",
    )

    # Multi-GPU handled by _run_multi_gpu which prints stats after merge.
    if num_gpus > 1 and not getattr(args, "shard_id", None):
        return

    print(f"wrote {n_written} rows → {shard_output}")
    if n_skipped:
        print(f"  skipped {n_skipped} positions (token count mismatch after re-tokenization)")
    _print_norm_stats(norms)


# ---------------------------------------------------------------------------
# extraction logic (shared by single-GPU and multi-GPU)
# ---------------------------------------------------------------------------


def _extract_shard(
    doc_items: list,
    *,
    model,
    tokenizer,
    layers,
    layer_index: int,
    d_model: int,
    batch_size: int,
    max_length: int,
    output_path: str,
    shard_label: str = "extracting",
) -> tuple[list[float], int, int]:
    """Run extraction on a slice of docs, write to output_path."""
    captured: torch.Tensor | None = None

    def _hook(_module, _inputs, output):
        nonlocal captured
        h = output[0] if isinstance(output, tuple) else output
        captured = h.detach().clone()

    handle = layers[layer_index].register_forward_hook(_hook)

    schema = pa.schema([
        ("doc_id", pa.string()),
        ("n_raw_tokens", pa.int64()),
        ("activation_vector", pa.list_(pa.float32(), d_model)),
        ("activation_layer", pa.int64()),
    ])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skipped = 0
    norms: list[float] = []
    device = model.get_input_embeddings().weight.device

    with pq.ParquetWriter(output_path, schema) as writer:
        pbar = tqdm(total=len(doc_items), desc=shard_label)
        i = 0
        while i < len(doc_items):
            batch_items = doc_items[i:i + batch_size]
            texts = [item[1]["longest_text"] for item in batch_items]

            enc = tokenizer(
                texts, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
                add_special_tokens=True,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            captured = None
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            assert captured is not None, (
                f"hook on layer {layer_index} did not fire — wrong architecture?"
            )

            hidden = captured.float().cpu()
            lengths = attention_mask.sum(dim=1).cpu()

            rows: dict[str, list] = {k: [] for k in schema.names}
            for j, (did, info) in enumerate(batch_items):
                doc_hidden = hidden[j, :int(lengths[j].item())]
                for nrt in info["positions"]:
                    pos = nrt - 1  # 1-indexed → 0-indexed
                    if pos >= doc_hidden.shape[0]:
                        n_skipped += 1
                        continue
                    vec = doc_hidden[pos].clone()
                    norms.append(float(torch.linalg.norm(vec).item()))
                    rows["doc_id"].append(did)
                    rows["n_raw_tokens"].append(nrt)
                    rows["activation_vector"].append(vec.tolist())
                    rows["activation_layer"].append(layer_index)

            writer.write_table(pa.Table.from_pydict(rows, schema=schema))
            n_written += len(rows["doc_id"])
            i += len(batch_items)
            pbar.update(len(batch_items))

            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

    handle.remove()
    pbar.close()
    return norms, n_written, n_skipped


def _run_multi_gpu(args, doc_items: list, d_model: int, layer_index: int) -> None:
    """Spawn one process per GPU, merge shard outputs."""
    import torch.multiprocessing as mp

    num_gpus = args.gpus

    def _worker(gpu_id: int) -> None:
        """Run extraction on a single GPU slice."""
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Build args namespace for this worker
        from argparse import Namespace
        w_args = Namespace(
            input=args.input, model_name=args.model_name,
            layer_index=args.layer_index, output=args.output,
            batch_size=args.batch_size, max_length=args.max_length,
            device=f"cuda:0",  # CUDA_VISIBLE_DEVICES remaps to physical GPU
            gpus=None, shard_id=gpu_id, num_shards=num_gpus,
            config=None,
        )
        _run_shard(w_args, doc_items, d_model, layer_index)

    print(f"launching {num_gpus} GPU workers ...")
    mp.spawn(_worker, nprocs=num_gpus, join=True)

    # Merge shard parquets
    print("merging shard outputs ...")
    tables = []
    for gid in range(num_gpus):
        shard_path = f"{args.output}.shard_{gid:03d}"
        t = pq.read_table(shard_path)
        tables.append(t)
        Path(shard_path).unlink()
    merged = pa.concat_tables(tables)
    pq.write_table(merged, args.output)
    print(f"merged {merged.num_rows} rows → {args.output}")

    # Print norm stats
    col = merged.column("activation_vector")
    norms = [float(torch.linalg.norm(torch.tensor(v.as_py())).item()) for v in col]
    _print_norm_stats(norms)


def _run_shard(args, doc_items: list, d_model: int, layer_index: int) -> None:
    """Load model on this GPU and extract vectors for our shard."""
    # Detect device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    use_device_map = device.type == "cuda"

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    model_kwargs = dict(dtype=torch.bfloat16)
    if use_device_map:
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, **model_kwargs,
    ).eval()
    if not use_device_map:
        model = model.to(device)

    layers = _resolve_decoder_layers(model)

    shard_id = args.shard_id or 0
    num_shards = args.num_shards or 1
    shard_slice = slice(
        shard_id * len(doc_items) // num_shards,
        (shard_id + 1) * len(doc_items) // num_shards
        if shard_id < num_shards - 1 else len(doc_items),
    )
    shard_output = f"{args.output}.shard_{shard_id:03d}"

    norms, n_written, n_skipped = _extract_shard(
        doc_items[shard_slice],
        model=model, tokenizer=tokenizer, layers=layers,
        layer_index=layer_index, d_model=d_model,
        batch_size=args.batch_size, max_length=args.max_length,
        output_path=shard_output,
        shard_label=f"gpu {shard_id}/{num_shards}",
    )
    print(f"[gpu {shard_id}] wrote {n_written} rows (skipped {n_skipped})")


def _print_norm_stats(norms: list[float]) -> None:
    """Print L2-norm statistics for extracted vectors."""
    if not norms:
        return
    mean = sum(norms) / len(norms)
    variance = sum((n - mean) ** 2 for n in norms) / len(norms)
    std = variance ** 0.5
    print(f"\nvector L2-norm stats:")
    print(f"  mean: {mean:.2f}  std: {std:.2f}  min: {min(norms):.2f}  max: {max(norms):.2f}")
    scale = round(mean, -1) if mean >= 10 else round(mean, 1)
    print(f"  recommended injection_scale: {scale:.1f}  (round value near mean norm)")


if __name__ == "__main__":
    main()
