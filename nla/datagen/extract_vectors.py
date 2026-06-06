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

    # ---- forward hook ------------------------------------------------------
    captured: torch.Tensor | None = None

    def _hook(_module, _inputs, output):
        nonlocal captured
        h = output[0] if isinstance(output, tuple) else output
        captured = h.detach().clone()

    handle = layers[layer_index].register_forward_hook(_hook)

    # ---- output schema -----------------------------------------------------
    schema = pa.schema([
        ("doc_id", pa.string()),
        ("n_raw_tokens", pa.int64()),
        ("activation_vector", pa.list_(pa.float32(), d_model)),
        ("activation_layer", pa.int64()),
    ])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_skipped = 0
    norms: list[float] = []

    with pq.ParquetWriter(args.output, schema) as writer:
        pbar = tqdm(total=len(docs), desc="extracting vectors")
        i = 0
        while i < len(docs):
            batch_items = doc_items[i:i + args.batch_size]
            texts = [item[1]["longest_text"] for item in batch_items]

            enc = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
                add_special_tokens=True,
            )
            device = model.get_input_embeddings().weight.device
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            captured = None
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            assert captured is not None, (
                f"hook on layer {layer_index} did not fire — wrong architecture?"
            )

            hidden = captured.float().cpu()  # [batch, max_seq, d_model]
            lengths = attention_mask.sum(dim=1).cpu()  # actual seq lengths

            rows: dict[str, list] = {k: [] for k in schema.names}
            for j, (did, info) in enumerate(batch_items):
                seq_len = int(lengths[j].item())
                doc_hidden = hidden[j, :seq_len]  # unpacked [seq, d_model]

                for nrt in info["positions"]:
                    pos = nrt - 1  # n_raw_tokens is 1-indexed
                    if pos >= doc_hidden.shape[0]:
                        n_skipped += 1
                        continue
                    vec = doc_hidden[pos].clone()
                    nrm = float(torch.linalg.norm(vec).item())
                    norms.append(nrm)
                    rows["doc_id"].append(did)
                    rows["n_raw_tokens"].append(nrt)
                    rows["activation_vector"].append(vec.tolist())
                    rows["activation_layer"].append(layer_index)

            writer.write_table(pa.Table.from_pydict(rows, schema=schema))
            n_written += len(rows["doc_id"])
            i += len(batch_items)
            pbar.update(len(batch_items))

            # Free MPS/CUDA cache between batches to prevent accumulation.
            # MPS in particular does not auto-release cached allocations.
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

    handle.remove()
    pbar.close()

    print(f"wrote {n_written} rows → {args.output}")
    if n_skipped:
        print(f"  skipped {n_skipped} positions (token count mismatch after re-tokenization)")

    # ---- injection scale recommendation ------------------------------------
    if norms:
        mean = sum(norms) / len(norms)
        std = (sum((n - mean) ** 2 for n in norms) / len(norms)) ** 0.5
        print(f"\nvector L2-norm stats:")
        print(f"  mean: {mean:.2f}  std: {std:.2f}  min: {min(norms):.2f}  max: {max(norms):.2f}")
        scale = round(mean, -1) if mean >= 10 else round(mean, 1)
        print(f"  recommended injection_scale: {scale:.1f}  (round value near mean norm)")


if __name__ == "__main__":
    main()
