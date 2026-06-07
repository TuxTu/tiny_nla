#!/usr/bin/env python3
"""NLA inference: combined actor (AV → explanation) + critic (explanation → AR vector).

Modes:
  --text "Your text here"  : extract AV from base model, then full NLA pipeline
  --av-file vector.npy     : skip extraction, provide AV directly
  --interactive            : type AV vectors as comma-separated floats
  (no input)               : demo mode with random AV

Usage:
  # Extract AV from text and run full pipeline:
  python scripts/nla_infer.py --text "The capital of France is"

  # Custom model:
  python scripts/nla_infer.py \\
      --model-name Qwen/Qwen3-4B \\
      --actor-ckpt checkpoints/actor_sft \\
      --critic-ckpt checkpoints/critic_sft \\
      --sidecar data/av_sft_train.parquet.nla_meta.yaml \\
      --av-file vector.npy

  # 0.6B model:
  python scripts/nla_infer.py \\
      --model-name Qwen/Qwen3-0.6B \\
      --actor-ckpt checkpoints/actor_sft_0.6B \\
      --critic-ckpt checkpoints/critic_sft_0.6B \\
      --sidecar data/av_sft_train_0.6B.parquet.nla_meta.yaml \\
      --interactive

  # Interactive mode:
  python scripts/nla_infer.py --interactive

  # Save results:
  python scripts/nla_infer.py --av-file vector.npy --output result.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJ = Path(__file__).resolve().parent.parent

# ── Defaults (4B model) ─────────────────────────────────────────────────
DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B"
DEFAULT_ACTOR_CKPT = str(PROJ / "checkpoints/actor_sft")
DEFAULT_CRITIC_CKPT = str(PROJ / "checkpoints/critic_sft")
DEFAULT_SIDECAR = str(PROJ / "data/av_sft_train.parquet.nla_meta.yaml")


def load_models(
    model_name: str = DEFAULT_MODEL_NAME,
    actor_ckpt: str = DEFAULT_ACTOR_CKPT,
    critic_ckpt: str = DEFAULT_CRITIC_CKPT,
    sidecar_path: str = DEFAULT_SIDECAR,
    device: str = "cuda",
):
    """Load actor and critic models from SFT checkpoints.

    Parameters
    ----------
    model_name : HF model ID for tokenizer config (e.g. Qwen/Qwen3-4B)
    actor_ckpt : path to actor SFT checkpoint (saved via save_pretrained)
    critic_ckpt : path to critic SFT checkpoint (saved via save_pretrained)
    sidecar_path : path to .nla_meta.yaml with injection token metadata
    device : "cuda" or "cpu"
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from nla.training.models import NLACriticModel
    from nla.training.sidecar import read_sidecar

    print(f"Loading actor from {actor_ckpt} ...")
    actor = AutoModelForCausalLM.from_pretrained(
        actor_ckpt,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    actor.eval()

    print(f"Loading critic from {critic_ckpt} ...")
    critic = NLACriticModel.from_pretrained(
        critic_ckpt,
        torch_dtype=torch.bfloat16,
    )
    critic = critic.to(device)
    critic.eval()

    # Tokenizer from actor checkpoint (has chat template), fall back to model_name
    try:
        tokenizer = AutoTokenizer.from_pretrained(actor_ckpt)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load injection metadata from sidecar.
    # read_sidecar appends .nla_meta.yaml, so strip if already present.
    if sidecar_path.endswith(".nla_meta.yaml"):
        sidecar_path = sidecar_path[: -len(".nla_meta.yaml")]
    sidecar = read_sidecar(sidecar_path)
    tokens = sidecar["tokens"]
    d_model = sidecar["extraction"]["d_model"]

    model_cfg = actor.config
    print(f"Models loaded. model={model_name}, layers={model_cfg.num_hidden_layers}, "
          f"d_model={d_model}, injection_char={tokens['injection_char']}")
    return actor, critic, tokenizer, tokens, d_model


def extract_av(
    text: str,
    model_name: str = DEFAULT_MODEL_NAME,
    layer_index: int | None = None,
    device: str = "cuda",
) -> tuple[np.ndarray, int]:
    """Extract activation vector from a base model at a given layer.

    Runs the text through the base model, hooks into layer ``layer_index``
    (default: 2/3 of total layers), and captures the hidden state at the
    last token position.

    Returns
    -------
    (av_vector, layer_index) — av_vector is [d_model] float32
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model {model_name} for extraction ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    base_model.eval()

    # Resolve layer index
    if hasattr(base_model, "language_model"):
        inner = base_model.language_model
    else:
        inner = base_model
    if hasattr(inner, "model"):
        layers = inner.model.layers
    elif hasattr(inner, "transformer"):
        layers = inner.transformer.h
    else:
        raise RuntimeError(f"Cannot find decoder layers in {type(base_model).__name__}")

    num_layers = len(layers)
    if layer_index is None:
        layer_index = (2 * num_layers) // 3
    d_model = layers[0].weight.shape[1] if hasattr(layers[0], "weight") else base_model.config.hidden_size

    print(f"  layers={num_layers}  extraction_layer={layer_index}  d_model={d_model}")

    # Hook to capture hidden states
    captured: torch.Tensor | None = None

    def _hook(_module, _inputs, output):
        nonlocal captured
        h = output[0] if isinstance(output, tuple) else output
        captured = h.detach().clone()

    handle = layers[layer_index].register_forward_hook(_hook)

    # Tokenize and forward
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        base_model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)

    handle.remove()
    assert captured is not None, f"Hook on layer {layer_index} did not fire!"

    # Take hidden state at the last non-padding token
    seq_len = attn_mask.sum(dim=1).item()
    av = captured[0, seq_len - 1, :].float().cpu().numpy()  # [d_model]

    # Free base model memory
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  AV extracted: shape={av.shape}, norm={np.linalg.norm(av):.2f}")
    return av, layer_index


def nla_translate(
    av_vector: np.ndarray,
    actor,
    critic,
    tokenizer,
    tokens: dict,
    d_model: int,
    injection_scale: float | None = None,
    max_new_tokens: int = 150,
    device: str = "cuda",
) -> dict:
    """Translate an AV vector through the full NLA pipeline.

    Parameters
    ----------
    av_vector : [d_model] float32 numpy array

    Returns
    -------
    dict with keys: explanation, ar_vector, actor_text (full generation)
    """
    from nla.training.injection import inject_at_marked_positions
    from nla.training.schema import normalize_activation, extract_explanation

    inj_id = tokens["injection_token_id"]
    left_id = tokens["injection_left_neighbor_id"]
    right_id = tokens["injection_right_neighbor_id"]
    inj_char = tokens["injection_char"]

    vec = torch.from_numpy(av_vector.astype(np.float32)).to(device)

    # Apply injection normalization (same as training: 2.5 * sqrt(d_model))
    scale = injection_scale if injection_scale is not None else 2.5 * np.sqrt(d_model)
    vec = normalize_activation(vec.unsqueeze(0), scale).squeeze(0)

    # ── 1. Actor: generate explanation from AV ──────────────────────────
    user_content = (
        "You are a meticulous AI researcher conducting an important investigation "
        "into activation vectors from a language model. Your overall task is to describe "
        "the semantic content of that activation vector.\n\n"
        "We will pass the vector enclosed in <concept> tags into your context. You must "
        "then produce an explanation for the vector, enclosed within <explanation> tags. "
        "The explanation consists of 2-3 text snippets describing that vector.\n\n"
        "Here is the vector:\n\n"
        f"<concept>{inj_char}</concept>\n\n"
        "Please provide an explanation."
    )
    messages = [{"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                          max_length=2048)["input_ids"].to(device)

    embed_layer = actor.get_input_embeddings()
    with torch.no_grad():
        embeddings = embed_layer(prompt_ids)
        embeddings = inject_at_marked_positions(
            prompt_ids, embeddings,
            vec.unsqueeze(0).to(embeddings.device, embeddings.dtype),
            inj_id, left_id, right_id,
        )

        # Register hook for generation
        def _gen_hook(_vec, _inj_id, _left_id, _right_id):
            def _hook(_module, args, output):
                actual_ids = args[0]
                if output.shape[1] > 1:
                    return inject_at_marked_positions(
                        actual_ids, output, _vec.repeat(output.shape[0], 1),
                        _inj_id, _left_id, _right_id,
                    )
                return output
            return _hook

        hook = embed_layer.register_forward_hook(
            _gen_hook(vec.unsqueeze(0), inj_id, left_id, right_id)
        )
        try:
            gen_out = actor.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        finally:
            hook.remove()

    full_text = tokenizer.decode(gen_out[0], skip_special_tokens=True)
    explanation = extract_explanation(full_text)

    result = {
        "actor_output": full_text,
        "explanation": explanation,
        "ar_vector": None,
        "ar_vector_norm": None,
    }

    # ── 2. Critic: predict AR vector from explanation ───────────────────
    if explanation is not None:
        critic_prompt = (
            "You are given a description of an activation vector. Predict the vector.\n\n"
            f"Description: {explanation}\n\n"
            "The predicted vector is:"
        )

        critic_enc = tokenizer(critic_prompt, return_tensors="pt", truncation=True,
                              max_length=512)
        c_ids = critic_enc["input_ids"].to(device)
        c_mask = critic_enc["attention_mask"].to(device)

        with torch.no_grad():
            c_out = critic(input_ids=c_ids, attention_mask=c_mask)
            seq_len = c_mask.sum(dim=1) - 1
            ar_vector = c_out.values[0, seq_len[0], :]  # [d_model]

        result["ar_vector"] = ar_vector.cpu().float().numpy().tolist()
        result["ar_vector_norm"] = float(ar_vector.norm().item())

    return result


def main():
    p = argparse.ArgumentParser(description="NLA inference: AV → explanation → AR")
    # ── Model config ─────────────────────────────────────────────────────
    p.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME,
                   help=f"HF model ID for tokenizer (default: {DEFAULT_MODEL_NAME})")
    p.add_argument("--actor-ckpt", type=str, default=DEFAULT_ACTOR_CKPT,
                   help=f"Path to actor SFT checkpoint (default: {DEFAULT_ACTOR_CKPT})")
    p.add_argument("--critic-ckpt", type=str, default=DEFAULT_CRITIC_CKPT,
                   help=f"Path to critic SFT checkpoint (default: {DEFAULT_CRITIC_CKPT})")
    p.add_argument("--sidecar", type=str, default=DEFAULT_SIDECAR,
                   help=f"Path to .nla_meta.yaml with injection tokens (default: {DEFAULT_SIDECAR})")
    # ── Input/output ─────────────────────────────────────────────────────
    p.add_argument("--text", type=str, help="Input text: extract AV from base model, then full NLA")
    p.add_argument("--av-file", type=str, help="Path to .npy file with AV vector [d_model]")
    p.add_argument("--av-json", type=str, help="Path to JSON file with AV vector as list")
    p.add_argument("--output", type=str, help="Save results to JSON file")
    p.add_argument("--interactive", action="store_true",
                   help="Launch interactive mode (type vectors as comma-separated floats)")
    # ── Extraction params ────────────────────────────────────────────────
    p.add_argument("--extract-layer", type=int, default=None,
                   help="Layer for AV extraction (default: 2/3 of total layers)")
    # ── Runtime ──────────────────────────────────────────────────────────
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--injection-scale", type=float, default=None,
                   help="Override injection scale (default: 2.5*sqrt(d_model))")
    args = p.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    actor, critic, tokenizer, tokens, d_model = load_models(
        model_name=args.model_name,
        actor_ckpt=args.actor_ckpt,
        critic_ckpt=args.critic_ckpt,
        sidecar_path=args.sidecar,
        device=device,
    )

    def _process(av: np.ndarray) -> dict:
        if av.shape[-1] != d_model:
            raise ValueError(f"AV vector has dim {av.shape[-1]}, expected {d_model}")
        return nla_translate(
            av.reshape(-1)[:d_model],  # flatten and ensure correct size
            actor, critic, tokenizer, tokens, d_model,
            injection_scale=args.injection_scale,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )

    # ── Text mode: extract AV from base model first ─────────────────────
    if args.text:
        av, layer = extract_av(
            args.text,
            model_name=args.model_name,
            layer_index=args.extract_layer,
            device=device,
        )
        print(f"\nInput: \"{args.text}\"")
        print(f"Extracted AV at layer {layer}, norm={np.linalg.norm(av):.2f}")
        result = _process(av)
        print(f"\nRaw actor output:\n{result['actor_output']}")
        print(f"\nExplanation: {result['explanation']}")
        if result["ar_vector"] is not None:
            print(f"AR vector norm: {result['ar_vector_norm']:.4f}")
        if args.output:
            result["input_text"] = args.text
            result["extraction_layer"] = layer
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Saved to {args.output}")

    # ── Batch mode (file input) ──────────────────────────────────────────
    elif args.av_file or args.av_json:
        if args.av_file:
            av = np.load(args.av_file)
        else:
            with open(args.av_json) as f:
                av = np.array(json.load(f), dtype=np.float32)
        result = _process(av)
        print(f"\nExplanation: {result['explanation']}")
        if result["ar_vector"] is not None:
            print(f"AR vector norm: {result['ar_vector_norm']:.4f}")
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Saved to {args.output}")

    # ── Interactive mode ─────────────────────────────────────────────────
    elif args.interactive:
        print(f"\nNLA Interactive Inference (d_model={d_model})")
        print("Enter AV vector as comma-separated floats, or 'q' to quit.")
        print(f"Example: {','.join(['0.1'] * 5)}... (need {d_model} values)\n")
        while True:
            try:
                line = input("AV> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if line.lower() in ("q", "quit", "exit"):
                break
            if not line:
                continue
            try:
                values = [float(x.strip()) for x in line.split(",")]
                if len(values) != d_model:
                    print(f"  Expected {d_model} values, got {len(values)}")
                    continue
                av = np.array(values, dtype=np.float32)
                result = _process(av)
                print(f"\n  Actor output:\n  {result['actor_output'][:500]}")
                print(f"\n  Explanation: {result['explanation']}")
                if result["ar_vector"] is not None:
                    print(f"  AR vector norm: {result['ar_vector_norm']:.4f}")
                print()
            except ValueError as e:
                print(f"  Parse error: {e}")

    else:
        # Default: demo mode with a random AV vector
        print("\nNo input specified. Running demo with random AV vector...")
        rng = np.random.RandomState(42)
        av = rng.randn(d_model).astype(np.float32)
        result = _process(av)
        print(f"\nActor output:\n{result['actor_output'][:500]}")
        print(f"\nExtracted explanation: {result['explanation']}")
        if result["ar_vector"] is not None:
            print(f"AR vector norm: {result['ar_vector_norm']:.4f}")


if __name__ == "__main__":
    main()
