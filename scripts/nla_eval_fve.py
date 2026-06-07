#!/usr/bin/env python3
"""Evaluate NLA pipeline FVE (Fraction of Variance Explained).

Mirrors the original NLA's FVE metric (loss.py:110, schema.py:156-174):
  fve_nrm = 1.0 - MSE(pred, gold) / baseline_rawvar

where baseline_rawvar = MSE(gold, mean(gold)) — the per-element variance
of the normalized gold vectors.  FVE > 0 means the critic is doing better
than the constant "predict the mean" baseline.

Two variants:
  fve_nrm          — classical FVE: MSE vs. unnormalized mean
  fve_nrm_meannorm — tighter: MSE vs. normalized mean (critic output
                      also gets normalized, so this is the best a constant
                      predictor can do)

Usage:
  # Full evaluation on parquet data:
  python scripts/nla_eval_fve.py \
      --data data/rl_train.parquet \
      --actor-ckpt checkpoints/rl/actor \
      --critic-ckpt checkpoints/rl/critic \
      --n-samples 500

  # Quick single-text evaluation:
  python scripts/nla_eval_fve.py \
      --text "The capital of France is" \
      --actor-ckpt checkpoints/actor_sft \
      --critic-ckpt checkpoints/critic_sft
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJ = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B"
DEFAULT_ACTOR_CKPT = str(PROJ / "checkpoints/actor_sft")
DEFAULT_CRITIC_CKPT = str(PROJ / "checkpoints/critic_sft")
DEFAULT_SIDECAR = str(PROJ / "data/av_sft_train.parquet.nla_meta.yaml")


# ---------------------------------------------------------------------------
# FVE baselines (from original NLA schema.py:152-175)
# ---------------------------------------------------------------------------


def compute_fve_baselines(
    vectors: torch.Tensor, mse_scale: float
) -> tuple[float, float]:
    """Two predict-the-mean baseline MSEs for FVE normalisation.

    Parameters
    ----------
    vectors : [N, d] float32 tensor of gold activation vectors.
    mse_scale : normalisation scale (sqrt(d_model) in the original).

    Returns
    -------
    (meannorm_baseline, raw_variance_baseline)
    """
    v_norm = _normalize(vectors.float(), mse_scale)
    mu = v_norm.mean(dim=0, keepdim=True)
    mu_normed = _normalize(mu, mse_scale)
    mse_meannorm = ((v_norm - mu_normed) ** 2).mean().item()
    mse_rawvar = ((v_norm - mu) ** 2).mean().item()
    return mse_meannorm, mse_rawvar


def _normalize(v: torch.Tensor, target_scale: float) -> torch.Tensor:
    """Scale rows of *v* to L2-norm == target_scale."""
    norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v / (norm / target_scale)


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def load_models(
    model_name: str,
    actor_ckpt: str,
    critic_ckpt: str,
    sidecar_path: str,
    device: str = "cuda",
):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from nla.training.models import NLACriticModel
    from nla.training.sidecar import read_sidecar

    print(f"Loading actor  {actor_ckpt}")
    actor = AutoModelForCausalLM.from_pretrained(
        actor_ckpt, torch_dtype=torch.bfloat16, device_map={"": device},
    )
    actor.eval()

    print(f"Loading critic {critic_ckpt}")
    critic = NLACriticModel.from_pretrained(critic_ckpt, torch_dtype=torch.bfloat16)
    critic = critic.to(device).eval()

    try:
        tokenizer = AutoTokenizer.from_pretrained(actor_ckpt)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if sidecar_path.endswith(".nla_meta.yaml"):
        sidecar_path = sidecar_path[: -len(".nla_meta.yaml")]
    sidecar = read_sidecar(sidecar_path)
    tokens = sidecar["tokens"]
    d_model = sidecar["extraction"]["d_model"]
    mse_scale = sidecar.get("extraction", {}).get(
        "mse_scale", math.sqrt(d_model)
    )
    if isinstance(mse_scale, str):
        mse_scale = float(mse_scale) if mse_scale != "sqrt_d_model" else math.sqrt(d_model)

    return actor, critic, tokenizer, tokens, d_model, mse_scale


def extract_av(
    text: str,
    model_name: str = DEFAULT_MODEL_NAME,
    layer_index: int | None = None,
    device: str = "cuda",
) -> np.ndarray:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map={"": device},
    )
    model.eval()

    if hasattr(model, "language_model"):
        inner = model.language_model
    else:
        inner = model
    if hasattr(inner, "model"):
        layers = inner.model.layers
    elif hasattr(inner, "transformer"):
        layers = inner.transformer.h
    else:
        raise RuntimeError(f"Cannot find layers in {type(model).__name__}")

    if layer_index is None:
        layer_index = (2 * len(layers)) // 3

    captured = None

    def _hook(_m, _in, out):
        nonlocal captured
        h = out[0] if isinstance(out, tuple) else out
        captured = h.detach().clone()

    handle = layers[layer_index].register_forward_hook(_hook)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=enc["attention_mask"].to(device), use_cache=False)
    handle.remove()

    seq_len = enc["attention_mask"].sum(dim=1).item()
    av = captured[0, seq_len - 1, :].float().cpu().numpy()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return av


def nla_predict_explanation_and_ar(
    av: torch.Tensor,
    actor,
    critic,
    tokenizer,
    tokens: dict,
    d_model: int,
    mse_scale: float,
    max_new_tokens: int = 256,
    device: str = "cuda",
) -> str | None:
    """Run full NLA pipeline for one AV.  Returns (explanation_text, ar_vector)."""
    from nla.training.injection import inject_at_marked_positions
    from nla.training.schema import normalize_activation, extract_explanation

    inj_id = tokens["injection_token_id"]
    left_id = tokens["injection_left_neighbor_id"]
    right_id = tokens["injection_right_neighbor_id"]
    inj_char = tokens["injection_char"]

    vec = torch.from_numpy(av.astype(np.float32)).to(device)
    scale = 2.5 * math.sqrt(d_model)
    vec = normalize_activation(vec.unsqueeze(0), scale).squeeze(0)

    # ── 1. Actor: generate explanation ──────────────────────────────
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
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)["input_ids"].to(device)

    embed_layer = actor.get_input_embeddings()
    with torch.no_grad():
        embeddings = embed_layer(prompt_ids)
        embeddings = inject_at_marked_positions(
            prompt_ids, embeddings, vec.unsqueeze(0).to(embeddings.device, embeddings.dtype),
            inj_id, left_id, right_id,
        )

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
                prompt_ids, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        finally:
            hook.remove()

    full_text = tokenizer.decode(gen_out[0], skip_special_tokens=True)
    explanation = extract_explanation(full_text)
    if explanation is None:
        return None, None

    # ── 2. Critic: predict AR vector ─────────────────────────────────
    critic_prompt = (
        "You are given a description of an activation vector. Predict the vector.\n\n"
        f"Description: {explanation}\n\n"
        "The predicted vector is:"
    )
    critic_enc = tokenizer(critic_prompt, return_tensors="pt", truncation=True, max_length=512)
    c_ids = critic_enc["input_ids"].to(device)
    c_mask = critic_enc["attention_mask"].to(device)
    with torch.no_grad():
        c_out = critic(input_ids=c_ids, attention_mask=c_mask)
        seq_len = c_mask.sum(dim=1) - 1
        ar_vec = c_out.values[0, seq_len[0], :]  # [d_model]

    return explanation, ar_vec.float()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="NLA FVE evaluation")
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    p.add_argument("--actor-ckpt", default=DEFAULT_ACTOR_CKPT)
    p.add_argument("--critic-ckpt", default=DEFAULT_CRITIC_CKPT)
    p.add_argument("--sidecar", default=DEFAULT_SIDECAR)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-new-tokens", type=int, default=256)

    # Data source
    p.add_argument("--data", help="Parquet file with activation_vector column")
    p.add_argument("--n-samples", type=int, default=500, help="Samples to evaluate (from parquet)")
    p.add_argument("--text", help="Single text for quick eval")

    p.add_argument("--jsonl", help="Save per-sample results to JSONL")

    args = p.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    actor, critic, tokenizer, tokens, d_model, mse_scale = load_models(
        args.model_name, args.actor_ckpt, args.critic_ckpt, args.sidecar, device,
    )

    # ── Load / collect gold vectors ──────────────────────────────────
    gold_vecs = []
    if args.text:
        av = extract_av(args.text, model_name=args.model_name, device=device)
        gold_vecs.append(av)
    elif args.data:
        import pandas as pd
        df = pd.read_parquet(args.data)
        if args.n_samples and len(df) > args.n_samples:
            df = df.sample(n=args.n_samples, random_state=42)
        for vec in df["activation_vector"]:
            gold_vecs.append(np.array(vec, dtype=np.float32))
    else:
        print("ERROR: need --text or --data")
        sys.exit(1)

    gold_t = torch.tensor(np.stack(gold_vecs), dtype=torch.float32)
    print(f"Evaluating {len(gold_t)} samples  d_model={d_model}")

    # ── Compute baselines ────────────────────────────────────────────
    mse_meannorm_baseline, mse_rawvar_baseline = compute_fve_baselines(
        gold_t, float(mse_scale)
    )
    print(f"Baselines:  meannorm_MSE={mse_meannorm_baseline:.4f}  "
          f"rawvar_MSE={mse_rawvar_baseline:.4f}")

    # ── Evaluate ─────────────────────────────────────────────────────
    preds = []
    n_failed = 0
    results: list[dict] = []

    for i, av in enumerate(gold_vecs):
        explanation, ar = nla_predict_explanation_and_ar(
            av, actor, critic, tokenizer, tokens, d_model,
            float(mse_scale), max_new_tokens=args.max_new_tokens, device=device,
        )
        if explanation is None:
            n_failed += 1
            continue
        preds.append(ar.cpu())
        results.append({"idx": i, "explanation": explanation, "ar_vector": ar.tolist()})

        if (i + 1) % 50 == 0:
            mse_now = F.mse_loss(
                _normalize(torch.stack(preds), float(mse_scale)),
                _normalize(gold_t[list(range(len(preds)))], float(mse_scale)),
            ).item()
            fve = 1.0 - mse_now / mse_rawvar_baseline
            print(f"  {len(preds)}/{i+1}  MSE={mse_now:.4f}  "
                  f"FVE_nrm={fve:.3f}  failed={n_failed}")

    if not preds:
        print(f"ERROR: all {len(gold_vecs)} samples failed extraction")
        sys.exit(1)

    # ── Final metrics ────────────────────────────────────────────────
    pred_t = torch.stack(preds)
    gold_sub = gold_t[torch.tensor([r["idx"] for r in results])]

    pred_n = _normalize(pred_t, float(mse_scale))
    gold_n = _normalize(gold_sub, float(mse_scale))

    mse_final = ((pred_n - gold_n) ** 2).mean().item()
    fve_nrm = 1.0 - mse_final / mse_rawvar_baseline
    fve_meannorm = 1.0 - mse_final / mse_meannorm_baseline

    print(f"\n{'='*60}")
    print(f"Samples evaluated:  {len(preds)}  (failed extraction: {n_failed})")
    print(f"MSE:                {mse_final:.4f}")
    print(f"baseline (rawvar):  {mse_rawvar_baseline:.4f}")
    print(f"baseline (meannorm):{mse_meannorm_baseline:.4f}")
    print(f"FVE_nrm:            {fve_nrm:.4f}  (>0 ⇒ better than predict-mean)")
    print(f"FVE_nrm_meannorm:   {fve_meannorm:.4f}  (>0 ⇒ better than constant)")
    print(f"reward equivalent:  {-mse_final:.4f}")
    print(f"{'='*60}")

    if args.jsonl:
        import json
        with open(args.jsonl, "w") as f:
            for r in results:
                json.dump(r, f)
                f.write("\n")
        print(f"Per-sample results → {args.jsonl}")


if __name__ == "__main__":
    main()
