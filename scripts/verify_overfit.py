"""Verify the critic can overfit to a tiny batch — diagnostic for zero FVE.

Loads the 64k-trained 8B critic, takes N samples, trains on just those for K steps.
If MSE drops well below baseline (0.673), training mechanism works.
If not, there's a fundamental bug.
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path

from nla.training.models import NLACriticModel
from nla.training.loss import nla_critic_loss, normalize_activation
from nla.training.schema import SCALE_SQRT_D
import pyarrow.parquet as pq
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--train-full", action="store_true",
                   help="Train full model (not just value_head)")
    args = p.parse_args()

    device = args.device
    d_model = 4096
    mse_scale = 64.0  # sqrt(d_model)

    print(f"=== Overfit test: {args.n_samples} samples, {args.n_steps} steps, lr={args.lr} ===")
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {args.ckpt}...")
    model = NLACriticModel.from_pretrained(args.ckpt, nla_num_layers=24, torch_dtype=torch.bfloat16)
    model = model.to(device)
    model.train()

    # Freeze backbone, only train value_head (for faster test)
    if not args.train_full:
        for name, p in model.named_parameters():
            if "value_head" not in name:
                p.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} {'(full model)' if args.train_full else '(value_head only)'}")

    # Load data
    print(f"Loading data from {args.data}...")
    pf = pq.ParquetFile(args.data)
    batch = next(pf.iter_batches(batch_size=args.n_samples))
    cols = pf.schema_arrow.names
    av_col = batch.column("activation_vector")
    av_flat = av_col.flatten().to_numpy(zero_copy_only=False).astype(np.float32)
    gold_vectors = av_flat.reshape(len(av_col), -1)
    gold = torch.tensor(gold_vectors, dtype=torch.float32, device=device)

    prompts = batch.column("prompt").to_pylist()
    print(f"Loaded {len(prompts)} samples, gold shape: {gold.shape}")

    # Compute baseline: MSE if we always predict the mean gold vector
    mean_gold = gold.mean(dim=0, keepdim=True)
    baseline_mse = F.mse_loss(
        normalize_activation(mean_gold.expand_as(gold), mse_scale),
        normalize_activation(gold, mse_scale),
    ).item()
    # Mean predictor baseline (predict mean for each sample)
    mean_pred_mse = 0.0
    for i in range(len(gold)):
        others = torch.cat([gold[:i], gold[i + 1 :]], dim=0)
        mean_other = others.mean(dim=0)
        mean_pred_mse += F.mse_loss(
            normalize_activation(mean_other.unsqueeze(0), mse_scale),
            normalize_activation(gold[i : i + 1], mse_scale),
        ).item()
    mean_pred_mse /= len(gold)
    print(f"Baseline MSE (predict global mean): {baseline_mse:.6f}")
    print(f"Baseline MSE (leave-one-out mean): {mean_pred_mse:.6f}")

    # Tokenize prompts (critic: string prompts, last token extraction)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    # Get last-token positions
    seq_lens = attention_mask.sum(dim=1) - 1
    print(f"Tokenized: {input_ids.shape}, last positions: {seq_lens.tolist()}")

    # Optimizer
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    print(f"\n{'Step':>6s} {'MSE':>10s} {'FVE%':>8s} {'pred_norm':>10s} {'gold_norm':>10s}")
    print("-" * 50)

    best_mse = float("inf")

    for step in range(args.n_steps + 1):
        opt.zero_grad()

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        pred = out.values[torch.arange(len(seq_lens)), seq_lens]

        with torch.no_grad():
            # Per-sample MSE (mean over d_model, then mean over batch)
            per_sample_mse = F.mse_loss(
                normalize_activation(pred.float(), mse_scale),
                normalize_activation(gold, mse_scale),
                reduction="none",
            ).mean(dim=-1)
            mse = per_sample_mse.mean().item()
            fve = max(0.0, (1.0 - mse / baseline_mse) * 100)
            pred_norm = pred.float().norm(dim=-1).mean().item()
            gold_norm = gold.norm(dim=-1).mean().item()

        if step < args.n_steps:
            # Use same loss as training
            loss = F.mse_loss(
                normalize_activation(pred.float(), mse_scale),
                normalize_activation(gold, mse_scale),
            )
            loss.backward()
            opt.step()

        if step % 5 == 0:
            print(f"{step:>6d} {mse:>10.6f} {fve:>8.2f} {pred_norm:>10.4f} {gold_norm:>10.4f}")
            if mse < best_mse:
                best_mse = mse

    print(f"\n=== Results ===")
    print(f"Best MSE: {best_mse:.6f}")
    print(f"Baseline MSE: {baseline_mse:.6f}")
    best_fve = max(0.0, (1.0 - best_mse / baseline_mse) * 100)
    print(f"Best FVE: {best_fve:.2f}%")
    if best_fve > 5:
        print("✓ Model CAN reduce loss below baseline — training mechanism works")
    else:
        print("✗ Model CANNOT reduce loss below baseline — FUNDAMENTAL BUG suspected")


if __name__ == "__main__":
    main()
