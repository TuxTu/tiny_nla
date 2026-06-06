"""Critic SL training — truncated model learns MSE regression (text → activation vector).

Simplest training stage: no generation, no injection hook. Tokenize prompt,
forward through truncated model, compute MSE at last-token position.

Usage:
  python -m nla.training.train_critic_sft \
    --data data/test/ar_sft_train.parquet \
    --model-name Qwen/Qwen3-0.6B \
    --output-dir data/test/critic_checkpoint \
    --micro-batch-size 2 --num-steps 10
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from nla.training.env_config import detect
from nla.training.loss import nla_critic_loss
from nla.training.models import NLACriticModel
from nla.training.schema import (
    ACTIVATION_COLUMN,
    resolve_target_scale,
)
from nla.training.sidecar import read_sidecar


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------


class CriticDataset(Dataset):
    """AR-SFT parquet → (prompt_str, activation_vector) pairs."""

    def __init__(self, parquet_path: str):
        self.table = pq.read_table(parquet_path)
        self.prompts = self.table.column("prompt").to_pylist()
        # Read FixedSizeList activation vectors → numpy
        col = self.table.column(ACTIVATION_COLUMN)
        self.vectors = np.array([v.as_py() for v in col], dtype=np.float32)
        self.d_model = self.vectors.shape[1]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx], torch.from_numpy(self.vectors[idx])


# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------


def train(args) -> None:
    env = detect()
    print(f"device: {env.device}  dtype: {env.dtype}")

    # ---- data ----------------------------------------------------------------
    ds = CriticDataset(args.data)
    print(f"dataset: {len(ds)} rows  d_model={ds.d_model}")

    dl = DataLoader(ds, batch_size=args.micro_batch_size, shuffle=True)

    # ---- sidecar -------------------------------------------------------------
    sidecar = read_sidecar(args.data)
    mse_scale_raw = sidecar.get("extraction", {}).get("mse_scale", "sqrt_d_model")
    mse_scale = resolve_target_scale(mse_scale_raw, ds.d_model)
    print(f"mse_scale: {mse_scale}")

    # ---- model ---------------------------------------------------------------
    layer_index = args.layer_index
    if layer_index is None:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
        layer_index = (2 * cfg.num_hidden_layers) // 3
        print(f"auto layer_index: {layer_index}  (2/3 × {cfg.num_hidden_layers})")

    print(f"loading {args.model_name} ...")
    model = NLACriticModel.from_pretrained(
        args.model_name, nla_num_layers=layer_index,
        torch_dtype=env.dtype, device_map={"": env.device} if not env.is_mps else None,
    )
    if env.is_mps:
        model = model.to(env.device)
    model.train()
    model.gradient_checkpointing_enable()
    print(f"critic: {layer_index + 1} layers  d_model={model.config.hidden_size}")

    # ---- optimizer -----------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ---- tokenizer -----------------------------------------------------------
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    # ---- training ------------------------------------------------------------
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    global_step = 0
    losses = []

    while global_step < args.num_steps:
        pbar = tqdm(dl, desc=f"critic  step={global_step}/{args.num_steps}")
        for prompts, gold_vectors in pbar:
            if global_step >= args.num_steps:
                break

            # tokenize
            enc = tokenizer(
                list(prompts), padding=True, truncation=True,
                max_length=args.max_length, return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(env.device)
            attention_mask = enc["attention_mask"].to(env.device)
            gold = gold_vectors.to(env.device)

            with torch.autocast(device_type=env.device.type, dtype=env.dtype,
                                enabled=env.amp_enabled):
                output = model(input_ids=input_ids, attention_mask=attention_mask)
                # Extract last-token position per sample
                seq_lens = attention_mask.sum(dim=1) - 1  # [B]
                pred = output.values[torch.arange(len(seq_lens)), seq_lens]  # [B, d]
                loss = nla_critic_loss(pred, gold, mse_scale)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if env.is_mps and global_step % 5 == 0:
                torch.mps.empty_cache()

            losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            global_step += 1

    # ---- save ----------------------------------------------------------------
    avg_loss = sum(losses) / len(losses)
    print(f"\nfinal loss: {avg_loss:.4f}  ({len(losses)} steps)")

    save_dir = Path(args.output_dir)
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    print(f"saved → {save_dir}")


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="AR-SFT training parquet")
    p.add_argument("--model-name", required=True, help="HF base model")
    p.add_argument("--layer-index", type=int, default=None,
                   help="extraction layer (default: 2/3 * num_layers)")
    p.add_argument("--output-dir", required=True, help="checkpoint directory")
    p.add_argument("--micro-batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--max-length", type=int, default=512,
                   help="max token length for critic prompt (shorter than extraction)")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
