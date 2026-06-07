"""Critic SL training — truncated model learns MSE regression (text → activation vector).

Simplest training stage: no generation, no injection hook. Tokenize prompt,
forward through truncated model, compute MSE at last-token position.

Usage (single-GPU):
  python -m nla.training.train_critic_sft \
    --data data/test/ar_sft_train.parquet \
    --model-name Qwen/Qwen3-0.6B \
    --output-dir data/test/critic_checkpoint \
    --micro-batch-size 2 --num-steps 10

Usage (multi-GPU DDP via torchrun):
  torchrun --nproc_per_node=4 -m nla.training.train_critic_sft \
    --data data/test/ar_sft_train.parquet \
    --model-name Qwen/Qwen3-0.6B \
    --output-dir data/test/critic_checkpoint \
    --micro-batch-size 8 --global-batch-size 256 --num-steps 250 \
    --ddp --lr 2e-5 --min-lr 2e-6 --lr-warmup-iters 50 --lr-decay-style cosine
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from nla.training.env_config import detect, EnvConfig
from nla.training.loss import nla_critic_loss
from nla.training.models import NLACriticModel
from nla.training.resolve import resolve_parquet
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
        self.table = pq.read_table(resolve_parquet(parquet_path))
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
# LR schedule
# ---------------------------------------------------------------------------


def _get_lr_scheduler(optimizer, args, env: EnvConfig) -> torch.optim.lr_scheduler.LambdaLR | None:
    """Build cosine LR schedule with linear warmup.

    Returns None if decay_style is 'constant'.
    """
    decay = getattr(args, "lr_decay_style", "cosine")
    if decay == "constant":
        return None

    warmup = getattr(args, "lr_warmup_iters", 0)
    min_lr = getattr(args, "min_lr", args.lr)
    total_steps = args.num_steps

    def _lr_lambda(step):
        # Linear warmup
        if step < warmup and warmup > 0:
            return float(step) / float(max(1, warmup))
        # Cosine decay from lr → min_lr
        if step >= total_steps:
            return min_lr / args.lr
        progress = float(step - warmup) / float(max(1, total_steps - warmup))
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / args.lr, cosine_factor)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------


def train(args) -> None:
    env = detect()

    # ---- DDP setup -----------------------------------------------------------
    if args.ddp and env.world_size > 1:
        dist.init_process_group(backend="nccl")
        print(f"[rank {env.global_rank}/{env.world_size}] "
              f"device={env.device}  gpu={env.gpu_name}")
        # Ensure this process only sees its assigned GPU
        torch.cuda.set_device(env.local_rank)
    elif args.ddp:
        print("[ddp] WARNING: --ddp passed but WORLD_SIZE=1 — running single-GPU")
        args.ddp = False

    if env.is_main_process:
        print(f"device: {env.device}  dtype: {env.dtype}  "
              f"ddp: {args.ddp}  world_size: {env.world_size}")

    # ---- resolve data path (may be HF Hub repo) --------------------------------
    data_path = resolve_parquet(args.data)

    # ---- data ----------------------------------------------------------------
    ds = CriticDataset(data_path)
    if env.is_main_process:
        print(f"dataset: {len(ds)} rows  d_model={ds.d_model}")

    sampler = DistributedSampler(ds, num_replicas=env.world_size,
                                  rank=env.global_rank,
                                  shuffle=True) if args.ddp else None
    dl = DataLoader(ds,
                    batch_size=args.micro_batch_size,
                    shuffle=(sampler is None),
                    sampler=sampler,
                    drop_last=args.ddp,  # avoid uneven batch in DDP
                    )

    # ---- sidecar -------------------------------------------------------------
    sidecar = read_sidecar(data_path)
    mse_scale_raw = sidecar.get("extraction", {}).get("mse_scale", "sqrt_d_model")
    mse_scale = resolve_target_scale(mse_scale_raw, ds.d_model)
    if env.is_main_process:
        print(f"mse_scale: {mse_scale}")

    # ---- model ---------------------------------------------------------------
    layer_index = args.layer_index
    if layer_index is None:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
        layer_index = (2 * cfg.num_hidden_layers) // 3
        if env.is_main_process:
            print(f"auto layer_index: {layer_index}  (2/3 × {cfg.num_hidden_layers})")

    if args.resume:
        resume_path = str(Path(args.output_dir).resolve())
        assert Path(resume_path, "value_head.safetensors").exists(), (
            f"no critic checkpoint found at {resume_path} — cannot resume"
        )
        if env.is_main_process:
            print(f"resuming critic from {resume_path} ...")
        model = NLACriticModel.from_pretrained(
            resume_path,
            torch_dtype=env.dtype, device_map={"": env.device} if not args.ddp else None,
        )
    else:
        if env.is_main_process:
            print(f"loading {args.model_name} ...")
        model = NLACriticModel.from_pretrained(
            args.model_name, nla_num_layers=layer_index,
            torch_dtype=env.dtype, device_map={"": env.device} if not args.ddp else None,
        )
    if env.is_mps:
        model = model.to(env.device)
    if args.ddp:
        model = model.to(env.device)
    model.train()
    model.gradient_checkpointing_enable()

    if args.ddp:
        model = DDP(model, device_ids=[env.local_rank] if torch.cuda.is_available() else None,
                    find_unused_parameters=False)

    if env.is_main_process:
        print(f"critic: {model.module.config.num_hidden_layers if args.ddp else model.config.num_hidden_layers} "
              f"layers  d_model={model.module.config.hidden_size if args.ddp else model.config.hidden_size}")

    # ---- optimizer -----------------------------------------------------------
    optimizer = torch.optim.AdamW(
        (model.parameters() if not args.ddp else model.parameters()),
        lr=args.lr,
    )
    lr_scheduler = _get_lr_scheduler(optimizer, args, env)

    # ---- tokenizer -----------------------------------------------------------
    from transformers import AutoTokenizer
    if args.resume:
        tokenizer = AutoTokenizer.from_pretrained(str(Path(args.output_dir).resolve()))
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    # ---- gradient accumulation ------------------------------------------------
    global_batch = getattr(args, "global_batch_size", args.micro_batch_size * env.world_size)
    grad_accum = max(1, global_batch // (args.micro_batch_size * env.world_size))
    if env.is_main_process:
        print(f"global_batch={global_batch}  micro_batch={args.micro_batch_size}  "
              f"world_size={env.world_size}  grad_accum={grad_accum}")

    # ---- training ------------------------------------------------------------
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    global_step = 0
    losses = []

    def _save():
        if not env.is_main_process:
            return
        save_dir = Path(args.output_dir)
        model_to_save = model.module if args.ddp else model
        model_to_save.save_pretrained(str(save_dir))
        tokenizer.save_pretrained(str(save_dir))
        print(f"  checkpoint saved → {save_dir}  (step {global_step})")

    while global_step < args.num_steps:
        if sampler is not None:
            sampler.set_epoch(global_step)
        pbar = tqdm(dl, desc=f"critic  step={global_step}/{args.num_steps}",
                    disable=not env.is_main_process)
        any_data = False
        accum_loss = 0.0
        optimizer.zero_grad()

        for prompts, gold_vectors in pbar:
            any_data = True
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

            model_fwd = model.module if args.ddp else model
            with torch.autocast(device_type=env.device.type, dtype=env.dtype,
                                enabled=env.amp_enabled):
                output = model_fwd(input_ids=input_ids, attention_mask=attention_mask)
                # Extract last-token position per sample
                seq_lens = attention_mask.sum(dim=1) - 1  # [B]
                pred = output.values[torch.arange(len(seq_lens)), seq_lens]  # [B, d]
                loss = nla_critic_loss(pred, gold, mse_scale)
                loss = loss / grad_accum

            loss.backward()
            accum_loss += loss.item() * grad_accum

            # Step only after accumulation
            if ((global_step + 1) % grad_accum == 0) or (global_step + 1 >= args.num_steps):
                optimizer.step()
                optimizer.zero_grad()
                if lr_scheduler is not None:
                    lr_scheduler.step()

            if env.is_mps and global_step % 5 == 0:
                torch.mps.empty_cache()

            if global_step % args.save_every == 0 and global_step > 0:
                _save()

            current_lr = optimizer.param_groups[0]["lr"]
            losses.append(accum_loss)
            if env.is_main_process:
                pbar.set_postfix(loss=f"{accum_loss:.4f}", lr=f"{current_lr:.2e}")
            accum_loss = 0.0
            global_step += 1

        if not any_data:
            if env.is_main_process:
                print("  DataLoader exhausted — stopping.")
            break

    # ---- final save ----------------------------------------------------------
    if args.ddp:
        dist.barrier()
    avg_loss = sum(losses) / len(losses) if losses else 0
    if env.is_main_process:
        print(f"\nfinal loss: {avg_loss:.4f}  ({len(losses)} steps)")
        _save()

    if args.ddp:
        dist.destroy_process_group()


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
    p.add_argument("--global-batch-size", type=int, default=None,
                   help="global batch size for gradient accumulation (DDP: micro*gpus*accum)")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--min-lr", type=float, default=None,
                   help="minimum LR for cosine decay (default: same as --lr)")
    p.add_argument("--lr-warmup-iters", type=int, default=0,
                   help="linear warmup steps")
    p.add_argument("--lr-decay-style", type=str, default="constant",
                   choices=["constant", "cosine"],
                   help="LR decay style (default: constant)")
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--max-length", type=int, default=512,
                   help="max token length for critic prompt (shorter than extraction)")
    p.add_argument("--save-every", type=int, default=500,
                   help="save checkpoint every N steps (default: 500)")
    p.add_argument("--resume", action="store_true",
                   help="resume from checkpoint in --output-dir")
    p.add_argument("--ddp", action="store_true",
                   help="enable DistributedDataParallel (use with torchrun)")
    args = p.parse_args()

    # Defaults
    if args.min_lr is None:
        args.min_lr = args.lr
    if args.global_batch_size is None:
        args.global_batch_size = args.micro_batch_size

    train(args)


if __name__ == "__main__":
    main()
