"""Actor SFT training — full model learns to generate explanations from injected activations.

Teacher-forcing CE loss on response tokens only. The injection hook replaces the
embedding at the marker token position with the activation vector during forward.

Usage (single-GPU):
  python -m nla.training.train_actor_sft \
    --data data/test/av_sft_train.parquet \
    --model-name Qwen/Qwen3-0.6B \
    --output-dir data/test/actor_checkpoint \
    --micro-batch-size 2 --num-steps 10

Usage (multi-GPU DDP via torchrun):
  torchrun --nproc_per_node=4 -m nla.training.train_actor_sft \
    --data data/test/av_sft_train.parquet \
    --model-name Qwen/Qwen3-0.6B \
    --output-dir data/test/actor_checkpoint \
    --micro-batch-size 4 --global-batch-size 256 --num-steps 250 \
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
from nla.training.injection import inject_at_marked_positions
from nla.training.loss import sft_loss
from nla.training.schema import (
    ACTIVATION_COLUMN,
    INJECT_PLACEHOLDER,
    extract_explanation,
    normalize_activation,
    resolve_target_scale,
)
from nla.training.sidecar import read_sidecar


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------


class ActorDataset(Dataset):
    """AV-SFT parquet → (messages, response, activation_vector) tuples."""

    def __init__(self, parquet_path: str, injection_char: str):
        from nla.training.resolve import resolve_parquet
        table = pq.read_table(resolve_parquet(parquet_path))
        raw_prompts = table.column("prompt").to_pylist()
        self.responses = table.column("response").to_pylist()

        col = table.column(ACTIVATION_COLUMN)
        self.vectors = np.array([v.as_py() for v in col], dtype=np.float32)

        # Swap <INJECT> placeholder → real injection char
        self.prompts = []
        for msg_list in raw_prompts:
            fixed = []
            for msg in msg_list:
                fixed.append({
                    "role": msg["role"],
                    "content": msg["content"].replace(INJECT_PLACEHOLDER, injection_char),
                })
            self.prompts.append(fixed)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx], self.responses[idx], torch.from_numpy(self.vectors[idx])


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def _get_lr_scheduler(optimizer, args, env: EnvConfig) -> torch.optim.lr_scheduler.LambdaLR | None:
    """Build cosine LR schedule with linear warmup."""
    decay = getattr(args, "lr_decay_style", "cosine")
    if decay == "constant":
        return None

    warmup = getattr(args, "lr_warmup_iters", 0)
    min_lr = getattr(args, "min_lr", args.lr)
    total_steps = args.num_steps

    def _lr_lambda(step):
        if step < warmup and warmup > 0:
            return float(step) / float(max(1, warmup))
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
        torch.cuda.set_device(env.local_rank)
    elif args.ddp:
        print("[ddp] WARNING: --ddp passed but WORLD_SIZE=1 — running single-GPU")
        args.ddp = False

    if env.is_main_process:
        print(f"device: {env.device}  dtype: {env.dtype}  "
              f"ddp: {args.ddp}  world_size: {env.world_size}")

    # ---- resolve data path ---------------------------------------------------
    from nla.training.resolve import resolve_parquet
    data_path = resolve_parquet(args.data)

    # ---- sidecar + tokenizer ------------------------------------------------
    sidecar = read_sidecar(data_path)
    tokens = sidecar.get("tokens", {})
    injection_char = tokens["injection_char"]
    inj_id = tokens["injection_token_id"]
    left_id = tokens["injection_left_neighbor_id"]
    right_id = tokens["injection_right_neighbor_id"]
    if env.is_main_process:
        print(f"injection: char={injection_char!r}  id={inj_id}  "
              f"neighbors=({left_id}, {right_id})")

    d_model = sidecar["extraction"]["d_model"]
    injection_scale = resolve_target_scale(
        sidecar.get("extraction", {}).get("injection_scale"),
        d_model,
    )
    if injection_scale is None:
        injection_scale = 2.5 * math.sqrt(d_model)
    if env.is_main_process:
        print(f"injection_scale: {injection_scale}")

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

    # ---- data ----------------------------------------------------------------
    ds = ActorDataset(data_path, injection_char)
    if env.is_main_process:
        print(f"dataset: {len(ds)} rows  d_model={ds.vectors.shape[1]}")

    def _collate(batch):
        """Custom collate — preserves list-of-dicts for prompts (no tensor stacking)."""
        prompts, responses, vectors = zip(*batch)
        return list(prompts), list(responses), torch.stack(vectors)

    sampler = DistributedSampler(ds, num_replicas=env.world_size,
                                  rank=env.global_rank,
                                  shuffle=True) if args.ddp else None
    dl = DataLoader(ds, batch_size=args.micro_batch_size,
                    shuffle=(sampler is None),
                    sampler=sampler,
                    drop_last=args.ddp,
                    collate_fn=_collate)

    # ---- model ---------------------------------------------------------------
    from transformers import AutoModelForCausalLM
    if args.resume:
        resume_path = str(Path(args.output_dir).resolve())
        assert Path(resume_path, "config.json").exists(), (
            f"no actor checkpoint found at {resume_path} — cannot resume"
        )
        if env.is_main_process:
            print(f"resuming actor from {resume_path} ...")
        model = AutoModelForCausalLM.from_pretrained(
            resume_path, torch_dtype=env.dtype,
            device_map={"": env.device} if not args.ddp else None,
        )
    else:
        if env.is_main_process:
            print(f"loading {args.model_name} ...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=env.dtype,
            device_map={"": env.device} if not args.ddp else None,
        )
    if env.is_mps:
        model = model.to(env.device)
    if args.ddp:
        model = model.to(env.device)
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if args.ddp:
        # find_unused_parameters=True because injection hook may skip some params
        model = DDP(model, device_ids=[env.local_rank] if torch.cuda.is_available() else None,
                    find_unused_parameters=True)

    if env.is_main_process:
        m = model.module if args.ddp else model
        print(f"actor: {m.config.num_hidden_layers} layers  d_model={m.config.hidden_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = _get_lr_scheduler(optimizer, args, env)

    # ---- gradient accumulation ------------------------------------------------
    global_batch = getattr(args, "global_batch_size", args.micro_batch_size * env.world_size)
    grad_accum = max(1, global_batch // (args.micro_batch_size * env.world_size))
    if env.is_main_process:
        print(f"global_batch={global_batch}  micro_batch={args.micro_batch_size}  "
              f"world_size={env.world_size}  grad_accum={grad_accum}")

    # ---- training ------------------------------------------------------------
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    opt_step = 0          # counts optimizer steps (= global_step in original)
    micro_step = 0         # counts micro-batches for grad_accum
    losses = []

    def _save():
        if not env.is_main_process:
            return
        save_dir = Path(args.output_dir)
        model_to_save = model.module if args.ddp else model
        model_to_save.save_pretrained(str(save_dir))
        tokenizer.save_pretrained(str(save_dir))
        print(f"  checkpoint saved → {save_dir}  (step {opt_step})")

    while opt_step < args.num_steps:
        if sampler is not None:
            sampler.set_epoch(opt_step)
        pbar = tqdm(dl, desc=f"actor  step={opt_step}/{args.num_steps}",
                    disable=not env.is_main_process)
        any_data = False
        accum_loss = 0.0
        optimizer.zero_grad()

        for messages_batch, responses_batch, vectors_batch in pbar:
            any_data = True
            if opt_step >= args.num_steps:
                break

            model_fwd = model.module if args.ddp else model

            # Build [prompt | response] conversations and tokenize
            texts = []
            for msgs, resp in zip(messages_batch, responses_batch):
                prompt_str = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                )
                texts.append(prompt_str + resp)

            prompt_only = [
                tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                for msgs in messages_batch
            ]

            prompt_enc = tokenizer(
                prompt_only, padding=True, truncation=True,
                max_length=args.max_length, return_tensors="pt",
            )
            full_enc = tokenizer(
                texts, padding=True, truncation=True,
                max_length=args.max_length, return_tensors="pt",
            )

            input_ids = full_enc["input_ids"].to(env.device)
            attention_mask = full_enc["attention_mask"].to(env.device)

            # Labels: -100 for prompt tokens, actual ids for response
            labels = input_ids.clone()
            prompt_lens = prompt_enc["attention_mask"].sum(dim=1)
            for b in range(len(prompt_lens)):
                labels[b, :prompt_lens[b]] = -100

            # Normalize activation vectors
            vectors = vectors_batch.to(env.device)
            vectors = normalize_activation(vectors, injection_scale)

            # Register injection hook
            embed = model_fwd.get_input_embeddings()

            def _make_hook(_input_ids, _vectors, _inj_id, _left_id, _right_id):
                def _hook(module, args, output):
                    return inject_at_marked_positions(
                        _input_ids, output, _vectors,
                        _inj_id, _left_id, _right_id,
                    )
                return _hook

            hook = embed.register_forward_hook(
                _make_hook(input_ids, vectors, inj_id, left_id, right_id)
            )

            try:
                with torch.autocast(device_type=env.device.type, dtype=env.dtype,
                                    enabled=env.amp_enabled):
                    outputs = model_fwd(input_ids=input_ids, attention_mask=attention_mask)
                    loss = sft_loss(outputs.logits, labels)
                    loss = loss / grad_accum
            finally:
                hook.remove()

            loss.backward()
            accum_loss += loss.item() * grad_accum
            micro_step += 1

            # Optimizer step after grad_accum micro-batches
            if (micro_step % grad_accum == 0) or (opt_step >= args.num_steps - 1 and accum_loss > 0):
                optimizer.step()
                optimizer.zero_grad()
                if lr_scheduler is not None:
                    lr_scheduler.step()

                losses.append(accum_loss)
                current_lr = optimizer.param_groups[0]["lr"]
                if env.is_main_process:
                    pbar.set_postfix(loss=f"{accum_loss:.4f}", lr=f"{current_lr:.2e}")
                accum_loss = 0.0
                opt_step += 1

                if env.is_mps and opt_step % 5 == 0:
                    torch.mps.empty_cache()

                if opt_step % args.save_every == 0 and opt_step > 0:
                    _save()

            if opt_step >= args.num_steps:
                break

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
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data", required=True, help="AV-SFT training parquet")
    p.add_argument("--model-name", required=True, help="HF base model")
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
    p.add_argument("--max-length", type=int, default=2048)
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
