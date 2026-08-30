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

        # Direct Arrow → numpy (avoids Python-float overhead)
        col = table.column(ACTIVATION_COLUMN)
        chunked = col.combine_chunks()
        d_model = chunked.type.list_size
        self.vectors = chunked.values.to_numpy().reshape(-1, d_model)
        self.d_model = d_model
        del table  # free Arrow table

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

    # ---- DDP / FSDP setup ----------------------------------------------------
    use_fsdp_init = getattr(args, "fsdp", False)
    if (args.ddp or use_fsdp_init) and env.world_size > 1:
        dist.init_process_group(backend="nccl")
        print(f"[rank {env.global_rank}/{env.world_size}] "
              f"device={env.device}  gpu={env.gpu_name}")
        torch.cuda.set_device(env.local_rank)
    elif args.ddp:
        print("[ddp] WARNING: --ddp passed but WORLD_SIZE=1 — running single-GPU")
        args.ddp = False

    # fp32 master weights; FSDP MixedPrecision casts to bf16 for compute only.
    # bf16 masters silently drop optimizer steps below half a bf16 ulp (at |w|~0.02
    # the ulp is 6.1e-5 vs an AdamW step of ~lr=2e-5) -> backbone never moves.
    param_dtype = getattr(torch, getattr(args, "param_dtype", "float32"))
    if env.is_main_process:
        print(f"device: {env.device}  dtype: {env.dtype}  "
              f"ddp: {args.ddp}  fsdp: {use_fsdp_init}  world_size: {env.world_size}")

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
    if args.injection_scale is not None:
        injection_scale = resolve_target_scale(args.injection_scale, d_model)
    if injection_scale is None:
        # The original asserts here rather than defaulting -- it is a training
        # hyperparameter picked as a round number just above the mean L2 norm of
        # the dataset's vectors (ours: 268 -> 300). The old 2.5*sqrt(d_model)
        # fallback gave 160, i.e. 60% of the mean, silently.
        raise SystemExit(
            "injection_scale is required: pass --injection-scale (e.g. 300, "
            "'raw', 'sqrt_d_model') or set extraction.injection_scale in the "
            "sidecar. It is a training hyperparameter -- pick it explicitly."
        )
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

    # Keyed on use_dist, NOT args.ddp: the slurm scripts pass --fsdp alone, so
    # `if args.ddp` left every rank iterating the FULL dataset instead of its
    # 1/world_size shard -- 4x the intended work for 1x the data coverage.
    use_dist = (args.ddp or use_fsdp_init) and env.world_size > 1
    sampler = DistributedSampler(ds, num_replicas=env.world_size,
                                  rank=env.global_rank,
                                  shuffle=True) if use_dist else None
    dl = DataLoader(ds, batch_size=args.micro_batch_size,
                    shuffle=(sampler is None),
                    sampler=sampler,
                    drop_last=use_dist,  # avoid uneven batch across ranks
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
            resume_path, torch_dtype=param_dtype,
            device_map={"": env.device} if not args.ddp else None,
        )
    else:
        if env.is_main_process:
            print(f"loading {args.model_name} ...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=param_dtype,
            device_map={"": env.device} if not args.ddp else None,
        )
    if env.is_mps:
        model = model.to(env.device)
    if args.ddp or args.fsdp:
        model = model.to(env.device)
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    use_fsdp = getattr(args, "fsdp", False)
    if use_fsdp:
        from functools import partial
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
            MixedPrecision,
        )
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

        auto_wrap = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Qwen3DecoderLayer,},
        )
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=auto_wrap,
            mixed_precision=mixed_precision,
            device_id=env.local_rank if torch.cuda.is_available() else None,
        )
    elif args.ddp:
        # find_unused_parameters=True because injection hook may skip some params
        model = DDP(model, device_ids=[env.local_rank] if torch.cuda.is_available() else None,
                    find_unused_parameters=True)

    if env.is_main_process:
        m = model.module if (args.ddp and not use_fsdp) else model
        print(f"actor: {m.config.num_hidden_layers} layers  d_model={m.config.hidden_size}  "
              f"sharding={'FSDP' if use_fsdp else 'DDP' if args.ddp else 'none'}")

    if args.ddp and env.world_size > 1:
        from torch.distributed.optim import ZeroRedundancyOptimizer
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=torch.optim.AdamW,
            lr=args.lr,
        )
    else:
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
        save_dir = Path(args.output_dir)
        if use_fsdp:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            # ALL ranks must enter the context — FSDP state dict is collective.
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
                full_sd = model.state_dict()
            if env.is_main_process:
                # Masters stay fp32 in memory; the ARTIFACT is bf16 (31GB -> 16GB),
                # matching the critic checkpoint and what RL/eval load anyway.
                full_sd = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v)
                           for k, v in full_sd.items()}
                model.module.save_pretrained(str(save_dir), state_dict=full_sd)
                tokenizer.save_pretrained(str(save_dir))
                print(f"  checkpoint saved → {save_dir}  (step {opt_step})")
        else:
            if env.is_main_process:
                model_to_save = model.module if args.ddp else model
                model_to_save.save_pretrained(str(save_dir))
                tokenizer.save_pretrained(str(save_dir))
                print(f"  checkpoint saved → {save_dir}  (step {opt_step})")

    def _batch_loss(messages_batch, responses_batch, vectors_batch, model_fwd,
                    vec_mode="correct"):
        """Tokenize a batch, inject vectors at the marked position, return CE loss.

        Shared by the training step and the held-out eval so the two can never
        drift apart in template, padding, or injection scale.
        """
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
        # Right-padding is scored otherwise: pad == eos is trivially predictable,
        # so ~15% of targets were free wins that flattered the reported CE.
        # --legacy-unmasked-loss reproduces the pre-2026-08-23 objective EXACTLY
        # (pad scored as targets + ce.mean() over all positions). It exists to
        # settle one question: was the null conditioning gap caused by the loss
        # bug, or simply by stopping at too few steps? Never use it for a real run.
        if not getattr(args, "legacy_unmasked_loss", False):
            labels[attention_mask == 0] = -100

        # sft_loss() without a mask returns ce.mean() over ALL positions, and
        # cross_entropy(reduction="none") writes 0.0 at every -100 slot. That
        # divides by prompt+response+pad instead of response alone -- response is
        # ~51% of tokens, so the loss (and its gradient) came out ~2x too small.
        # The original passes --loss-mask-type qwen for exactly this reason.
        loss_mask = None if getattr(args, "legacy_unmasked_loss", False) \
            else (labels != -100).float()

        # vec_mode drives the conditioning-gap eval. "zeros" must NOT go through
        # normalize_activation: a zero row has norm 0, which clamps to 1e-12 and
        # then divides into a huge vector -- the opposite of "no information".
        if vec_mode == "zeros":
            vectors = torch.zeros(vectors_batch.shape[0], vectors_batch.shape[1],
                                  device=env.device, dtype=torch.float32)
        else:
            vb = vectors_batch if vec_mode == "correct" else vectors_batch.roll(1, 0)
            vectors = normalize_activation(vb.to(env.device), injection_scale)

        embed = model_fwd.get_input_embeddings()

        def _hook(module, _args, output):
            return inject_at_marked_positions(
                input_ids, output, vectors, inj_id, left_id, right_id,
            )

        hook = embed.register_forward_hook(_hook)
        try:
            with torch.autocast(device_type=env.device.type, dtype=env.dtype,
                                enabled=env.amp_enabled):
                outputs = model_fwd(input_ids=input_ids, attention_mask=attention_mask)
                return sft_loss(outputs.logits, labels, loss_mask)
        finally:
            hook.remove()

    # ---- held-out eval -------------------------------------------------------
    eval_dl = None
    if args.eval_data:
        eval_ds = ActorDataset(resolve_parquet(args.eval_data), injection_char)
        n_eval = min(args.eval_samples, len(eval_ds))
        # Fixed subset, same on every rank and every eval -> steps are comparable.
        eval_idx = np.random.default_rng(0).permutation(len(eval_ds))[:n_eval].tolist()
        eval_dl = DataLoader(torch.utils.data.Subset(eval_ds, eval_idx),
                             batch_size=args.eval_batch_size, shuffle=False,
                             collate_fn=_collate)
        if env.is_main_process:
            print(f"eval: {n_eval} held-out rows from {args.eval_data}")

    def _run_eval(tag):
        """Held-out CE, plus the conditioning gap when --eval-gap is set.

        CE alone cannot tell "learned to read the vector" from "learned the
        marginal distribution of explanations": the pre-2026-08-23 actor trained
        to a healthy CE curve with a gap of +0.0013. The gap is the metric that
        catches that, so log it per eval rather than only at the end.
        """
        if eval_dl is None:
            return None
        model.eval()
        modes = ("correct", "shuffled", "zeros") if args.eval_gap else ("correct",)
        tot = {m: 0.0 for m in modes}
        nb = 0
        with torch.no_grad():
            for mb, rb, vb in eval_dl:
                fwd = model.module if (args.ddp and not use_fsdp) else model
                for m in modes:
                    tot[m] += _batch_loss(mb, rb, vb, fwd, vec_mode=m).item()
                nb += 1
        model.train()
        avg = {m: tot[m] / max(nb, 1) for m in modes}
        if env.is_main_process:
            line = (f"  [eval] {tag}: held-out CE = {avg['correct']:.4f}  "
                    f"(ppl {math.exp(min(avg['correct'], 20)):.2f})")
            if args.eval_gap:
                gap = avg["shuffled"] - avg["correct"]
                zgap = avg["zeros"] - avg["correct"]
                line += (f"  | shuf={avg['shuffled']:.4f} zero={avg['zeros']:.4f}"
                         f"  GAP={gap:+.4f}  zero-gap={zgap:+.4f}")
            print(line, flush=True)
        return avg["correct"]

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

            model_fwd = model.module if (args.ddp and not getattr(args, 'fsdp', False)) else model

            loss = _batch_loss(messages_batch, responses_batch,
                               vectors_batch, model_fwd) / grad_accum

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

                if args.eval_every and opt_step % args.eval_every == 0 and opt_step > 0:
                    _run_eval(f"step {opt_step}")

                if opt_step % args.save_every == 0 and opt_step > 0:
                    _save()

            if opt_step >= args.num_steps:
                break

        if not any_data:
            if env.is_main_process:
                print("  DataLoader exhausted — stopping.")
            break

    # ---- final save ----------------------------------------------------------
    if use_dist:
        dist.barrier()
    _run_eval(f"final (step {opt_step})")
    avg_loss = sum(losses) / len(losses) if losses else 0
    if env.is_main_process:
        print(f"\nfinal loss: {avg_loss:.4f}  ({len(losses)} steps)")
    # NOT inside is_main_process: FSDP's FULL_STATE_DICT gather is a COLLECTIVE.
    # Rank-0-only entry deadlocks until the other ranks exit and NCCL SIGABRTs.
    # _save() writes on rank 0 only, internally.
    _save()

    if use_dist:
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
    p.add_argument("--legacy-unmasked-loss", action="store_true",
                   help="DEBUG ONLY: restore the pre-fix objective (pad scored, "
                        "CE averaged over all positions). Controlled experiment "
                        "for the loss-bug-vs-too-few-steps question.")
    p.add_argument("--injection-scale", type=str, default=None,
                   help="L2 norm to rescale activation vectors to before "
                        "injection. Float, 'raw', or 'sqrt_d_model'. Overrides "
                        "the sidecar. Rule of thumb: a round number just above "
                        "the dataset's mean vector norm.")
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
    p.add_argument("--param-dtype", default="float32",
                   choices=["float32", "bfloat16"],
                   help="master weight dtype (float32 = fp32 masters + bf16 compute)")
    p.add_argument("--eval-data", default=None,
                   help="held-out AV-SFT parquet for periodic CE eval")
    p.add_argument("--eval-every", type=int, default=0,
                   help="run held-out eval every N optimizer steps (0 = off)")
    p.add_argument("--eval-samples", type=int, default=1024)
    p.add_argument("--eval-gap", action="store_true",
                   help="also log the conditioning gap (shuffled/zeros vector CE) "
                        "at every eval -- 3x eval cost, negligible vs step time")
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--fsdp", action="store_true",
                   help="enable FullyShardedDataParallel / ZeRO-3 (use with torchrun)")
    args = p.parse_args()

    # Defaults
    if args.min_lr is None:
        args.min_lr = args.lr
    if args.global_batch_size is None:
        args.global_batch_size = args.micro_batch_size

    train(args)


if __name__ == "__main__":
    main()
