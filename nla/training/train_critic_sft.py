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
        table = pq.read_table(resolve_parquet(parquet_path))
        self.prompts = table.column("prompt").to_pylist()
        # Direct Arrow → numpy (avoids Python-float overhead: 498k×4096 floats
        # → ~57 GB of PyObject overhead per process, which OOMs a node).
        col = table.column(ACTIVATION_COLUMN)
        chunked = col.combine_chunks()
        d_model = chunked.type.list_size  # FixedSizeList: uniform element count
        self.vectors = chunked.values.to_numpy().reshape(-1, d_model)
        self.d_model = d_model
        del table  # free the Arrow table (values copy is now in numpy)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx], torch.from_numpy(self.vectors[idx])


# ---------------------------------------------------------------------------
# held-out evaluation (FVE)
# ---------------------------------------------------------------------------


def _norm_rows(v: torch.Tensor, scale: float | None) -> torch.Tensor:
    if scale is None:
        return v
    return v / (v.norm(dim=-1, keepdim=True).clamp_min(1e-12) / scale)


def compute_fve_baselines(gold: torch.Tensor, mse_scale: float | None) -> tuple[float, float]:
    """(meannorm, rawvar) predict-the-mean baselines — mirrors the original's
    nla.schema.compute_predict_mean_baselines."""
    g = _norm_rows(gold.float(), mse_scale)
    mu = g.mean(dim=0, keepdim=True)
    mu_n = _norm_rows(mu, mse_scale)
    return ((g - mu_n) ** 2).mean().item(), ((g - mu) ** 2).mean().item()


@torch.no_grad()
def evaluate(model, tokenizer, prompts, gold, mse_scale, env, args) -> tuple[float, float, float]:
    """Held-out MSE + FVE. Every rank runs the SAME batches — an FSDP forward is
    collective, so ranks cannot diverge here."""
    was_training = model.training
    model.eval()
    preds = []
    for i in range(0, len(prompts), args.eval_batch_size):
        enc = tokenizer(prompts[i:i + args.eval_batch_size], padding=True,
                        truncation=True, max_length=args.max_length, return_tensors="pt")
        input_ids = enc["input_ids"].to(env.device)
        attention_mask = enc["attention_mask"].to(env.device)
        with torch.autocast(device_type=env.device.type, dtype=env.dtype,
                            enabled=env.amp_enabled):
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        seq_lens = attention_mask.sum(dim=1) - 1
        preds.append(out.values[torch.arange(len(seq_lens)), seq_lens].float().cpu())
    if was_training:
        model.train()
    pred_n = _norm_rows(torch.cat(preds), mse_scale)
    gold_n = _norm_rows(gold.float(), mse_scale)
    mse = ((pred_n - gold_n) ** 2).mean().item()
    mn, rv = args._eval_baselines
    return mse, 1.0 - mse / mn, 1.0 - mse / rv


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
    use_fsdp_init = getattr(args, "fsdp", False)
    if (args.ddp or use_fsdp_init) and env.world_size > 1:
        dist.init_process_group(backend="nccl")
        print(f"[rank {env.global_rank}/{env.world_size}] "
              f"device={env.device}  gpu={env.gpu_name}")
        torch.cuda.set_device(env.local_rank)
    elif args.ddp:
        print("[ddp] WARNING: --ddp passed but WORLD_SIZE=1 — running single-GPU")
        args.ddp = False

    # Master weights stay fp32; FSDP MixedPrecision casts to bf16 for compute only.
    # Storing master weights in bf16 (the previous default) silently discards every
    # optimizer step smaller than half a bf16 ulp: at |w|~0.02 the ulp is 6.1e-5, so
    # an AdamW step of ~lr=2e-5 rounds straight back to the original value and the
    # backbone never moves. Measured: 92% of backbone weights were bit-identical to
    # init after 974 steps. The original repo keeps fp32 masters for this reason
    # (nla/train_actor.py:793 "model stored fp32; MixedPrecision is compute-only").
    param_dtype = getattr(torch, args.param_dtype)
    if env.is_main_process:
        print(f"device: {env.device}  compute dtype: {env.dtype}  "
              f"master param dtype: {param_dtype}  "
              f"ddp: {args.ddp}  fsdp: {use_fsdp_init}  world_size: {env.world_size}")

    # ---- resolve data path (may be HF Hub repo) --------------------------------
    data_path = resolve_parquet(args.data)

    # ---- data ----------------------------------------------------------------
    ds = CriticDataset(data_path)
    if env.is_main_process:
        print(f"dataset: {len(ds)} rows  d_model={ds.d_model}")

    # Shard the dataset across ranks for BOTH DDP and FSDP. Previously this was
    # gated on args.ddp alone, so an FSDP run gave every rank the full dataset
    # with an independent shuffle -- ranks saw overlapping samples and the
    # "steps = rows / global_batch = 1 epoch" accounting was wrong.
    use_dist = (args.ddp or use_fsdp_init) and env.world_size > 1
    sampler = DistributedSampler(ds, num_replicas=env.world_size,
                                  rank=env.global_rank,
                                  shuffle=True) if use_dist else None
    dl = DataLoader(ds,
                    batch_size=args.micro_batch_size,
                    shuffle=(sampler is None),
                    sampler=sampler,
                    drop_last=use_dist,  # avoid uneven batch across ranks
                    )

    # ---- sidecar -------------------------------------------------------------
    sidecar = read_sidecar(data_path)
    mse_scale_raw = sidecar.get("extraction", {}).get("mse_scale", "sqrt_d_model")
    mse_scale = resolve_target_scale(mse_scale_raw, ds.d_model)
    if env.is_main_process:
        print(f"mse_scale: {mse_scale}")

    # ---- model ---------------------------------------------------------------
    # An already-prepared critic checkpoint (from a prep script or a prior critic
    # run) has config.num_hidden_layers == K+1 ALREADY. Re-deriving a layer index
    # from that truncated count and truncating again silently halves the critic
    # (25 layers -> (2*25)//3 = 16 -> 17 layers) while the gold vectors still come
    # from layer K of the FULL base model. Detect prepared checkpoints and load
    # them as-is, matching the original repo's contract (nla/models.py:89-94:
    # "config.json already has the truncated num_hidden_layers. Just load.").
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    is_prepared_critic = Path(args.model_name, "value_head.safetensors").exists()

    if is_prepared_critic:
        ckpt_k = cfg.num_hidden_layers - 1
        assert args.layer_index is None or args.layer_index == ckpt_k, (
            f"--layer-index {args.layer_index} disagrees with the prepared critic "
            f"checkpoint at {args.model_name}, which has num_hidden_layers="
            f"{cfg.num_hidden_layers} => K={ckpt_k}. The gold vectors must come "
            f"from layer K of the FULL base model. Fix one of the two."
        )
        layer_index = ckpt_k
        nla_num_layers = None  # already truncated -- load as-is
        if env.is_main_process:
            print(f"prepared critic checkpoint: K={layer_index}  "
                  f"num_hidden_layers={cfg.num_hidden_layers} (loading as-is, no re-truncation)")
    else:
        layer_index = args.layer_index
        if layer_index is None:
            layer_index = (2 * cfg.num_hidden_layers) // 3
            if env.is_main_process:
                print(f"auto layer_index: {layer_index}  (2/3 \u00d7 {cfg.num_hidden_layers})")
        nla_num_layers = layer_index
        if env.is_main_process:
            print(f"base checkpoint: truncating to blocks 0..{layer_index} "
                  f"({layer_index + 1} layers)")

    if args.resume:
        resume_path = str(Path(args.output_dir).resolve())
        assert Path(resume_path, "value_head.safetensors").exists(), (
            f"no critic checkpoint found at {resume_path} — cannot resume"
        )
        if env.is_main_process:
            print(f"resuming critic from {resume_path} ...")
        model = NLACriticModel.from_pretrained(
            resume_path,
            torch_dtype=param_dtype, device_map={"": env.device} if not args.ddp else None,
        )
    else:
        if env.is_main_process:
            print(f"loading {args.model_name} ...")
        model = NLACriticModel.from_pretrained(
            args.model_name, nla_num_layers=nla_num_layers,
            value_head_depth=args.value_head_depth,
            torch_dtype=param_dtype, device_map={"": env.device} if not args.ddp else None,
        )
    if env.is_mps:
        model = model.to(env.device)
    if args.ddp or args.fsdp:
        model = model.to(env.device)
    model.train()
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
        model = DDP(model, device_ids=[env.local_rank] if torch.cuda.is_available() else None,
                    find_unused_parameters=False)

    if env.is_main_process:
        unwrapped = model
        if args.ddp and not use_fsdp:
            unwrapped = model.module
        print(f"critic: {unwrapped.config.num_hidden_layers} "
              f"layers  d_model={unwrapped.config.hidden_size}  "
              f"sharding={'FSDP' if use_fsdp else 'DDP' if args.ddp else 'none'}")

    # ---- optimizer -----------------------------------------------------------
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

    # ---- held-out eval set ----------------------------------------------------
    eval_prompts, eval_gold = None, None
    if args.eval_data:
        eval_ds = CriticDataset(resolve_parquet(args.eval_data))
        n_eval = min(args.eval_samples, len(eval_ds))
        # Deterministic subset, identical on every rank.
        idx = np.random.default_rng(0).permutation(len(eval_ds))[:n_eval]
        eval_prompts = [eval_ds.prompts[i] for i in idx]
        eval_gold = torch.from_numpy(eval_ds.vectors[idx].copy()).float()
        args._eval_baselines = compute_fve_baselines(eval_gold, mse_scale)
        if env.is_main_process:
            print(f"eval set: {n_eval} rows from {args.eval_data}")
            print(f"  baselines  meannorm={args._eval_baselines[0]:.4f}  "
                  f"rawvar={args._eval_baselines[1]:.4f}")
        del eval_ds

    # ---- gradient accumulation ------------------------------------------------
    global_batch = getattr(args, "global_batch_size", args.micro_batch_size * env.world_size)
    grad_accum = max(1, global_batch // (args.micro_batch_size * env.world_size))
    if env.is_main_process:
        print(f"global_batch={global_batch}  micro_batch={args.micro_batch_size}  "
              f"world_size={env.world_size}  grad_accum={grad_accum}")

    # ---- training ------------------------------------------------------------
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    opt_step = 0          # counts optimizer steps
    micro_step = 0         # counts micro-batches for grad_accum
    losses = []

    best_fve = float("-inf")

    def _run_eval(tag: str):
        nonlocal best_fve
        if eval_prompts is None:
            return
        mse, fve_mn, fve_rv = evaluate(
            model, tokenizer, eval_prompts, eval_gold, mse_scale, env, args)
        if env.is_main_process:
            print(f"[eval @ {tag}]  MSE={mse:.4f}  "
                  f"FVE_nrm_meannorm={fve_mn * 100:+.2f}%  FVE_nrm={fve_rv * 100:+.2f}%",
                  flush=True)
        if fve_mn > best_fve:
            best_fve = fve_mn
            if args.save_best and eval_prompts is not None and tag != "step 0 (init)":
                if env.is_main_process:
                    print(f"  new best FVE -> saving checkpoint", flush=True)
                _save()

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

    _run_eval("step 0 (init)")

    while opt_step < args.num_steps:
        if sampler is not None:
            sampler.set_epoch(opt_step)
        pbar = tqdm(dl, desc=f"critic  step={opt_step}/{args.num_steps}",
                    disable=not env.is_main_process)
        any_data = False
        accum_loss = 0.0
        optimizer.zero_grad()

        for prompts, gold_vectors in pbar:
            any_data = True
            if opt_step >= args.num_steps:
                break

            # tokenize
            enc = tokenizer(
                list(prompts), padding=True, truncation=True,
                max_length=args.max_length, return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(env.device)
            attention_mask = enc["attention_mask"].to(env.device)
            gold = gold_vectors.to(env.device)

            model_fwd = model.module if (args.ddp and not getattr(args, 'fsdp', False)) else model
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

                if args.eval_every and opt_step % args.eval_every == 0 and opt_step > 0:
                    _run_eval(f"step {opt_step}")

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
        print(f"best held-out FVE_nrm_meannorm: {best_fve * 100:+.2f}%")
    # NOT inside is_main_process: FSDP's FULL_STATE_DICT gather is a COLLECTIVE.
    # Rank-0-only entry deadlocks until the other ranks exit and NCCL SIGABRTs,
    # which is what silently destroyed every previous run's final checkpoint.
    # _save() writes on rank 0 only, internally.
    _save()

    if use_dist:
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
    p.add_argument("--value-head-depth", type=int, default=1,
                   help="MLP depth for value head (1 = single linear, default)")
    p.add_argument("--save-every", type=int, default=500,
                   help="save checkpoint every N steps (default: 500)")
    p.add_argument("--save-best", action="store_true",
                   help="save whenever held-out FVE improves (requires --eval-data)")
    p.add_argument("--resume", action="store_true",
                   help="resume from checkpoint in --output-dir")
    p.add_argument("--ddp", action="store_true",
                   help="enable DistributedDataParallel (use with torchrun)")
    p.add_argument("--param-dtype", default="float32",
                   choices=["float32", "bfloat16"],
                   help="master weight dtype. float32 (default) + FSDP MixedPrecision "
                        "= bf16 compute with fp32 masters. bfloat16 reproduces the old "
                        "behaviour where sub-ulp optimizer steps are silently dropped.")
    p.add_argument("--eval-data", default=None,
                   help="held-out AR parquet for periodic FVE evaluation")
    p.add_argument("--eval-every", type=int, default=100,
                   help="run held-out eval every N optimizer steps (0 = only at start/end)")
    p.add_argument("--eval-samples", type=int, default=2048,
                   help="rows sampled from --eval-data (deterministic)")
    p.add_argument("--eval-batch-size", type=int, default=32)
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
