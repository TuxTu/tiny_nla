"""GRPO training — on-policy RL with SGLang/HF generate for rollout, critic reward, actor+critic updates.

Supports both single-GPU and multi-GPU DDP training.

Usage (single-GPU, test):
  python -m nla.training.train_rl \
    --data data/test/rl_train.parquet \
    --model-name Qwen/Qwen3-0.6B \
    --actor-ckpt data/test/actor_checkpoint \
    --critic-ckpt data/test/critic_checkpoint \
    --output-dir data/test/rl_checkpoint \
    --n-samples 4 --rollout-batch 2 --micro-batch-size 1 --num-steps 2

Usage (multi-GPU DDP + SGLang on dedicated GPU):
  # Reserve GPU 0 for SGLang, GPUs 1-7 for DDP training
  CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node=7 \
    -m nla.training.train_rl \
    --data data/rl_train.parquet \
    --model-name Qwen/Qwen3-4B \
    --actor-ckpt checkpoints/actor_sft \
    --critic-ckpt checkpoints/critic_sft \
    --output-dir checkpoints/rl \
    --n-samples 8 --rollout-batch 32 --global-batch-size 256 \
    --lr-actor 1.41e-5 --lr-critic 1.41e-5 \
    --kl-coef 0.01 --max-response-len 150 --num-steps 3900 \
    --ddp --use-sglang --sglang-gpu-id 0 --sglang-mem-fraction 0.70
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from nla.training.env_config import detect, EnvConfig
from nla.training.injection import inject_at_marked_positions
from nla.training.loss import nla_critic_loss
from nla.training.models import NLACriticModel
from nla.training.schema import (
    ACTIVATION_COLUMN,
    INJECT_PLACEHOLDER,
    extract_explanation,
    normalize_activation,
    resolve_target_scale,
    wrap_explanation,
)
from nla.training.sidecar import read_sidecar


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------


class RLDataset(Dataset):
    """RL parquet → (messages, activation_vector) pairs. No response column."""

    def __init__(self, parquet_path: str, injection_char: str):
        from nla.training.resolve import resolve_parquet
        table = pq.read_table(resolve_parquet(parquet_path))
        raw_prompts = table.column("prompt").to_pylist()
        col = table.column(ACTIVATION_COLUMN)
        self.vectors = np.array([v.as_py() for v in col], dtype=np.float32)

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
        return self.prompts[idx], torch.from_numpy(self.vectors[idx])


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------


def _gather_tensors_across_ranks(tensor: torch.Tensor, env: EnvConfig,
                                 dst: int = 0) -> list[torch.Tensor] | None:
    """Gather tensors from all ranks to dst rank. Returns list on dst, None elsewhere."""
    if not env.ddp_enabled:
        return [tensor]
    local_size = torch.tensor([tensor.shape[0]], device=env.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(env.world_size)]
    dist.all_gather(all_sizes, local_size)

    # Pad to max size for all_gather
    max_size = max(s.item() for s in all_sizes)
    pad = max_size - tensor.shape[0]
    if pad > 0:
        tensor = torch.cat([tensor, torch.zeros(pad, *tensor.shape[1:],
                                                 device=env.device, dtype=tensor.dtype)])
    gathered = [torch.zeros_like(tensor) for _ in range(env.world_size)]
    dist.all_gather(gathered, tensor)

    # Trim padding
    result = [gathered[i][:all_sizes[i].item()] for i in range(env.world_size)]
    if env.global_rank == dst:
        return result
    return None


def _broadcast_object(obj, env: EnvConfig, src: int = 0):
    """Broadcast a picklable Python object from src to all ranks."""
    if not env.ddp_enabled:
        return obj
    buffer = [obj] if env.global_rank == src else [None]
    dist.broadcast_object_list(buffer, src=src)
    return buffer[0]


def _truncate_to_cross_rank_min(tokens_list: list, env) -> int:
    """All-reduce valid-sample count to cross-rank MIN and truncate.

    After filtering (e.g. skipping invalid explanations), each rank may have a
    different number of valid samples.  Different counts → different forward-pass
    counts → DDP gradient-allreduce desync → NCCL timeout (hang).

    Returns n_min: the count to use (same across all ranks after truncation).
    """
    n = torch.tensor([len(tokens_list)], device=env.device, dtype=torch.long)
    dist.all_reduce(n, op=dist.ReduceOp.MIN)
    n_min = int(n.item())
    if n_min == 0:
        # If any rank has zero valid, all ranks skip this step
        return 0
    # Truncate in-place
    del tokens_list[n_min:]
    return n_min


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

    # ---- sidecar -------------------------------------------------------------
    sidecar = read_sidecar(data_path)
    tokens = sidecar.get("tokens", {})
    injection_char = tokens["injection_char"]
    inj_id = tokens["injection_token_id"]
    left_id = tokens["injection_left_neighbor_id"]
    right_id = tokens["injection_right_neighbor_id"]
    d_model = sidecar["extraction"]["d_model"]
    if env.is_main_process:
        print(f"injection: char={injection_char!r}  id={inj_id}  d_model={d_model}")

    mse_scale = resolve_target_scale(
        sidecar.get("extraction", {}).get("mse_scale", "sqrt_d_model"), d_model,
    )
    injection_scale = resolve_target_scale(
        sidecar.get("extraction", {}).get("injection_scale"), d_model,
    )
    if injection_scale is None:
        injection_scale = 2.5 * math.sqrt(d_model)
    if env.is_main_process:
        print(f"mse_scale={mse_scale}  injection_scale={injection_scale}")

    critic_template = sidecar.get("prompt_templates", {}).get(
        "critic", "Summary of the following text: <text>{explanation}</text> <summary>"
    )

    # ---- tokenizer -----------------------------------------------------------
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # for generation
    tokenizer.truncation_side = "right"

    # ---- data ----------------------------------------------------------------
    ds = RLDataset(data_path, injection_char)
    if env.is_main_process:
        print(f"dataset: {len(ds)} rows  d_model={d_model}")
    # Effective rollout batch: each rank contributes rollout_batch prompts
    # When DDP is enabled, total prompts per step = rollout_batch * world_size
    effective_rollout_batch = args.rollout_batch
    if args.ddp:
        effective_rollout_batch = args.rollout_batch  # per-rank batch size

    def _collate(batch):
        prompts, vectors = zip(*batch)
        return list(prompts), torch.stack(vectors)

    # In DDP mode: use DistributedSampler so each rank gets different prompts
    sampler = DistributedSampler(ds, num_replicas=env.world_size,
                                  rank=env.global_rank,
                                  shuffle=True) if args.ddp else None
    dl = DataLoader(ds, batch_size=effective_rollout_batch,
                    shuffle=(sampler is None),
                    sampler=sampler,
                    drop_last=args.ddp,
                    collate_fn=_collate)

    # ---- actor model ---------------------------------------------------------
    from transformers import AutoModelForCausalLM
    actor_ckpt = args.actor_ckpt
    if args.resume:
        resume_actor = Path(args.output_dir) / "actor"
        if resume_actor.joinpath("config.json").exists():
            actor_ckpt = str(resume_actor.resolve())
            if env.is_main_process:
                print(f"resuming actor from {actor_ckpt} ...")
        else:
            if env.is_main_process:
                print(f"no RL actor checkpoint at {resume_actor} — starting from scratch")
    else:
        if env.is_main_process:
            print(f"loading actor from {actor_ckpt} ...")
    actor = AutoModelForCausalLM.from_pretrained(
        actor_ckpt, torch_dtype=env.dtype,
        device_map={"": env.device} if not args.ddp else None,
    )
    if env.is_mps:
        actor = actor.to(env.device)
    if args.ddp:
        actor = actor.to(env.device)
    actor.train()
    if hasattr(actor, "gradient_checkpointing_enable"):
        actor.gradient_checkpointing_enable()

    if args.ddp:
        actor = DDP(actor, device_ids=[env.local_rank] if torch.cuda.is_available() else None,
                    find_unused_parameters=False)

    if env.is_main_process:
        a = actor.module if args.ddp else actor
        print(f"actor: {a.config.num_hidden_layers} layers")

    # ---- reference model (frozen, for KL) ------------------------------------
    ref_model = None
    ref_embed = None
    if args.kl_coef > 0:
        if env.is_main_process:
            print(f"loading reference model from {args.actor_ckpt} ...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.actor_ckpt, torch_dtype=env.dtype,
            device_map={"": env.device} if not args.ddp else None,
        )
        if env.is_mps:
            ref_model = ref_model.to(env.device)
        if args.ddp:
            ref_model = ref_model.to(env.device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
        ref_embed = ref_model.get_input_embeddings()

    # ---- critic model --------------------------------------------------------
    critic_ckpt = args.critic_ckpt
    if args.resume:
        resume_critic = Path(args.output_dir) / "critic"
        if resume_critic.joinpath("value_head.safetensors").exists():
            critic_ckpt = str(resume_critic.resolve())
            if env.is_main_process:
                print(f"resuming critic from {critic_ckpt} ...")
        else:
            if env.is_main_process:
                print(f"no RL critic checkpoint at {resume_critic} — starting from scratch")
    else:
        if env.is_main_process:
            print(f"loading critic from {critic_ckpt} ...")
    critic = NLACriticModel.from_pretrained(
        critic_ckpt,
        torch_dtype=env.dtype,
        device_map={"": env.device} if not args.ddp else None,
    )
    if env.is_mps:
        critic = critic.to(env.device)
    if args.ddp:
        critic = critic.to(env.device)

    critic.train()

    if args.ddp:
        critic = DDP(critic, device_ids=[env.local_rank] if torch.cuda.is_available() else None,
                     find_unused_parameters=False)

    if env.is_main_process:
        c = critic.module if args.ddp else critic
        print(f"critic: {c.config.num_hidden_layers} layers")

    # ---- optimizers (ZeroRedundancyOptimizer shards states across ranks) ----
    # Each rank stores 1/world_size of AdamW states → saves ~46 GB per GPU
    if args.ddp and env.world_size > 1:
        from torch.distributed.optim import ZeroRedundancyOptimizer
        actor_optimizer = ZeroRedundancyOptimizer(
            actor.parameters(),
            optimizer_class=torch.optim.AdamW,
            lr=args.lr_actor,
        )
        critic_optimizer = ZeroRedundancyOptimizer(
            critic.parameters(),
            optimizer_class=torch.optim.AdamW,
            lr=args.lr_critic,
        )
    else:
        actor_optimizer = torch.optim.AdamW(
            actor.parameters(), lr=args.lr_actor,
        )
        critic_optimizer = torch.optim.AdamW(
            critic.parameters(), lr=args.lr_critic,
        )
    if env.is_main_process:
        print(f"actor_lr={args.lr_actor}  critic_lr={args.lr_critic}  "
              f"kl_coef={args.kl_coef}  n_samples={args.n_samples}  "
              f"lr_decay=constant")

    # ---- SGLang rollout server (launched on dedicated GPU if specified) ------
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    actor_save = Path(args.output_dir) / "actor"
    critic_save = Path(args.output_dir) / "critic"

    rollout = None
    if getattr(args, "use_sglang", False):
        from nla.training.sglang_rollout import (
            SGLangRollout,
            prepare_batch_embeddings,
        )

        sglang_gpu = getattr(args, "sglang_gpu_id", None)

        # Initial save so SGLang has weights to load (rank 0 only)
        if env.is_main_process:
            _actor_save = actor.module if args.ddp else actor
            _actor_save.save_pretrained(str(actor_save))
            tokenizer.save_pretrained(str(actor_save))

        # ALL ranks create a rollout object — each rank calls SGLang
        # independently for per-rank generation (no cross-rank gather needed).
        rollout = SGLangRollout(
            str(actor_save),
            mem_fraction=getattr(args, "sglang_mem_fraction", 0.70),
            gpu_id=sglang_gpu,
        )
        if getattr(args, "sglang_external", False):
            # SGLang was launched externally (e.g. by SLURM script on GPU 0
            # before torchrun). All ranks verify it's healthy.
            rollout.wait_ready()
            if env.is_main_process:
                print(f"SGLang (external) ready on GPU {sglang_gpu}.")
        else:
            # Only rank 0 launches SGLang; other ranks wait for it
            if env.is_main_process:
                rollout.start()
                print(f"SGLang server started on GPU {sglang_gpu} (will hot-reload weights each step).")
            else:
                rollout.wait_ready()

    # ---- training loop -------------------------------------------------------
    global_step = 0
    actor_losses = []
    critic_losses = []
    reward_history = []

    def _save():
        if not env.is_main_process:
            return
        _actor_save = actor.module if args.ddp else actor
        _critic_save = critic.module if args.ddp else critic
        _actor_save.save_pretrained(str(actor_save))
        _critic_save.save_pretrained(str(critic_save))
        tokenizer.save_pretrained(str(actor_save))
        print(f"  checkpoint saved → {args.output_dir}  (step {global_step})")

    while global_step < args.num_steps:
        if sampler is not None:
            sampler.set_epoch(global_step)
        pbar = tqdm(dl, desc=f"rl  step={global_step}/{args.num_steps}",
                    disable=not env.is_main_process)
        any_data = False
        for messages_batch, vectors_batch in pbar:
            any_data = True
            if global_step >= args.num_steps:
                break

            B = len(messages_batch)
            N = args.n_samples  # responses per prompt
            vectors = vectors_batch.to(env.device)
            vectors = normalize_activation(vectors, injection_scale)

            # ================================================================
            # 1. GENERATE N responses per prompt
            # ================================================================
            prompt_texts = [
                tokenizer.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True)
                for msgs in messages_batch
            ]

            # Each rank independently generates via SGLang (no gather-broadcast).
            # Vectors are already normalized at line 397.
            if rollout is not None:
                # ---- SGLang rollout (per-rank, no cross-rank comm) -------------
                embed_layer = (actor.module if args.ddp else actor).get_input_embeddings()
                embeds_list = []
                for i, (msgs, vec) in enumerate(zip(messages_batch, vectors)):
                    for _ in range(N):
                        emb = prepare_batch_embeddings(
                            tokenizer, [msgs], vec.unsqueeze(0),
                            embed_layer,
                            injection_char=injection_char,
                            inj_token_id=inj_id,
                            left_neighbor_id=left_id,
                            right_neighbor_id=right_id,
                            injection_scale=None,  # already normalized
                            max_length=args.max_length,
                            device=env.device,
                        )
                        embeds_list.append(emb[0])

                sglang_responses = rollout.generate(
                    embeds_list,
                    max_new_tokens=args.max_response_len,
                    temperature=1.0,
                )

                all_responses = []
                all_tokens = []
                for idx, (ptext, sglang_text) in enumerate(
                    zip(
                        [pt for pt in prompt_texts for _ in range(N)],
                        sglang_responses,
                    )
                ):
                    all_responses.append(sglang_text)
                    full_text = ptext + sglang_text
                    full_ids = tokenizer(
                        full_text, return_tensors="pt",
                        truncation=True, max_length=args.max_length,
                    )["input_ids"][0]
                    all_tokens.append(full_ids)

                del embeds_list

            else:
                # ---- HF generate (default, single-GPU) -----------------------
                _actor = actor.module if args.ddp else actor
                _actor.eval()
                all_responses = []
                all_tokens = []
                with torch.no_grad():
                    for i, prompt_text in enumerate(prompt_texts):
                        prompt_enc = tokenizer(
                            prompt_text, return_tensors="pt",
                            truncation=True, max_length=args.max_length,
                        )
                        prompt_ids = prompt_enc["input_ids"].to(env.device)
                        vec = vectors[i:i+1]

                        def _make_gen_hook(_vec, _inj_id, _left_id, _right_id):
                            def _hook(module, args, output):
                                actual_ids = args[0]
                                if output.shape[1] > 1:
                                    return inject_at_marked_positions(
                                        actual_ids, output, _vec.repeat(output.shape[0], 1),
                                        _inj_id, _left_id, _right_id,
                                    )
                                return output
                            return _hook

                        embed = _actor.get_input_embeddings()
                        hook = embed.register_forward_hook(
                            _make_gen_hook(vec, inj_id, left_id, right_id)
                        )

                        try:
                            gen_out = _actor.generate(
                                prompt_ids,
                                max_new_tokens=args.max_response_len,
                                do_sample=True,
                                temperature=1.0,
                                num_return_sequences=N,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                            )
                        finally:
                            hook.remove()

                        for s in range(N):
                            seq = gen_out[s]
                            resp_ids = seq[prompt_ids.shape[1]:]
                            resp_text = tokenizer.decode(resp_ids, skip_special_tokens=True)
                            all_responses.append(resp_text)
                            all_tokens.append(seq)

                _actor.train()

            actor.train()  # ensure training mode after generation

            # ================================================================
            # 2. REWARD via critic (batched for efficiency)
            # ================================================================
            _critic = critic.module if args.ddp else critic
            rewards = []
            critic_prompts = []
            valid_indices = []
            for s_idx, resp_text in enumerate(all_responses):
                explanation = extract_explanation(resp_text)
                if explanation is None:
                    rewards.append(-1.0)
                    continue
                critic_prompt = critic_template.format(explanation=explanation)
                critic_prompts.append(critic_prompt)
                valid_indices.append(s_idx)

            # Batch critic forward for all valid explanations
            critic_preds = {}
            if critic_prompts:
                critic_enc = tokenizer(
                    critic_prompts, return_tensors="pt",
                    padding=True, truncation=True, max_length=512,
                )
                c_ids = critic_enc["input_ids"].to(env.device)
                c_mask = critic_enc["attention_mask"].to(env.device)
                with torch.no_grad():
                    c_out = _critic(input_ids=c_ids, attention_mask=c_mask)
                    seq_lens = c_mask.sum(dim=1) - 1
                    preds = c_out.values[torch.arange(len(seq_lens)), seq_lens]  # [V, d]

                for i, s_idx in enumerate(valid_indices):
                    gold_idx = s_idx // N
                    if gold_idx >= len(vectors):
                        continue  # safety: scatter may have fewer vectors than expected
                    gold_vec = vectors[gold_idx]
                    mse = F.mse_loss(
                        normalize_activation(preds[i:i+1], mse_scale),
                        normalize_activation(gold_vec.unsqueeze(0), mse_scale),
                    )
                    if s_idx < len(rewards):
                        rewards[s_idx] = -mse.item()

            # Pad rewards to B*N in case SGLang returned fewer responses than expected
            while len(rewards) < B * N:
                rewards.append(-1.0)
            rewards_t = torch.tensor(rewards[:B * N], device=env.device).float()
            reward_history.append(rewards_t.mean().item())

            # ================================================================
            # 3. GRPO ADVANTAGES
            # ================================================================
            r = rewards_t.reshape(B, N)  # [B, N]
            mean_r = r.mean(dim=1, keepdim=True)
            std_r = r.std(dim=1, keepdim=True).clamp_min(1e-8)
            advantages = ((r - mean_r) / std_r).reshape(-1)  # [B*N]

            # ================================================================
            # 4. ACTOR UPDATE (policy gradient + optional KL)
            # ================================================================
            _actor = actor.module if args.ddp else actor
            actor_embed = _actor.get_input_embeddings()

            actor_loss_sum = 0.0
            actor_optimizer.zero_grad()

            for s in range(B * N):
                seq = all_tokens[s].to(env.device).unsqueeze(0)
                attn = torch.ones_like(seq)
                parent_idx = s // N
                vec = vectors[parent_idx:parent_idx + 1]

                def _make_actor_hook(_ids, _vec, _inj, _left, _right):
                    def _hook(_module, _args, output):
                        return inject_at_marked_positions(
                            _ids, output, _vec, _inj, _left, _right,
                        )
                    return _hook

                actor_hook = actor_embed.register_forward_hook(
                    _make_actor_hook(seq, vec, inj_id, left_id, right_id)
                )
                try:
                    with torch.autocast(device_type=env.device.type, dtype=env.dtype,
                                        enabled=env.amp_enabled):
                        out = _actor(input_ids=seq, attention_mask=attn, use_cache=False)
                        logits = out.logits

                        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
                        target_ids = seq[:, 1:]
                        token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
                        seq_log_prob = token_log_probs.sum()

                        kl_penalty = torch.tensor(0.0, device=env.device)
                        if ref_model is not None:
                            with torch.no_grad():
                                ref_hook = ref_embed.register_forward_hook(
                                    _make_actor_hook(seq, vec, inj_id, left_id, right_id)
                                )
                                try:
                                    ref_out = ref_model(input_ids=seq, attention_mask=attn,
                                                       use_cache=False)
                                finally:
                                    ref_hook.remove()
                                ref_log_probs = F.log_softmax(ref_out.logits[:, :-1, :], dim=-1)
                                ref_token_log_probs = ref_log_probs.gather(
                                    -1, target_ids.unsqueeze(-1)
                                ).squeeze(-1)
                                ref_seq_log_prob = ref_token_log_probs.sum()
                                kl_penalty = seq_log_prob - ref_seq_log_prob

                        pg_loss = -(advantages[s] * seq_log_prob)
                        loss = pg_loss + args.kl_coef * kl_penalty
                finally:
                    actor_hook.remove()

                loss.backward()
                actor_loss_sum += loss.detach().item()

            # Average gradients across DDP ranks (handled by DDP automatically)
            actor_optimizer.step()
            actor_losses.append(actor_loss_sum / (B * N))

            # ---- sync updated weights to SGLang for next step ----------
            if rollout is not None and env.is_main_process:
                _actor_save = actor.module if args.ddp else actor
                _actor_save.save_pretrained(str(actor_save))
                tokenizer.save_pretrained(str(actor_save))
                rollout.update_weights(str(actor_save))

            # Barrier: ensure weight sync finishes before next generation
            if args.ddp and rollout is not None:
                dist.barrier()

            # ================================================================
            # 5. CRITIC UPDATE (batched MSE)
            # ================================================================
            critic_loss_sum = 0.0
            critic_optimizer.zero_grad()

            critic_batch_inputs = []
            critic_batch_gold = []
            for s in range(B * N):
                explanation = extract_explanation(all_responses[s])
                if explanation is None:
                    continue
                critic_prompt = critic_template.format(explanation=explanation)
                critic_enc = tokenizer(
                    critic_prompt, return_tensors="pt",
                    truncation=True, max_length=512,
                )
                critic_batch_inputs.append((critic_enc["input_ids"][0], critic_enc["attention_mask"][0]))
                gold_idx = s // N
                critic_batch_gold.append(vectors[gold_idx])

            # Cross-rank min truncation: ensures all ranks have the same number
            # of valid critic samples → same forward-pass count → no DDP hang.
            if args.ddp and env.world_size > 1:
                n_critic = _truncate_to_cross_rank_min(critic_batch_inputs, env)
                if n_critic > 0:
                    del critic_batch_gold[n_critic:]
                critic_batch_count = n_critic
            else:
                critic_batch_count = len(critic_batch_inputs)

            if critic_batch_count > 0:
                # Pad and batch
                c_ids_batch = torch.nn.utils.rnn.pad_sequence(
                    [ci[0] for ci in critic_batch_inputs], batch_first=True, padding_value=0
                ).to(env.device)
                c_mask_batch = torch.nn.utils.rnn.pad_sequence(
                    [ci[1] for ci in critic_batch_inputs], batch_first=True, padding_value=0
                ).to(env.device)
                gold_batch = torch.stack(critic_batch_gold).to(env.device)

                with torch.autocast(device_type=env.device.type, dtype=env.dtype,
                                    enabled=env.amp_enabled):
                    _critic_fwd = critic.module if args.ddp else critic
                    c_out = _critic_fwd(input_ids=c_ids_batch, attention_mask=c_mask_batch)
                    seq_lens = c_mask_batch.sum(dim=1) - 1
                    pred = c_out.values[torch.arange(len(seq_lens)), seq_lens]
                    c_loss = nla_critic_loss(pred, gold_batch, mse_scale)

                c_loss.backward()
                critic_loss_sum = c_loss.detach().item()

            critic_optimizer.step()
            critic_losses.append(critic_loss_sum / max(1, B * N))

            if env.is_mps and global_step % 5 == 0:
                torch.mps.empty_cache()

            if global_step % args.save_every == 0 and global_step > 0:
                _save()

            if env.is_main_process:
                pbar.set_postfix(
                    actor_loss=f"{actor_losses[-1]:.4f}",
                    critic_loss=f"{critic_losses[-1]:.4f}",
                    reward=f"{reward_history[-1]:.4f}",
                )

            global_step += 1

        if not any_data:
            if env.is_main_process:
                print("  DataLoader exhausted — stopping.")
            break

    # ---- cleanup --------------------------------------------------------------
    if rollout is not None and env.is_main_process:
        if not getattr(args, "sglang_external", False):
            rollout.stop()
        print("SGLang server stopped.")

    if args.ddp:
        dist.barrier()

    # ---- final save ----------------------------------------------------------
    if env.is_main_process:
        if actor_losses:
            print(f"\nactor_loss: {actor_losses[-1]:.4f}  "
                  f"critic_loss: {critic_losses[-1]:.4f}  "
                  f"reward: {reward_history[-1]:.4f}")
        _save()

    if args.ddp:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="RL training parquet")
    p.add_argument("--model-name", required=True, help="HF base model")
    p.add_argument("--actor-ckpt", required=True, help="actor SFT checkpoint")
    p.add_argument("--critic-ckpt", required=True, help="critic SL checkpoint")
    p.add_argument("--output-dir", required=True, help="save directory")
    p.add_argument("--n-samples", type=int, default=4,
                   help="responses per prompt for GRPO (default: 4)")
    p.add_argument("--rollout-batch", type=int, default=2,
                   help="prompts per generation batch PER RANK (default: 2)")
    p.add_argument("--global-batch-size", type=int, default=None,
                   help="total samples per step (default: rollout_batch * n_samples * world_size)")
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--lr-actor", type=float, default=1.41e-5)
    p.add_argument("--lr-critic", type=float, default=1.41e-5)
    p.add_argument("--kl-coef", type=float, default=0.01,
                   help="KL penalty coefficient (0=disabled)")
    p.add_argument("--max-response-len", type=int, default=150)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--num-steps", type=int, default=2)
    p.add_argument("--save-every", type=int, default=100,
                   help="save checkpoint every N steps (default: 100)")
    p.add_argument("--resume", action="store_true",
                   help="resume from checkpoint in --output-dir")
    p.add_argument("--use-sglang", action="store_true",
                   help="use SGLang for batched rollout generation")
    p.add_argument("--sglang-external", action="store_true",
                   help="SGLang is managed externally (skip launch/shutdown)")
    p.add_argument("--sglang-mem-fraction", type=float, default=0.70,
                   help="GPU memory fraction for SGLang (default: 0.70)")
    p.add_argument("--sglang-gpu-id", type=int, default=None,
                   help="dedicated GPU ID for SGLang (when DDP uses other GPUs)")
    p.add_argument("--ddp", action="store_true",
                   help="enable DistributedDataParallel (use with torchrun)")
    args = p.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.global_batch_size is None:
        # Each DDP rank processes different prompts (DistributedSampler).
        # Total global batch = per_rank_rollout_batch × world_size × n_samples.
        # Original repo enforces: rollout_batch × n_samples == global_batch.
        args.global_batch_size = args.rollout_batch * args.n_samples * world_size

    train(args)


if __name__ == "__main__":
    main()
