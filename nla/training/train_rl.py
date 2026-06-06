"""GRPO training — on-policy RL with HF generate for rollout, critic reward, actor+critic updates.

Usage (test, 0.6B):
  python -m nla.training.train_rl \
    --data data/test/rl_train.parquet \
    --model-name Qwen/Qwen3-0.6B \
    --actor-ckpt data/test/actor_checkpoint \
    --critic-ckpt data/test/critic_checkpoint \
    --output-dir data/test/rl_checkpoint \
    --n-samples 4 --rollout-batch 2 --micro-batch-size 1 --num-steps 2
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from nla.training.env_config import detect
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
        raw_prompts = pq.read_table(parquet_path).column("prompt").to_pylist()
        col = pq.read_table(parquet_path).column(ACTIVATION_COLUMN)
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
# training loop
# ---------------------------------------------------------------------------


def train(args) -> None:
    env = detect()
    print(f"device: {env.device}  dtype: {env.dtype}")

    # ---- sidecar -------------------------------------------------------------
    sidecar = read_sidecar(args.data)
    tokens = sidecar.get("tokens", {})
    injection_char = tokens["injection_char"]
    inj_id = tokens["injection_token_id"]
    left_id = tokens["injection_left_neighbor_id"]
    right_id = tokens["injection_right_neighbor_id"]
    d_model = sidecar["extraction"]["d_model"]
    print(f"injection: char={injection_char!r}  id={inj_id}")

    mse_scale = resolve_target_scale(
        sidecar.get("extraction", {}).get("mse_scale", "sqrt_d_model"), d_model,
    )
    injection_scale = resolve_target_scale(
        sidecar.get("extraction", {}).get("injection_scale"), d_model,
    )
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
    ds = RLDataset(args.data, injection_char)
    print(f"dataset: {len(ds)} rows  d_model={d_model}")

    # ---- actor model ---------------------------------------------------------
    print(f"loading actor from {args.actor_ckpt} ...")
    from transformers import AutoModelForCausalLM
    actor = AutoModelForCausalLM.from_pretrained(
        args.actor_ckpt, torch_dtype=env.dtype,
        device_map={"": env.device} if not env.is_mps else None,
    )
    if env.is_mps:
        actor = actor.to(env.device)
    actor.train()
    if hasattr(actor, "gradient_checkpointing_enable"):
        actor.gradient_checkpointing_enable()
    print(f"actor: {actor.config.num_hidden_layers} layers")

    # ---- reference model (frozen, for KL) ------------------------------------
    ref_model = None
    if args.kl_coef > 0:
        print(f"loading reference model from {args.actor_ckpt} ...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.actor_ckpt, torch_dtype=env.dtype,
            device_map={"": env.device} if not env.is_mps else None,
        )
        if env.is_mps:
            ref_model = ref_model.to(env.device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

    # ---- critic model --------------------------------------------------------
    print(f"loading critic from {args.critic_ckpt} ...")
    # Critic checkpoint already has truncated config — don't pass nla_num_layers
    critic = NLACriticModel.from_pretrained(
        args.critic_ckpt,
        torch_dtype=env.dtype,
        device_map={"": env.device} if not env.is_mps else None,
    )
    if env.is_mps:
        critic = critic.to(env.device)
    critic.train()
    print(f"critic: {critic.config.num_hidden_layers} layers")

    # ---- optimizers ----------------------------------------------------------
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=args.lr_actor)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=args.lr_critic)

    # ---- training loop -------------------------------------------------------
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    actor_save = Path(args.output_dir) / "actor"
    critic_save = Path(args.output_dir) / "critic"
    global_step = 0
    actor_losses = []
    critic_losses = []
    reward_history = []

    def _collate(batch):
        prompts, vectors = zip(*batch)
        return list(prompts), torch.stack(vectors)

    dl = DataLoader(ds, batch_size=args.rollout_batch, shuffle=True,
                    collate_fn=_collate)

    while global_step < args.num_steps:
        pbar = tqdm(dl, desc=f"rl  step={global_step}/{args.num_steps}")
        for messages_batch, vectors_batch in pbar:
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
            all_responses = []
            all_tokens = []

            actor.eval()  # generation mode
            with torch.no_grad():
                for i, prompt_text in enumerate(prompt_texts):
                    # Tokenize prompt
                    prompt_enc = tokenizer(
                        prompt_text, return_tensors="pt",
                        truncation=True, max_length=args.max_length,
                    )
                    prompt_ids = prompt_enc["input_ids"].to(env.device)

                    # Register injection hook for this prompt's generation
                    vec = vectors[i:i+1]  # [1, d]

                    def _make_gen_hook(_vec, _inj_id, _left_id, _right_id):
                        def _hook(module, args, output):
                            # args[0] is the input_ids passed to embed_tokens.
                            # During generate() with num_return_sequences>1, the batch
                            # is duplicated, so use the actual input_ids, not captured.
                            actual_ids = args[0]
                            # Only inject during prefill (seq len matches prompt).
                            # During decode, KV cache is used and only 1 new token embedded.
                            if output.shape[1] > 1:
                                return inject_at_marked_positions(
                                    actual_ids, output, _vec.repeat(output.shape[0], 1),
                                    _inj_id, _left_id, _right_id,
                                )
                            return output
                        return _hook

                    embed = actor.get_input_embeddings()
                    hook = embed.register_forward_hook(
                        _make_gen_hook(vec, inj_id, left_id, right_id)
                    )

                    try:
                        gen_out = actor.generate(
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

                    # Decode responses (strip prompt)
                    for s in range(N):
                        seq = gen_out[s]
                        resp_ids = seq[prompt_ids.shape[1]:]  # strip prompt
                        resp_text = tokenizer.decode(resp_ids, skip_special_tokens=True)
                        all_responses.append(resp_text)
                        all_tokens.append(seq)

            actor.train()  # back to training mode

            # ================================================================
            # 2. REWARD via critic
            # ================================================================
            rewards = []
            for resp_text in all_responses:
                explanation = extract_explanation(resp_text)
                if explanation is None:
                    rewards.append(-1.0)  # failed to parse — penalize
                    continue

                critic_prompt = critic_template.format(explanation=explanation)
                critic_enc = tokenizer(
                    critic_prompt, return_tensors="pt",
                    truncation=True, max_length=512,
                )
                c_ids = critic_enc["input_ids"].to(env.device)
                c_mask = critic_enc["attention_mask"].to(env.device)

                with torch.no_grad():
                    c_out = critic(input_ids=c_ids, attention_mask=c_mask)
                    seq_lens = c_mask.sum(dim=1) - 1
                    pred = c_out.values[0, seq_lens[0]]  # [d]

                # Get gold vector for this response's parent prompt
                gold_idx = len(rewards) // N  # which prompt in the batch
                gold_vec = vectors[gold_idx]

                mse = F.mse_loss(
                    normalize_activation(pred.unsqueeze(0), mse_scale),
                    normalize_activation(gold_vec.unsqueeze(0), mse_scale),
                )
                rewards.append(-mse.item())

            rewards_t = torch.tensor(rewards, device=env.device).float()
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
            actor_loss_sum = 0.0
            for s in range(B * N):
                seq = all_tokens[s].unsqueeze(0)  # [1, T]
                attn = torch.ones_like(seq)

                with torch.autocast(device_type=env.device.type, dtype=env.dtype,
                                    enabled=env.amp_enabled):
                    out = actor(input_ids=seq, attention_mask=attn)
                    logits = out.logits  # [1, T, V]

                    # Log-probs of the generated sequence
                    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
                    target_ids = seq[:, 1:]
                    token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
                    seq_log_prob = token_log_probs.sum()

                    # Reference log-probs for KL
                    kl_penalty = torch.tensor(0.0, device=env.device)
                    if ref_model is not None:
                        with torch.no_grad():
                            ref_out = ref_model(input_ids=seq, attention_mask=attn)
                            ref_logits = ref_out.logits
                            ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
                            ref_token_log_probs = ref_log_probs.gather(
                                -1, target_ids.unsqueeze(-1)
                            ).squeeze(-1)
                            ref_seq_log_prob = ref_token_log_probs.sum()
                            kl_penalty = seq_log_prob - ref_seq_log_prob

                    # Policy gradient: -advantage * log_prob
                    pg_loss = -(advantages[s] * seq_log_prob)
                    loss = pg_loss + args.kl_coef * kl_penalty

                loss.backward()
                actor_loss_sum += loss.detach().item()

            actor_optimizer.step()
            actor_optimizer.zero_grad()
            actor_losses.append(actor_loss_sum / (B * N))

            # ================================================================
            # 5. CRITIC UPDATE (continued MSE)
            # ================================================================
            critic_loss_sum = 0.0
            for s in range(B * N):
                resp_text = all_responses[s]
                explanation = extract_explanation(resp_text)
                if explanation is None:
                    continue

                critic_prompt = critic_template.format(explanation=explanation)
                critic_enc = tokenizer(
                    critic_prompt, return_tensors="pt",
                    truncation=True, max_length=512,
                )
                c_ids = critic_enc["input_ids"].to(env.device)
                c_mask = critic_enc["attention_mask"].to(env.device)

                gold_idx = s // N
                gold_vec = vectors[gold_idx].unsqueeze(0)

                with torch.autocast(device_type=env.device.type, dtype=env.dtype,
                                    enabled=env.amp_enabled):
                    c_out = critic(input_ids=c_ids, attention_mask=c_mask)
                    seq_lens = c_mask.sum(dim=1) - 1
                    pred = c_out.values[torch.arange(1), seq_lens]  # [1, d]
                    c_loss = nla_critic_loss(pred, gold_vec, mse_scale)

                c_loss.backward()
                critic_loss_sum += c_loss.detach().item()

            critic_optimizer.step()
            critic_optimizer.zero_grad()
            critic_losses.append(critic_loss_sum / (B * N))

            if env.is_mps and global_step % 5 == 0:
                torch.mps.empty_cache()

            mean_r = sum(reward_history[-B*N:]) / min(B*N, len(reward_history))
            pbar.set_postfix(
                actor_loss=f"{actor_losses[-1]:.4f}",
                critic_loss=f"{critic_losses[-1]:.4f}",
                reward=f"{mean_r:.4f}",
            )
            global_step += 1

    # ---- save ----------------------------------------------------------------
    print(f"\nactor_loss: {actor_losses[-1]:.4f}  "
          f"critic_loss: {critic_losses[-1]:.4f}  "
          f"reward: {reward_history[-1]:.4f}")
    actor.save_pretrained(str(actor_save))
    critic.save_pretrained(str(critic_save))
    tokenizer.save_pretrained(str(actor_save))
    print(f"saved → {args.output_dir}")


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
                   help="prompts per generation batch (default: 2)")
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--lr-actor", type=float, default=1e-5)
    p.add_argument("--lr-critic", type=float, default=1e-5)
    p.add_argument("--kl-coef", type=float, default=0.0,
                   help="KL penalty coefficient (0=disabled, 0.01 for real)")
    p.add_argument("--max-response-len", type=int, default=300)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--num-steps", type=int, default=2)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
