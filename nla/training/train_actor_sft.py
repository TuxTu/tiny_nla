"""Actor SFT training — full model learns to generate explanations from injected activations.

Teacher-forcing CE loss on response tokens only. The injection hook replaces the
embedding at the marker token position with the activation vector during forward.

Usage:
  python -m nla.training.train_actor_sft \
    --data data/test/av_sft_train.parquet \
    --model-name Qwen/Qwen3-0.6B \
    --output-dir data/test/actor_checkpoint \
    --micro-batch-size 2 --num-steps 10
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from nla.training.env_config import detect
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
        raw_prompts = pq.read_table(parquet_path).column("prompt").to_pylist()
        self.responses = pq.read_table(parquet_path).column("response").to_pylist()

        col = pq.read_table(parquet_path).column(ACTIVATION_COLUMN)
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
# training loop
# ---------------------------------------------------------------------------


def train(args) -> None:
    env = detect()
    print(f"device: {env.device}  dtype: {env.dtype}")

    # ---- sidecar + tokenizer ------------------------------------------------
    sidecar = read_sidecar(args.data)
    tokens = sidecar.get("tokens", {})
    injection_char = tokens["injection_char"]
    inj_id = tokens["injection_token_id"]
    left_id = tokens["injection_left_neighbor_id"]
    right_id = tokens["injection_right_neighbor_id"]
    print(f"injection: char={injection_char!r}  id={inj_id}  "
          f"neighbors=({left_id}, {right_id})")

    injection_scale = resolve_target_scale(
        sidecar.get("extraction", {}).get("injection_scale"),
        sidecar["extraction"]["d_model"],
    )
    print(f"injection_scale: {injection_scale}")

    # ---- tokenizer -----------------------------------------------------------
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    # ---- data ----------------------------------------------------------------
    ds = ActorDataset(args.data, injection_char)
    print(f"dataset: {len(ds)} rows  d_model={ds.vectors.shape[1]}")

    def _collate(batch):
        """Custom collate — preserves list-of-dicts for prompts (no tensor stacking)."""
        prompts, responses, vectors = zip(*batch)
        return list(prompts), list(responses), torch.stack(vectors)

    dl = DataLoader(ds, batch_size=args.micro_batch_size, shuffle=True,
                    collate_fn=_collate)

    # ---- model ---------------------------------------------------------------
    print(f"loading {args.model_name} ...")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=env.dtype,
        device_map={"": env.device} if not env.is_mps else None,
    )
    if env.is_mps:
        model = model.to(env.device)
    model.train()
    # Enable gradient checkpointing for memory
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    print(f"actor: {model.config.num_hidden_layers} layers  "
          f"d_model={model.config.hidden_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ---- training ------------------------------------------------------------
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    global_step = 0
    losses = []

    while global_step < args.num_steps:
        pbar = tqdm(dl, desc=f"actor  step={global_step}/{args.num_steps}")
        for messages_batch, responses_batch, vectors_batch in pbar:
            if global_step >= args.num_steps:
                break

            # Build [prompt | response] conversations and tokenize
            texts = []
            for msgs, resp in zip(messages_batch, responses_batch):
                # Apply chat template to prompt, then append assistant response
                prompt_str = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                )
                texts.append(prompt_str + resp)

            # Tokenize with loss masking
            # We need per-position masks: 0 for prompt tokens, 1 for response tokens
            # Strategy: tokenize prompt-only and full-text, compute mask from lengths
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
            embed = model.get_input_embeddings()

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
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = sft_loss(outputs.logits, labels)
            finally:
                hook.remove()

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
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data", required=True, help="AV-SFT training parquet")
    p.add_argument("--model-name", required=True, help="HF base model")
    p.add_argument("--output-dir", required=True, help="checkpoint directory")
    p.add_argument("--micro-batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--max-length", type=int, default=2048)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
