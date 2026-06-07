"""Evaluate base model loss on NLA datasets (no training, forward only).

Computes the loss that the untrained Qwen3-4B model achieves on
AV-SFT and AR-SFT tasks, so we can compare against SFT checkpoints.

Usage:
  python -m nla.training.eval_base_loss \
    --data data/eval_ar_sft_train.parquet \
    --model-name Qwen/Qwen3-4B --task critic --max-length 512

  python -m nla.training.eval_base_loss \
    --data data/eval_av_sft_train.parquet \
    --model-name Qwen/Qwen3-4B --task actor --max-length 2048 \
    --injection-char ㈎
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from nla.training.env_config import detect, EnvConfig
from nla.training.injection import inject_at_marked_positions
from nla.training.loss import nla_critic_loss, sft_loss
from nla.training.models import NLACriticModel
from nla.training.resolve import resolve_parquet
from nla.training.schema import (
    ACTIVATION_COLUMN,
    INJECT_PLACEHOLDER,
    resolve_target_scale,
)
from nla.training.sidecar import read_sidecar

import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# datasets (lightweight copies from training scripts)
# ---------------------------------------------------------------------------

class CriticEvalDataset(Dataset):
    def __init__(self, parquet_path: str):
        table = pq.read_table(resolve_parquet(parquet_path))
        self.prompts = table.column("prompt").to_pylist()
        col = table.column(ACTIVATION_COLUMN)
        self.vectors = np.array([v.as_py() for v in col], dtype=np.float32)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx], torch.from_numpy(self.vectors[idx])


class ActorEvalDataset(Dataset):
    def __init__(self, parquet_path: str, injection_char: str):
        table = pq.read_table(resolve_parquet(parquet_path))
        raw_prompts = table.column("prompt").to_pylist()
        self.responses = table.column("response").to_pylist()
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
        return self.prompts[idx], self.responses[idx], torch.from_numpy(self.vectors[idx])


# ---------------------------------------------------------------------------
# evaluate critic
# ---------------------------------------------------------------------------

def eval_critic(args, env: EnvConfig) -> float:
    ds = CriticEvalDataset(args.data)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    d_model = ds.vectors.shape[1]

    # Sidecar → mse_scale
    sidecar = read_sidecar(args.data)
    mse_scale_raw = sidecar.get("extraction", {}).get("mse_scale", "sqrt_d_model")
    mse_scale = resolve_target_scale(mse_scale_raw, d_model)

    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    # Model
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    layer_index = (2 * cfg.num_hidden_layers) // 3
    model = NLACriticModel.from_pretrained(
        args.model_name, nla_num_layers=layer_index,
        torch_dtype=env.dtype, device_map={"": env.device},
    )
    model.eval()
    model.gradient_checkpointing_enable()

    losses = []
    with torch.no_grad():
        for prompts, gold_vectors in tqdm(dl, desc="critic eval"):
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
                seq_lens = attention_mask.sum(dim=1) - 1
                pred = output.values[torch.arange(len(seq_lens)), seq_lens]
                loss = nla_critic_loss(pred, gold, mse_scale)
            losses.append(loss.item())

    avg = sum(losses) / len(losses)
    print(f"  samples: {len(ds)}  batches: {len(losses)}  d_model={d_model}  mse_scale={mse_scale:.2f}")
    print(f"  base critic loss: {avg:.4f}")
    return avg


# ---------------------------------------------------------------------------
# evaluate actor
# ---------------------------------------------------------------------------

def eval_actor(args, env: EnvConfig) -> float:
    # Sidecar → token info + injection scale
    sidecar = read_sidecar(args.data)
    tokens = sidecar.get("tokens", {})
    injection_char = args.injection_char or tokens.get("injection_char", "㈎")
    inj_id = tokens.get("injection_token_id")
    left_id = tokens.get("injection_left_neighbor_id")
    right_id = tokens.get("injection_right_neighbor_id")

    injection_scale = resolve_target_scale(
        sidecar.get("extraction", {}).get("injection_scale"),
        sidecar["extraction"]["d_model"],
    )

    # Tokenizer — get injection IDs if not in sidecar
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    if inj_id is None:
        tok = tokenizer(injection_char, add_special_tokens=False)
        if len(tok["input_ids"]) != 1:
            raise ValueError(f"injection_char {injection_char!r} tokenized to {len(tok['input_ids'])} tokens")
        inj_id = tok["input_ids"][0]
    if left_id is None or right_id is None:
        # Use default neighbor IDs (safe fallback)
        inj_enc = tokenizer(f"A{injection_char}A", add_special_tokens=False)
        pos = (inj_enc["input_ids"] == inj_id).nonzero(as_tuple=True)[0]
        if len(pos) == 1:
            p = pos[0].item()
            left_id = left_id or inj_enc["input_ids"][p - 1]
            right_id = right_id or inj_enc["input_ids"][p + 1]
        else:
            left_id = left_id or 0
            right_id = right_id or 0

    print(f"injection: char={injection_char!r}  id={inj_id}  neighbors=({left_id}, {right_id})")
    print(f"injection_scale: {injection_scale}")

    # Dataset
    ds = ActorEvalDataset(args.data, injection_char)
    def _collate(batch):
        prompts, responses, vectors = zip(*batch)
        return list(prompts), list(responses), torch.stack(vectors)

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=_collate)

    # Model
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=env.dtype,
        device_map={"": env.device}, trust_remote_code=True,
    )
    model.eval()
    model.gradient_checkpointing_enable()

    losses = []
    with torch.no_grad():
        for prompts, responses, vectors in tqdm(dl, desc="actor eval"):
            # Tokenize prompt + response together
            full_texts = []
            resp_starts = []
            for msgs, resp in zip(prompts, responses):
                # Apply chat template to prompt messages (returns list with tokenize=True)
                prompt_ids = tokenizer.apply_chat_template(
                    msgs, tokenize=True, add_generation_prompt=True,
                )
                # Handle BatchEncoding/dict wrappers
                if hasattr(prompt_ids, "input_ids"):
                    prompt_ids = prompt_ids["input_ids"]
                if not isinstance(prompt_ids, list):
                    prompt_ids = list(prompt_ids)
                start_pos = len(prompt_ids)
                resp_ids = tokenizer.encode(resp, add_special_tokens=False)
                full_ids = prompt_ids + resp_ids
                full_texts.append(full_ids)
                resp_starts.append(start_pos)

            # Pad
            max_len = min(max(len(ids) for ids in full_texts), args.max_length)
            input_ids_list = [ids[:max_len] for ids in full_texts]
            input_ids = torch.full((len(input_ids_list), max_len),
                                   tokenizer.pad_token_id, dtype=torch.long)
            for i, ids in enumerate(input_ids_list):
                input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

            input_ids = input_ids.to(env.device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long()

            # Labels: -100 on prompt, keep response
            labels = input_ids.clone()
            for i, s in enumerate(resp_starts):
                labels[i, :min(s, max_len)] = -100

            # Inject activation at marker positions
            injections = vectors.to(env.device)
            embedding_layer = model.get_input_embeddings()
            inputs_embeds = embedding_layer(input_ids)
            inputs_embeds = inject_at_marked_positions(
                input_ids, inputs_embeds, injections, inj_id, left_id, right_id,
            )
            inputs_embeds = inputs_embeds.to(dtype=env.dtype)

            with torch.autocast(device_type=env.device.type, dtype=env.dtype,
                                enabled=env.amp_enabled):
                output = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
                loss = sft_loss(output.logits, labels)

            losses.append(loss.item())

    avg = sum(losses) / len(losses)
    print(f"  samples: {len(ds)}  batches: {len(losses)}  injection_scale={injection_scale}")
    print(f"  base actor loss: {avg:.4f}")
    return avg


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="eval parquet")
    p.add_argument("--model-name", required=True, help="HF base model")
    p.add_argument("--task", required=True, choices=["critic", "actor"])
    p.add_argument("--batch-size", type=int, default=8,
                   help="eval micro-batch size (default: 8)")
    p.add_argument("--max-length", type=int, default=512,
                   help="max token length (default: 512 for critic, 2048 for actor)")
    p.add_argument("--injection-char", type=str, default="㈎",
                   help="injection marker character (actor only)")
    args = p.parse_args()

    env = detect()
    print(f"device: {env.device}  dtype: {env.dtype}")
    print(f"task: {args.task}  model: {args.model_name}")
    print(f"data: {args.data}  batch_size: {args.batch_size}  max_length: {args.max_length}")

    if args.task == "critic":
        eval_critic(args, env)
    else:
        eval_actor(args, env)

    print("=== eval done ===")


if __name__ == "__main__":
    main()
