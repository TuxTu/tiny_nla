#!/usr/bin/env python3
"""Push trained NLA model checkpoints to HuggingFace Hub.

Usage:
  # Login first (one-time):
  huggingface-cli login

  # Push the SFT actor (base model anyone can use):
  python scripts/push_to_hub.py \
      --ckpt checkpoints/actor_sft \
      --repo TuHan/qwen3-4b-nla-actor-sft

  # Push RL step 100 checkpoint:
  python scripts/push_to_hub.py \
      --ckpt checkpoints/rl_checkpoints/step_100/actor \
      --repo TuHan/qwen3-4b-nla-actor-rl-step100
"""

import argparse
import shutil
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, create_repo


def push_checkpoint(ckpt_path: str, repo_id: str, private: bool = False):
    """Load a saved HF checkpoint and push it to the Hub."""
    ckpt = Path(ckpt_path)
    assert ckpt.is_dir(), f"Checkpoint not found: {ckpt_path}"
    assert (ckpt / "model.safetensors").exists(), f"No model.safetensors in {ckpt_path}"

    print(f"Loading model from {ckpt_path} ...")
    model = AutoModelForCausalLM.from_pretrained(str(ckpt))
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))

    create_repo(repo_id, private=private, exist_ok=True)

    print(f"Pushing to {repo_id} ...")
    model.push_to_hub(repo_id, private=private)
    tokenizer.push_to_hub(repo_id, private=private)

    print(f"Done: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to HF-format checkpoint directory")
    p.add_argument("--repo", required=True, help="HF repo ID, e.g. TuHan/qwen3-4b-nla-actor-sft")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()
    push_checkpoint(args.ckpt, args.repo, args.private)
