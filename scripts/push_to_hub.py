#!/usr/bin/env python3
"""Push NLA checkpoints into one Hub repo, under a folder layout.

    KTH-ASSERT/tiny-nla
      nla/actor/          text actor  (AV: vector -> explanation)
      nla/critic/         critic      (AR: explanation -> vector)
      nla/actor-rl/       GRPO run3@150, best FVE in the project
      code-decoder/       code reconstruction  + centre_mean.npy

Load a subfolder with:
    AutoModelForCausalLM.from_pretrained("KTH-ASSERT/tiny-nla", subfolder="nla/actor")

Uploads files directly rather than round-tripping through from_pretrained, which
would load 16 GB into RAM only to re-serialise it.

  hf auth login                       # once

  python scripts/push_to_hub.py --ckpt checkpoints/actor_sft_8b_s50fix \
      --repo KTH-ASSERT/tiny-nla --path-in-repo nla/actor --private \
      --sidecar data/split50/av_sft_train.parquet.nla_meta.yaml

  python scripts/push_to_hub.py --ckpt checkpoints/code_exp1s \
      --repo KTH-ASSERT/tiny-nla --path-in-repo code-decoder --private \
      --sidecar data/coderec/exp1s_eval.parquet.nla_meta.yaml \
      --centre-mean data/coderec/exp1s_mean.npy
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def push(ckpt, repo_id, path_in_repo, private=False, sidecar=None, centre_mean=None,
         dry_run=False):
    ckpt = Path(ckpt)
    assert ckpt.is_dir(), f"checkpoint not found: {ckpt}"
    assert (ckpt / "model.safetensors").exists() or \
           list(ckpt.glob("model-*.safetensors")), f"no safetensors in {ckpt}"

    files = sorted(f for f in ckpt.iterdir() if f.is_file())
    total = sum(f.stat().st_size for f in files)
    print(f"{ckpt}  ->  {repo_id}/{path_in_repo}")
    print(f"  {len(files)} files, {total/1e9:.1f} GB")

    extras = []
    if sidecar:
        extras.append((Path(sidecar), f"{path_in_repo}/nla_meta.yaml"))
    if centre_mean:
        extras.append((Path(centre_mean), f"{path_in_repo}/centre_mean.npy"))
    for src, dest in extras:
        print(f"  + {dest}" + ("" if src.exists() else "   *** MISSING ***"))
    if not centre_mean and "code" in path_in_repo:
        print("  WARNING: no --centre-mean for a code checkpoint. If this actor was\n"
              "  trained on centred vectors, inference without the mean silently\n"
              "  produces plausible but unrelated output.")
    if dry_run:
        print("  (dry run, nothing uploaded)")
        return

    api = HfApi()
    create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=str(ckpt), path_in_repo=path_in_repo,
                      repo_id=repo_id, repo_type="model",
                      commit_message=f"add {path_in_repo}")
    for src, dest in extras:
        if not src.exists():
            print(f"  SKIPPED (missing): {dest}")
            continue
        api.upload_file(path_or_fileobj=str(src), path_in_repo=dest,
                        repo_id=repo_id, repo_type="model")
    print(f"  done: https://huggingface.co/{repo_id}/tree/main/{path_in_repo}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--repo", required=True, help="e.g. KTH-ASSERT/tiny-nla")
    p.add_argument("--path-in-repo", required=True, help="e.g. nla/actor")
    p.add_argument("--private", action="store_true")
    p.add_argument("--sidecar", help="nla_meta.yaml — injection token id and neighbours")
    p.add_argument("--centre-mean", help="npy training mean; REQUIRED for a centred actor")
    p.add_argument("--dry-run", action="store_true", help="show what would upload")
    a = p.parse_args()
    push(a.ckpt, a.repo, a.path_in_repo, a.private, a.sidecar, a.centre_mean, a.dry_run)
