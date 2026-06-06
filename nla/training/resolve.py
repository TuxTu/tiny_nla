"""Resolve a data path that may be a local file or a HuggingFace Hub dataset."""

from pathlib import Path


def resolve_parquet(path: str) -> str:
    """Resolve a path-or-HF-repo reference to a local parquet file.

    Supports:
      - Local path:  ``data/foo.parquet``
      - HF repo + file:  ``TuxTu/qwen3-nla-250k/av_sft_explained.parquet``
      - HF repo (no file):  ``TuxTu/qwen3-nla-250k`` — downloads all parquets
        and returns the directory (caller must then pick the right file).
    """
    p = Path(path)
    if p.exists():
        return str(p.resolve())

    # Looks like a HF repo reference:  user/repo[/file]
    parts = path.split("/")
    if len(parts) >= 2 and not path.startswith((".", "/", "~")):
        from huggingface_hub import hf_hub_download

        repo_id = "/".join(parts[:2])
        if len(parts) > 2:
            filename = "/".join(parts[2:])
        else:
            # No filename — download the whole repo
            from huggingface_hub import snapshot_download
            return snapshot_download(repo_id, repo_type="dataset")

        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
        )

    raise FileNotFoundError(f"Cannot resolve path: {path!r}  (not a local file, "
                            f"not a HF Hub reference)")
