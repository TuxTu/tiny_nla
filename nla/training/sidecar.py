"""Sidecar read/write for NLA training metadata.

The sidecar ({parquet}.nla_meta.yaml or {checkpoint}/nla_meta.yaml) is the
contract between data generation and training — it records token IDs, prompt
templates, extraction params, and activation dimensions.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from nla.training.schema import sidecar_path_for


def write_sidecar(path: str | Path, meta: dict) -> None:
    """Write sidecar YAML for a parquet file or checkpoint directory."""
    sp = Path(sidecar_path_for(path))
    sp.parent.mkdir(parents=True, exist_ok=True)
    with open(sp, "w") as f:
        yaml.safe_dump(meta, f, allow_unicode=True, sort_keys=False)


def read_sidecar(path: str | Path) -> dict:
    """Read sidecar YAML. Returns empty dict if missing."""
    sp = Path(sidecar_path_for(path))
    if not sp.exists():
        return {}
    with open(sp) as f:
        return yaml.safe_load(f) or {}


def write_dataset_sidecar(
    parquet_path: str | Path,
    *,
    d_model: int,
    token_meta,
    split_type: str,
    actor_template: str,
    critic_template: str | None = None,
    injection_scale: float | str | None = None,
    mse_scale: float | str | None = "sqrt_d_model",
    num_rows: int = 0,
    corpus_name: str | None = None,
) -> None:
    """Write a dataset sidecar with all training metadata."""
    meta: dict = {
        "kind": "nla_dataset",
        "schema_version": 2,
        "dataset_id": str(Path(parquet_path).name),
        "split_type": split_type,
        "row_count": num_rows,
        "extraction": {
            "d_model": d_model,
            "injection_scale": injection_scale,
            "mse_scale": mse_scale,
            "norm": "none",
        },
        "tokens": {
            "injection_char": token_meta.injection_char,
            "injection_token_id": token_meta.injection_token_id,
            "injection_left_neighbor_id": token_meta.injection_left_neighbor_id,
            "injection_right_neighbor_id": token_meta.injection_right_neighbor_id,
        },
        "prompt_templates": {
            "actor": actor_template,
        },
    }
    if critic_template is not None:
        meta["prompt_templates"]["critic"] = critic_template
    if token_meta.critic_suffix_ids is not None:
        meta["tokens"]["critic_suffix_ids"] = token_meta.critic_suffix_ids
    if corpus_name is not None:
        meta["extraction"]["corpus"] = corpus_name

    write_sidecar(parquet_path, meta)
