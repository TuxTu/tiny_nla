"""Shared NLA constants and helpers for training.

Ports the training-relevant subset of original nla/schema.py.
"""

import math
import re
from dataclasses import dataclass
from pathlib import Path

import torch

# ---- sidecar conventions ----------------------------------------------------
SIDECAR_SUFFIX = ".nla_meta.yaml"
SIDECAR_BASENAME = "nla_meta.yaml"

# ---- explanation tags -------------------------------------------------------
EXPLANATION_OPEN = "<explanation>"
EXPLANATION_CLOSE = "</explanation>"
_EXPLANATION_RE = re.compile(
    f"{re.escape(EXPLANATION_OPEN)}(.*?){re.escape(EXPLANATION_CLOSE)}",
    re.DOTALL,
)


def wrap_explanation(text: str) -> str:
    return f"{EXPLANATION_OPEN}\n{text}\n{EXPLANATION_CLOSE}"


def extract_explanation(response: str) -> str | None:
    m = _EXPLANATION_RE.search(response)
    return m.group(1).strip() if m else None


# ---- parquet / training keys ------------------------------------------------
ACTIVATION_COLUMN = "activation_vector"
INJECT_PLACEHOLDER = "<INJECT>"
MM_ACTIVATION_KEY = "nla_activation"
MM_MSE_SCALE_KEY = "nla_mse_scale"

# ---- scale sentinel ---------------------------------------------------------
SCALE_SQRT_D = "sqrt_d_model"


def resolve_target_scale(raw: float | str | None, d_model: int) -> float | None:
    if raw is None or raw in ("raw", "none"):
        return None
    if raw == SCALE_SQRT_D:
        return math.sqrt(d_model)
    if isinstance(raw, str):
        return float(raw)
    return float(raw)


# ---- token metadata ---------------------------------------------------------


@dataclass
class NLATokenMeta:
    injection_char: str
    injection_token_id: int
    injection_left_neighbor_id: int
    injection_right_neighbor_id: int
    critic_suffix_ids: list[int] | None = None


# ---- normalization -----------------------------------------------------------


def normalize_activation(v: torch.Tensor, target_scale: float | None) -> torch.Tensor:
    """Scale vectors to target_scale L2-norm. No-op if target_scale is None."""
    if target_scale is None:
        return v
    norm_fp32 = v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v / (norm_fp32 / target_scale).to(v.dtype)


# ---- canonical neighbors ----------------------------------------------------


def compute_canonical_neighbors(
    tokenizer,
    actor_template: str,
    injection_char: str,
    injection_token_id: int,
) -> tuple[int, int]:
    """Tokenize canonical actor prompt, return (left_id, right_id) at injection site."""
    content = actor_template.format(injection_char=injection_char)
    result = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
    )
    # transformers >=4.46 returns BatchEncoding (Mapping, not dict); older returns list
    ids = result["input_ids"] if hasattr(result, "keys") else result
    matches = [i for i, tid in enumerate(ids) if tid == injection_token_id]
    assert len(matches) == 1, (
        f"injection token id {injection_token_id} ({injection_char!r}) appears "
        f"{len(matches)}x in canonical actor prompt (expected 1). Template: {content!r}"
    )
    p = matches[0]
    assert 0 < p < len(ids) - 1, (
        f"injection token at position {p} is at edge of sequence (len={len(ids)})"
    )
    return ids[p - 1], ids[p + 1]


# ---- sidecar path resolution ------------------------------------------------


def sidecar_path_for(path: str | Path) -> Path:
    p = Path(str(path).split("@[")[0])
    if p.is_dir() or (not p.exists() and p.suffix == ""):
        return p / SIDECAR_BASENAME
    return p.with_name(p.name + SIDECAR_SUFFIX)
