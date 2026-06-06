"""Injection token selection (auto-pick + cache) and critic-suffix computation.

Ported from original nla/datagen/injection_tokens.py.
"""

from pathlib import Path
from typing import Any

import yaml

from nla.training.schema import NLATokenMeta, compute_canonical_neighbors

_CacheEntry = dict[str, Any]
_CACHE_PATH = Path(__file__).parent / "injection_token_cache.yaml"
_INJECTION_RANGE = (0x3200, 0x33FF)  # CJK Enclosed Letters and Months


def _load_cache() -> dict[str, _CacheEntry]:
    if not _CACHE_PATH.exists():
        return {}
    loaded = yaml.safe_load(_CACHE_PATH.read_text())
    return loaded if isinstance(loaded, dict) else {}


def _save_cache(cache: dict[str, _CacheEntry]) -> None:
    _CACHE_PATH.write_text(yaml.safe_dump(cache, allow_unicode=True, sort_keys=True))


def _tokenize_one(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def find_injection_token(tokenizer: Any) -> tuple[str, int]:
    """Auto-pick a single-token CJK char for activation injection. Cached."""
    key = tokenizer.name_or_path
    cache = _load_cache()

    if key in cache:
        cached_char = cache[key]["char"]
        cached_id = cache[key]["token_id"]
        ids = _tokenize_one(tokenizer, cached_char)
        assert len(ids) == 1 and ids[0] == cached_id, (
            f"cached injection token for {key!r} no longer valid: "
            f"{cached_char!r} now tokenizes to {ids} (cached id={cached_id}). "
            f"Delete the cache entry and rerun."
        )
        return cached_char, cached_id

    lo, hi = _INJECTION_RANGE
    for codepoint in range(lo, hi + 1):
        char = chr(codepoint)
        ids = _tokenize_one(tokenizer, char)
        if len(ids) == 1:
            cache[key] = {"char": char, "token_id": ids[0]}
            _save_cache(cache)
            return char, ids[0]

    raise AssertionError(
        f"no single-token CJK char found in U+{lo:04X}–U+{hi:04X} for tokenizer "
        f"{key!r}. Hand-pick a character and add it to injection_token_cache.yaml."
    )


def compute_critic_suffix_ids(tokenizer: Any, critic_template: str) -> list[int]:
    """Return the stable tail of the critic template's suffix token IDs.

    Drops the first suffix token (BPE boundary with explanation's last char).
    """
    assert "{explanation}" in critic_template, (
        f"critic_template must contain '{{explanation}}' placeholder: {critic_template!r}"
    )
    suffix_str = critic_template.split("{explanation}")[-1]
    suffix_ids = _tokenize_one(tokenizer, suffix_str)
    assert len(suffix_ids) >= 2, (
        f"critic template suffix {suffix_str!r} tokenized to {len(suffix_ids)} tokens — "
        f"need at least 2 so we can drop the BPE-boundary token."
    )
    return suffix_ids[1:]


def build_token_meta(
    tokenizer: Any,
    actor_template: str,
    critic_template: str | None = None,
) -> NLATokenMeta:
    """One-shot: auto-pick injection char + neighbors, optionally compute critic suffix."""
    inj_char, inj_id = find_injection_token(tokenizer)
    left_id, right_id = compute_canonical_neighbors(tokenizer, actor_template, inj_char, inj_id)
    suffix_ids = compute_critic_suffix_ids(tokenizer, critic_template) if critic_template else None
    return NLATokenMeta(
        injection_char=inj_char,
        injection_token_id=inj_id,
        injection_left_neighbor_id=left_id,
        injection_right_neighbor_id=right_id,
        critic_suffix_ids=suffix_ids,
    )
