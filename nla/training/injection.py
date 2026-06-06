"""Pure injection-hook logic — ported from original nla/injection.py.

The most correctness-critical path in NLA: if injection fails the model sees
the literal marker character and generates Chinese. This is the one place that
must be right.
"""

import torch


def inject_at_marked_positions(
    input_ids: torch.Tensor,
    embeddings: torch.Tensor,
    vectors: torch.Tensor,
    inj_id: int,
    left_id: int,
    right_id: int,
) -> torch.Tensor:
    """Overwrite embedding rows at injection-marker positions with activation vectors.

    input_ids:  [B, S] — full token stream.
    embeddings: [B, S, d] — embedding layer output. Cloned; original unchanged.
    vectors:    [N, d] — activation vectors. N must equal count of valid matches.
    inj_id, left_id, right_id: injection token + canonical neighbors.

    A match is valid iff input_ids[b,p] == inj_id AND input_ids[b,p-1]==left_id
    AND input_ids[b,p+1]==right_id. Neighbor check prevents false positives
    from the marker char appearing in response text.
    """
    assert input_ids.shape == embeddings.shape[:-1], (
        f"input_ids {tuple(input_ids.shape)} and embeddings "
        f"{tuple(embeddings.shape[:-1])} batch dims must match"
    )
    assert vectors.ndim == 2 and vectors.shape[1] == embeddings.shape[-1], (
        f"vectors must be [N, d_model], got {tuple(vectors.shape)}, "
        f"d_model={embeddings.shape[-1]}"
    )

    out = embeddings.clone()
    vectors = vectors.to(out.device, out.dtype)

    seq_len = input_ids.shape[-1]
    matches = (input_ids == inj_id).nonzero(as_tuple=False)  # [M, 2]
    vec_idx = 0

    for b, p in matches.tolist():
        if p == 0 or p == seq_len - 1:
            continue
        if input_ids[b, p - 1] != left_id or input_ids[b, p + 1] != right_id:
            continue
        out[b, p] = vectors[vec_idx]
        vec_idx += 1

    expected = vectors.shape[0]
    if vec_idx != expected:
        raise RuntimeError(
            f"found {vec_idx} injection sites with correct neighbors, "
            f"expected {expected}. Check prompt template drift, tokenizer "
            f"version, or data corruption."
        )

    return out
