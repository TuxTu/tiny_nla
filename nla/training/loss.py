"""NLA loss functions — simplified from original nla/loss.py (no Miles protocol)."""

import torch
import torch.nn.functional as F

from nla.training.schema import normalize_activation


def nla_critic_loss(
    pred: torch.Tensor,
    gold: torch.Tensor,
    mse_scale: float | None,
) -> torch.Tensor:
    """MSE between normalized critic prediction and gold activation vector.

    pred, gold: [B, d_model] — vectors at the last-token position.
    mse_scale: L2 norm target for normalization (None = raw MSE).

    Returns scalar loss (mean over batch).
    """
    p = normalize_activation(pred, mse_scale)
    g = normalize_activation(gold, mse_scale)
    return F.mse_loss(p, g)


def sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy loss with optional per-position mask.

    logits:   [B, T, V]
    labels:   [B, T] — -100 for ignored positions
    loss_mask: [B, T] — 1.0 for response tokens, 0.0 for prompt (optional)
    """
    # shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    ce = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_labels.shape)

    if loss_mask is not None:
        shift_mask = loss_mask[..., 1:].to(ce.device)
        ce = ce * shift_mask
        return ce.sum() / shift_mask.sum().clamp_min(1)
    return ce.mean()
