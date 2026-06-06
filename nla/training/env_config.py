"""Hardware auto-detection for training. Single-GPU only for v1."""

import os
from dataclasses import dataclass

import torch


@dataclass
class EnvConfig:
    device: torch.device
    dtype: torch.dtype
    is_mps: bool

    @property
    def amp_enabled(self) -> bool:
        return self.dtype == torch.bfloat16


def detect() -> EnvConfig:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
        is_mps = False
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float32  # MPS bf16 is buggy on many ops
        is_mps = True
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        is_mps = False

    return EnvConfig(device=device, dtype=dtype, is_mps=is_mps)
