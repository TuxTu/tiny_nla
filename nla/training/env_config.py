"""Hardware auto-detection for training. Supports single-GPU and DDP (torchrun)."""

import os
from dataclasses import dataclass, field

import torch


@dataclass
class EnvConfig:
    device: torch.device
    dtype: torch.dtype
    is_mps: bool
    local_rank: int = 0
    world_size: int = 1
    global_rank: int = 0
    ddp_enabled: bool = False
    gpu_name: str = ""

    @property
    def amp_enabled(self) -> bool:
        return self.dtype == torch.bfloat16

    @property
    def is_main_process(self) -> bool:
        """True only for the coordinator process (global rank 0)."""
        return self.global_rank == 0

    @property
    def is_local_main(self) -> bool:
        """True only for the first process on each node (local rank 0)."""
        return self.local_rank == 0


def detect() -> EnvConfig:
    local_rank_str = os.environ.get("LOCAL_RANK", "0")
    world_size_str = os.environ.get("WORLD_SIZE", "1")
    rank_str = os.environ.get("RANK", "0")

    local_rank = int(local_rank_str)
    world_size = int(world_size_str)
    global_rank = int(rank_str)
    ddp_enabled = world_size > 1

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        # In DDP mode LOCAL_RANK picks the GPU; otherwise default to cuda:0
        device_id = local_rank if ddp_enabled else 0
        if device_id >= gpu_count:
            print(f"[env_config] WARNING: local_rank={device_id} >= "
                  f"device_count={gpu_count} — falling back to cuda:0")
            device_id = 0
        device = torch.device(f"cuda:{device_id}")
        dtype = torch.bfloat16
        is_mps = False
        gpu_name = torch.cuda.get_device_name(device_id) if gpu_count > 0 else "unknown"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float32  # MPS bf16 is buggy on many ops
        is_mps = True
        gpu_name = "Apple MPS"
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        is_mps = False
        gpu_name = "CPU"

    return EnvConfig(
        device=device, dtype=dtype, is_mps=is_mps,
        local_rank=local_rank, world_size=world_size, global_rank=global_rank,
        ddp_enabled=ddp_enabled, gpu_name=gpu_name,
    )
