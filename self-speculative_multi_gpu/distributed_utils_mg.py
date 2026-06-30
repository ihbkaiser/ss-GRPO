"""Small utilities for torchrun/NCCL synchronous data parallel training.

This module intentionally does not wrap models with DDP. It supports the
manual data-parallel pattern used by custom generation-heavy GRPO code:
local forward/generation on each GPU, then explicit gradient all-reduce before
optimizer.step().
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def setup_distributed(seed: Optional[int] = None) -> DistributedContext:
    """Initialize torch.distributed when launched by torchrun.

    Works in both modes:
    - Single GPU/CPU: no env vars required.
    - torchrun: reads RANK, LOCAL_RANK and WORLD_SIZE, initializes NCCL.
    """
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if torch.cuda.is_available():
        if distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda", 0)
            torch.cuda.set_device(device)
    else:
        if distributed:
            raise RuntimeError("Distributed NCCL training requires CUDA GPUs.")
        device = torch.device("cpu")

    if distributed and not is_distributed():
        dist.init_process_group(backend="nccl", init_method="env://")

    if seed is not None:
        seed_everything(seed + rank)

    return DistributedContext(
        distributed=distributed,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rank0_print(*args, **kwargs) -> None:
    if not is_distributed() or dist.get_rank() == 0:
        print(*args, **kwargs)


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def _as_device_tensor(value: float | int | torch.Tensor, device: torch.device, dtype=torch.float64) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, dtype=dtype).reshape(1)
    return torch.tensor([value], device=device, dtype=dtype)


def reduce_sum(value: float | int | torch.Tensor, device: torch.device) -> float:
    tensor = _as_device_tensor(value, device)
    if is_distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def reduce_max(value: float | int | torch.Tensor, device: torch.device) -> float:
    tensor = _as_device_tensor(value, device)
    if is_distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def broadcast_object(obj, src: int = 0):
    """Broadcast a Python object from src rank to all ranks."""
    if not is_distributed():
        return obj
    payload = [obj if dist.get_rank() == src else None]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def make_zero_loss(parameters: Iterable[torch.nn.Parameter], device: torch.device) -> torch.Tensor:
    """Create a differentiable zero scalar connected to all trainable params.

    This lets ranks with no valid local samples still participate in backward
    and subsequent gradient all-reduce without deadlock.
    """
    zero = torch.zeros((), device=device)
    has_trainable = False
    for p in parameters:
        if p.requires_grad:
            has_trainable = True
            zero = zero + p.float().sum() * 0.0
    if not has_trainable:
        zero = zero.requires_grad_(True)
    return zero


def average_gradients(
    parameters: Iterable[torch.nn.Parameter],
    world_size: Optional[int] = None,
    *,
    divide: bool = True,
    ensure_grads: bool = True,
) -> None:
    """All-reduce gradients over all ranks.

    Call this after backward and before optimizer.step().  Only trainable
    parameters are synchronized.  When ensure_grads=True, missing gradients are
    materialized as zeros so every rank performs the same all_reduce calls.
    """
    if not is_distributed():
        return

    if world_size is None:
        world_size = dist.get_world_size()

    params = [p for p in parameters if p.requires_grad]
    for p in params:
        if p.grad is None:
            if ensure_grads:
                p.grad = torch.zeros_like(p, memory_format=torch.preserve_format)
            else:
                continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        if divide:
            p.grad.div_(world_size)


def max_memory_allocated_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    local_peak = torch.cuda.max_memory_allocated(device) / 1024**3
    return reduce_max(local_peak, device)
