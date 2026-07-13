from __future__ import annotations

from dataclasses import dataclass
import gc
import os
from typing import Any


@dataclass(frozen=True)
class DistContext:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: Any = "cpu"
    group: Any | None = None
    nvshmem_initialized: bool = False

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0


def context_from_env(device: Any | None = None, group: Any | None = None, nvshmem_initialized: bool = False) -> DistContext:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return DistContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=_resolve_device(device, local_rank),
        group=group,
        nvshmem_initialized=nvshmem_initialized,
    )


def requires_nvshmem(cfg: Any) -> bool:
    from .schemes import REGISTRY

    return any(REGISTRY[scheme.code][0].requires_nvshmem for scheme in cfg.schemes)


def init_context(cfg: Any | None = None, device: Any | None = None) -> DistContext:
    if cfg is not None and requires_nvshmem(cfg):
        return _init_nvshmem_context(device)
    return _init_nccl_context(device)


def finalize_context(ctx: DistContext | None = None) -> None:
    if ctx is not None and ctx.nvshmem_initialized:
        import torch
        from triton_dist.utils import finalize_distributed

        torch.cuda.synchronize()
        gc.collect()
        finalize_distributed()
        return

    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


def _init_nccl_context(device: Any | None) -> DistContext:
    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    resolved_device = _resolve_device(device, local_rank)
    torch.cuda.set_device(resolved_device)

    if dist.is_initialized():
        dist.barrier()

    return DistContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=resolved_device,
        group=None,
        nvshmem_initialized=False,
    )


def _resolve_device(device: Any | None, local_rank: int) -> Any:
    if device is not None:
        return device
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def _init_nvshmem_context(device: Any | None) -> DistContext:
    from .tdx.compat import apply

    apply()
    try:
        from triton_dist.utils import initialize_distributed
    except ModuleNotFoundError:
        return _init_nccl_context(device)
    group = initialize_distributed()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    import torch
    resolved_device = _resolve_device(device, local_rank)
    torch.cuda.set_device(resolved_device)
    return DistContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=resolved_device,
        group=group,
        nvshmem_initialized=True,
    )
