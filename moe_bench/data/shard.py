from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .checkpoint import dequantize_block128, quantize_block128


@dataclass(frozen=True)
class ShardPlan:
    columns: int
    world_size: int
    block: int = 128
    require_aligned_kernel: bool = False

    @property
    def columns_per_rank(self) -> int:
        _require(self.columns % self.world_size == 0, "columns must be divisible by world_size")
        return self.columns // self.world_size

    def preferred_level(self) -> str:
        if self.columns_per_rank % self.block == 0:
            return "L0"
        return "L2" if self.require_aligned_kernel else "L1"


@dataclass
class ShardedTensor:
    level: str
    fp8: Any | None
    scale: Any | None
    bf16: Any
    original_columns: int
    padded_columns: int
    compute_overhead: float = 1.0


def shard_columns(fp8: Any, scale: Any, bf16: Any, rank: int, world_size: int, block: int = 128) -> ShardedTensor:
    columns = fp8.shape[1]
    start, end = _rank_bounds(columns, rank, world_size)
    if start % block == 0 and end % block == 0:
        scale_start = start // block
        scale_end = end // block
        fp8_shard = fp8[:, start:end, :].contiguous()
        scale_shard = scale[:, scale_start:scale_end, :].contiguous()
        return ShardedTensor("L0", fp8_shard, scale_shard, dequantize_block128(fp8_shard, scale_shard), end - start, end - start)
    bf16_shard = bf16[:, start:end, :].contiguous()
    req_fp8, req_scale = quantize_block128(bf16_shard.float())
    return ShardedTensor("L1", req_fp8, req_scale, dequantize_block128(req_fp8, req_scale), end - start, end - start)


def shard_with_padding(bf16: Any, rank: int, world_size: int, block: int = 128) -> ShardedTensor:
    torch = _torch()
    columns = bf16.shape[1]
    start, end = _rank_bounds(columns, rank, world_size)
    shard = bf16[:, start:end, :].contiguous()
    original = end - start
    padded_columns = ((original + block - 1) // block) * block
    if padded_columns == original:
        padded = shard
    else:
        pad_shape = list(shard.shape)
        pad_shape[1] = padded_columns - original
        padding = torch.zeros(tuple(pad_shape), dtype=shard.dtype, device=shard.device)
        padded = torch.cat([shard, padding], dim=1).contiguous()
    return ShardedTensor("L2", None, None, padded, original, padded_columns, padded_columns / original)


def _rank_bounds(columns: int, rank: int, world_size: int) -> tuple[int, int]:
    _require(world_size > 0, "world_size must be > 0")
    _require(0 <= rank < world_size, "rank must be in [0, world_size)")
    _require(columns % world_size == 0, "columns must be divisible by world_size")
    per_rank = columns // world_size
    start = rank * per_rank
    return start, start + per_rank


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _torch() -> Any:
    import torch

    return torch
