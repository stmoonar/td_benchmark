from __future__ import annotations

from typing import Any


def plan_ragged_block_quant_gemm_repro(M: int, K: int, n_gateup: int, world_size: int, block: int = 128) -> dict[str, Any]:
    if world_size <= 0:
        raise ValueError("world_size must be > 0")
    if n_gateup % 2 != 0:
        raise ValueError("n_gateup must be even")
    intermediate = n_gateup // 2
    if intermediate % world_size != 0:
        raise ValueError("intermediate must be divisible by world_size")
    per_rank = intermediate // world_size
    padded = _ceil_to(per_rank, block)
    return {
        "M": M,
        "K": K,
        "n_gateup": n_gateup,
        "world_size": world_size,
        "block": block,
        "intermediate": intermediate,
        "intermediate_per_rank": per_rank,
        "requires_l1_requant": per_rank % block != 0,
        "l2_padded_intermediate": padded,
        "l2_compute_overhead": padded / per_rank,
        "kernel_checks": [
            "vllm_fused_moe_group128",
            "tdx_fp8_group_gemm",
        ],
    }


def _ceil_to(value: int, block: int) -> int:
    return ((value + block - 1) // block) * block
