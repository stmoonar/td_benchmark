from __future__ import annotations

from typing import Any

from moe_bench.data.checkpoint import quantize_block128

from .shared_expert import StaticSharedExpert


BLOCK_SHAPE = [128, 128]


def require_data(data: Any) -> Any:
    if data is None:
        raise RuntimeError("Scheme execution requires a DataBundle; build-only tests may pass data=None.")
    return data


def routing_full(data: Any) -> tuple[Any, Any]:
    return data.routing.topk_ids_full, data.routing.topk_weights_full


def routing_local(data: Any) -> tuple[Any, Any]:
    return data.routing.topk_ids_local, data.routing.topk_weights_local


def tp_weight_views(data: Any, rank: int, world_size: int) -> dict[str, Any]:
    ckpt = data.ckpt
    if world_size == 1:
        return {
            "w1": ckpt.w1.fp8,
            "w1_scale": ckpt.w1.scale,
            "w1_bf16": ckpt.w1.bf16,
            "w2": ckpt.w2.fp8,
            "w2_scale": ckpt.w2.scale,
            "w2_bf16": ckpt.w2.bf16,
            "shard_level": "L0",
        }
    try:
        torch = _torch()
        w1_bf16 = _slice_fused_gateup_tp(ckpt.w1.bf16, rank, world_size)
        w2_bf16 = ckpt.w2.bf16.chunk(world_size, dim=2)[rank].contiguous()

        shard_level = "L0"
        intermediate_per_tp = w2_bf16.shape[2]
        if intermediate_per_tp % 128 != 0:
            padded = ((intermediate_per_tp + 127) // 128) * 128
            pad_n = padded - intermediate_per_tp
            shard_level = "L2"
            E = w1_bf16.shape[0]
            gate_shard, up_shard = w1_bf16.chunk(2, dim=1)
            gate_shard = torch.nn.functional.pad(gate_shard, (0, 0, 0, pad_n))
            up_shard = torch.nn.functional.pad(up_shard, (0, 0, 0, pad_n))
            w1_bf16 = torch.cat([gate_shard, up_shard], dim=1).contiguous()
            w2_bf16 = torch.nn.functional.pad(w2_bf16, (0, pad_n)).contiguous()

        w1, w1_scale = quantize_block128(w1_bf16.float())
        w2, w2_scale = quantize_block128(w2_bf16.float())
        return {
            "w1": w1,
            "w1_scale": w1_scale,
            "w1_bf16": w1_bf16,
            "w2": w2,
            "w2_scale": w2_scale,
            "w2_bf16": w2_bf16,
            "shard_level": shard_level,
        }
    except (AttributeError, TypeError):
        return tp_weight_views(data, rank=0, world_size=1)


def ep_weight_views(data: Any, rank: int, world_size: int) -> dict[str, Any]:
    ckpt = data.ckpt
    if world_size == 1:
        expert_map = "rank-local"
        return {
            "w1": ckpt.w1.fp8,
            "w1_scale": ckpt.w1.scale,
            "w1_bf16": ckpt.w1.bf16,
            "w2": ckpt.w2.fp8,
            "w2_scale": ckpt.w2.scale,
            "w2_bf16": ckpt.w2.bf16,
            "expert_map": expert_map,
        }
    try:
        local_e = ckpt.w1.bf16.shape[0] // world_size
        start = rank * local_e
        end = start + local_e
        w1_bf16 = ckpt.w1.bf16[start:end].contiguous()
        w2_bf16 = ckpt.w2.bf16[start:end].contiguous()
        w1, w1_scale = quantize_block128(w1_bf16.float())
        w2, w2_scale = quantize_block128(w2_bf16.float())
        expert_map = _expert_map(ckpt.w1.bf16.shape[0], rank, world_size, w1_bf16)
        return {
            "w1": w1,
            "w1_scale": w1_scale,
            "w1_bf16": w1_bf16,
            "w2": w2,
            "w2_scale": w2_scale,
            "w2_bf16": w2_bf16,
            "expert_map": expert_map,
        }
    except (AttributeError, TypeError):
        return ep_weight_views(data, rank=0, world_size=1)


def shared_tp(data: Any, cfg: Any, ctx: Any) -> StaticSharedExpert | None:
    if data.ckpt.shared_w1 is None or data.ckpt.shared_w2 is None:
        return None
    if ctx.world_size == 1:
        return StaticSharedExpert(
            data.ckpt.shared_w1.fp8,
            data.ckpt.shared_w1.scale,
            data.ckpt.shared_w2.fp8,
            data.ckpt.shared_w2.scale,
            BLOCK_SHAPE,
            cfg.shape.M_aligned(ctx.world_size),
        )
    try:
        torch = _torch()
        w1_bf16 = _slice_fused_gateup_tp(data.ckpt.shared_w1.bf16, ctx.rank, ctx.world_size)
        w2_bf16 = data.ckpt.shared_w2.bf16.chunk(ctx.world_size, dim=2)[ctx.rank].contiguous()

        intermediate_per_tp = w2_bf16.shape[2]
        if intermediate_per_tp % 128 != 0:
            padded = ((intermediate_per_tp + 127) // 128) * 128
            pad_n = padded - intermediate_per_tp
            gate_shard, up_shard = w1_bf16.chunk(2, dim=1)
            gate_shard = torch.nn.functional.pad(gate_shard, (0, 0, 0, pad_n))
            up_shard = torch.nn.functional.pad(up_shard, (0, 0, 0, pad_n))
            w1_bf16 = torch.cat([gate_shard, up_shard], dim=1).contiguous()
            w2_bf16 = torch.nn.functional.pad(w2_bf16, (0, pad_n)).contiguous()

        w1, w1_scale = quantize_block128(w1_bf16.float())
        w2, w2_scale = quantize_block128(w2_bf16.float())
        return StaticSharedExpert(w1, w1_scale, w2, w2_scale, BLOCK_SHAPE, cfg.shape.M_aligned(ctx.world_size))
    except (AttributeError, TypeError):
        return None


def shared_ep(data: Any, cfg: Any, ctx: Any) -> StaticSharedExpert | None:
    if data.ckpt.shared_w1 is None or data.ckpt.shared_w2 is None:
        return None
    return StaticSharedExpert(
        data.ckpt.shared_w1.fp8,
        data.ckpt.shared_w1.scale,
        data.ckpt.shared_w2.fp8,
        data.ckpt.shared_w2.scale,
        BLOCK_SHAPE,
        cfg.shape.M_aligned(ctx.world_size) // ctx.world_size,
    )


def _slice_fused_gateup_tp(w1_full: Any, rank: int, world_size: int) -> Any:
    gate_full, up_full = w1_full.chunk(2, dim=1)
    gate_shard = gate_full.chunk(world_size, dim=1)[rank]
    up_shard = up_full.chunk(world_size, dim=1)[rank]
    torch = _torch()
    return torch.cat([gate_shard, up_shard], dim=1).contiguous()


def _expert_map(num_experts: int, rank: int, world_size: int, like_tensor: Any) -> Any:
    torch = _torch()
    local_e = num_experts // world_size
    expert_map = torch.full((num_experts,), -1, dtype=torch.int32, device=like_tensor.device)
    expert_map[rank * local_e : (rank + 1) * local_e] = torch.arange(local_e, dtype=torch.int32, device=like_tensor.device)
    return expert_map


def _torch() -> Any:
    import torch

    return torch
