from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from moe_bench.config import RunCfg
from moe_bench.context import DistContext
from moe_bench.verify import golden_moe

from .checkpoint import build_checkpoint
from .checkpoint import WeightCheckpoint
from .routing import build_routing
from .routing import RoutingBundle


@dataclass
class GoldenOutputs:
    bf16: Any
    fp8sim_group128: Any
    fp8sim_rowwise: Any


@dataclass
class DataBundle:
    hidden_local: Any
    ckpt: WeightCheckpoint
    routing: RoutingBundle
    golden: GoldenOutputs


def build_data_bundle(cfg: RunCfg, ctx: DistContext, device: Any | None = None) -> DataBundle:
    torch = _torch()
    actual_device = device if device is not None else ctx.device
    hidden_full = _build_hidden(torch, cfg, ctx.world_size, actual_device)
    hidden_local = _rank_slice(hidden_full, ctx.rank, ctx.world_size).to(torch.bfloat16).contiguous()
    ckpt = build_checkpoint(cfg.shape, seed=cfg.seed, device=actual_device)
    all_gather_fn = _make_all_gather(torch, ctx)
    routing = build_routing(hidden_local, ckpt.gate_weight, cfg.routing, cfg.shape, ctx, all_gather=all_gather_fn)
    golden = GoldenOutputs(
        bf16=_golden(hidden_full, routing, ckpt, "none"),
        fp8sim_group128=_golden(hidden_full, routing, ckpt, "group128"),
        fp8sim_rowwise=_golden(hidden_full, routing, ckpt, "rowwise"),
    )
    return DataBundle(hidden_local=hidden_local, ckpt=ckpt, routing=routing, golden=golden)


def _build_hidden(torch: Any, cfg: RunCfg, world_size: int, device: Any) -> Any:
    generator = torch.Generator(device=device).manual_seed(cfg.seed + 17)
    return torch.randn(
        (cfg.shape.M_aligned(world_size), cfg.shape.K),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )


def _rank_slice(hidden_full: Any, rank: int, world_size: int) -> Any:
    local_m = hidden_full.shape[0] // world_size
    start = rank * local_m
    return hidden_full[start : start + local_m]


def _golden(hidden_full: Any, routing: RoutingBundle, ckpt: WeightCheckpoint, act_quant: str) -> Any:
    shared_w1 = None if ckpt.shared_w1 is None else ckpt.shared_w1.bf16
    shared_w2 = None if ckpt.shared_w2 is None else ckpt.shared_w2.bf16
    return golden_moe(hidden_full, routing, ckpt.w1.bf16, ckpt.w2.bf16, shared_w1, shared_w2, act_quant)


def _make_all_gather(torch: Any, ctx: DistContext) -> Any:
    if ctx.world_size <= 1:
        return None
    try:
        import torch.distributed as dist
        if not dist.is_initialized():
            return None
    except (ImportError, RuntimeError):
        return None

    def all_gather_fn(tensor: Any) -> Any:
        gathered = [torch.empty_like(tensor) for _ in range(ctx.world_size)]
        dist.all_gather(gathered, tensor, group=ctx.group)
        return torch.cat(gathered, dim=0)

    return all_gather_fn


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("DataBundle assembly requires torch; import-only module checks do not.") from exc
    return torch
