from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from moe_bench.config import RoutingCfg, ShapeCfg
from moe_bench.context import DistContext


@dataclass
class RoutingStats:
    per_expert_counts: Any
    imbalance_factor: float
    cv: float
    per_owner_counts: Any
    hot_expert_share: float


@dataclass
class RoutingBundle:
    topk_ids_local: Any
    topk_weights_local: Any
    topk_ids_full: Any
    topk_weights_full: Any
    stats: RoutingStats


def target_distribution(cfg: RoutingCfg, experts: int, device: Any) -> Any:
    torch = _torch()
    if cfg.imbalance_kind == "none":
        return torch.full((experts,), 1.0 / experts, dtype=torch.float32, device=device)
    if cfg.imbalance_kind == "zipf":
        ranks = torch.arange(1, experts + 1, dtype=torch.float32, device=device)
        probs = ranks.pow(-cfg.zipf_s)
        permutation = torch.randperm(experts, generator=torch.Generator(device=device).manual_seed(cfg.gate_seed), device=device)
        shuffled = torch.empty_like(probs)
        shuffled[permutation] = probs
        return shuffled / shuffled.sum()
    if cfg.imbalance_kind == "hotspot":
        hot = min(cfg.hot_experts, experts)
        probs = torch.full((experts,), (1.0 - cfg.hot_share) / max(experts - hot, 1), dtype=torch.float32, device=device)
        probs[:hot] = cfg.hot_share / hot
        if hot == experts:
            probs[:] = 1.0 / experts
        return probs / probs.sum()
    raise ValueError(f"unknown imbalance kind {cfg.imbalance_kind}")


def build_routing(
    hidden_local: Any,
    gate_weight: Any,
    cfg: RoutingCfg,
    shape: ShapeCfg,
    ctx: DistContext,
    all_gather: Callable[[Any], Any] | None = None,
) -> RoutingBundle:
    torch = _torch()
    logits = hidden_local.float() @ gate_weight.float().T
    if cfg.imbalance_kind != "none":
        probs = target_distribution(cfg, shape.E, hidden_local.device)
        logits = logits + cfg.strength * torch.log(shape.E * probs).to(logits)
    topk_weights, topk_ids = torch.topk(torch.softmax(logits, dim=-1), shape.top_k, dim=-1)
    topk_weights = (topk_weights / topk_weights.sum(dim=-1, keepdim=True)).to(torch.float32).contiguous()
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_ids_full = _gather_or_local(topk_ids, all_gather)
    topk_weights_full = _gather_or_local(topk_weights, all_gather)
    stats = _stats(torch, topk_ids_full, shape.E, ctx.world_size, cfg)
    return RoutingBundle(
        topk_ids_local=topk_ids,
        topk_weights_local=topk_weights,
        topk_ids_full=topk_ids_full.contiguous(),
        topk_weights_full=topk_weights_full.contiguous(),
        stats=stats,
    )


def _stats(torch: Any, topk_ids: Any, experts: int, world_size: int, cfg: RoutingCfg) -> RoutingStats:
    counts = torch.bincount(topk_ids.reshape(-1).to(torch.int64), minlength=experts).to(torch.int32)
    counts_f = counts.float()
    mean = counts_f.mean().clamp(min=1e-12)
    imbalance = float(counts_f.max() / mean)
    cv = float(counts_f.std(unbiased=False) / mean)
    owners = torch.arange(experts, device=counts.device) % world_size
    owner_counts = torch.zeros((world_size,), dtype=torch.int32, device=counts.device)
    owner_counts.scatter_add_(0, owners, counts)
    hot = min(cfg.hot_experts, experts)
    hot_share = float(counts[:hot].sum().float() / counts.sum().float().clamp(min=1))
    return RoutingStats(counts, imbalance, cv, owner_counts, hot_share)


def _gather_or_local(tensor: Any, all_gather: Callable[[Any], Any] | None) -> Any:
    if all_gather is None:
        return tensor
    gathered = all_gather(tensor)
    return gathered.contiguous()


def _torch() -> Any:
    import torch

    return torch
