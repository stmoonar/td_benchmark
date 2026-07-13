"""a2 dataflow: AG(hidden+routing) -> vLLM fused_experts(expert_map) -> RS -> +shared."""

from __future__ import annotations

from typing import Any

from .base import SchemeInstance, SchemeSpec, make_lazy_instance, validate_tunables
from .runtime import ep_weight_views, require_data, routing_local, shared_ep, BLOCK_SHAPE


SPEC = SchemeSpec(
    code="a2",
    name="vLLM-EP-Naive",
    family="baseline",
    parallel="EP",
    comm="AG(hidden+routing) -> vLLM fused_experts(expert_map) -> RS -> +shared",
    weight_dtype="fp8",
    act_quant="group128",
    requires_nvshmem=False,
    tunables_schema={},
)


def build(cfg: Any, ctx: Any, data: Any, scheme_cfg: Any) -> SchemeInstance:
    import torch
    import torch.distributed as dist

    tunables = validate_tunables(SPEC, scheme_cfg.tunables)
    bundle = require_data(data)
    weights = ep_weight_views(bundle, ctx.rank, ctx.world_size)
    topk_ids_local, topk_weights_local = routing_local(bundle)
    shared = shared_ep(bundle, cfg, ctx)

    M = cfg.shape.M_aligned(ctx.world_size)
    M_local = M // ctx.world_size
    K = cfg.shape.K
    top_k = cfg.shape.top_k

    hidden_full = torch.empty(M, K, device=ctx.device, dtype=torch.bfloat16)
    topk_ids_full = torch.empty(M, top_k, device=ctx.device, dtype=torch.int32)
    topk_weights_full = torch.empty(M, top_k, device=ctx.device, dtype=torch.float32)
    output = torch.empty(M_local, K, device=ctx.device, dtype=torch.bfloat16)

    from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    quant_config = fp8_w8a8_moe_quant_config(
        w1_scale=weights["w1_scale"],
        w2_scale=weights["w2_scale"],
        block_shape=BLOCK_SHAPE,
    )

    def run() -> Any:
        dist.all_gather_into_tensor(hidden_full, bundle.hidden_local.contiguous(), group=ctx.group)
        dist.all_gather_into_tensor(topk_ids_full, topk_ids_local.contiguous(), group=ctx.group)
        dist.all_gather_into_tensor(topk_weights_full, topk_weights_local.contiguous(), group=ctx.group)
        result = fused_experts(
            hidden_states=hidden_full,
            w1=weights["w1"],
            w2=weights["w2"],
            topk_weights=topk_weights_full,
            topk_ids=topk_ids_full,
            expert_map=weights["expert_map"],
            global_num_experts=cfg.shape.E,
            quant_config=quant_config,
        )
        dist.reduce_scatter_tensor(output, result, group=ctx.group)
        if shared is not None:
            output.add_(shared(bundle.hidden_local))
        return output

    return make_lazy_instance(SPEC, tunables, run)
