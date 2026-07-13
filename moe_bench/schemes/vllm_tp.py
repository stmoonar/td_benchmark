"""a1 dataflow: AG(bf16) -> vLLM fused_experts -> optional shared expert -> RS(bf16)."""

from __future__ import annotations

from typing import Any

from .base import SchemeInstance, SchemeSpec, make_lazy_instance, validate_tunables
from .runtime import require_data, routing_full, tp_weight_views, shared_tp, BLOCK_SHAPE


SPEC = SchemeSpec(
    code="a1",
    name="vLLM-TP-Serial",
    family="baseline",
    parallel="TP",
    comm="AG(bf16) -> vLLM fused_experts -> [+shared] -> RS(bf16)",
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
    weights = tp_weight_views(bundle, ctx.rank, ctx.world_size)
    topk_ids, topk_weights = routing_full(bundle)
    shared = shared_tp(bundle, cfg, ctx)

    M = cfg.shape.M_aligned(ctx.world_size)
    M_local = M // ctx.world_size
    K = cfg.shape.K

    hidden_full = torch.empty(M, K, device=ctx.device, dtype=torch.bfloat16)
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
        result = fused_experts(
            hidden_states=hidden_full,
            w1=weights["w1"],
            w2=weights["w2"],
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=cfg.shape.E,
            quant_config=quant_config,
        )
        if shared is not None:
            result = result + shared(hidden_full)
        dist.reduce_scatter_tensor(output, result, group=ctx.group)
        return output

    return make_lazy_instance(SPEC, tunables, run)
