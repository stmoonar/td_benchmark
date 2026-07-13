from __future__ import annotations

from typing import Any

from .base import SchemeInstance, SchemeSpec, TunableSpec, make_lazy_instance, validate_tunables
from .runtime import require_data, routing_full, shared_tp, tp_weight_views, BLOCK_SHAPE


OVERLAP_TUNABLES = {
    "n_chunks_gateup": TunableSpec(int, 2, "Number of gate/up all-gather chunks"),
    "n_chunks_down": TunableSpec(int, 4, "Number of down reduce-scatter chunks"),
}


def build_overlap_instance(spec: SchemeSpec, forward_name: str, cfg: Any, ctx: Any, data: Any, scheme_cfg: Any) -> SchemeInstance:
    tunables = validate_tunables(spec, scheme_cfg.tunables)
    bundle = require_data(data)
    weights = tp_weight_views(bundle, ctx.rank, ctx.world_size)
    topk_ids, topk_weights = routing_full(bundle)
    shared = shared_tp(bundle, cfg, ctx)

    import importlib
    overlap_mod = importlib.import_module("moe_bench.overlap")
    forward_fn = getattr(overlap_mod, forward_name)
    MoEOverlapState = getattr(overlap_mod, "MoEOverlapState")

    state = MoEOverlapState(
        n_chunks_gateup=tunables["n_chunks_gateup"],
        n_chunks_down=tunables["n_chunks_down"],
        tp_group=ctx.group,
    )
    state.ensure_streams(ctx.device)

    def run() -> Any:
        kwargs: dict[str, Any] = {
            "state": state,
            "hidden_states": bundle.hidden_local,
            "w1": weights["w1"],
            "w2": weights["w2"],
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
            "top_k": cfg.shape.top_k,
            "global_num_experts": cfg.shape.E,
            "shared_mlp": shared,
            "w1_scale": weights["w1_scale"],
            "w2_scale": weights["w2_scale"],
            "block_shape": BLOCK_SHAPE,
        }
        if forward_name == "moe_forward_overlap":
            kwargs["use_fp8_w8a8"] = True
        return forward_fn(**kwargs)

    return make_lazy_instance(spec, tunables, run)
