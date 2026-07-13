from __future__ import annotations

import importlib
import os
from typing import Any

from .base import SchemeInstance, SchemeSpec, TunableSpec, make_lazy_instance, missing_dependency, validate_tunables
from .runtime import ep_weight_views, require_data, routing_local, tp_weight_views


BF16_EP_TUNABLES = {
    "num_dispatch_warps": TunableSpec(int, 8, "Dispatch kernel warps"),
    "num_combine_warps": TunableSpec(int, 8, "Combine kernel warps"),
    "num_sm": TunableSpec(int, 110, "A2A communication SM allocation"),
    "bf16_fwd_gemm_block_size_n": TunableSpec(int, 256, "Fork default BF16 EP forward GEMM block N"),
    "bf16_fwd_gemm_block_size_k": TunableSpec(int, 64, "Fork default BF16 EP forward GEMM block K"),
    "bf16_fwd_gemm_num_stages": TunableSpec(int, 3, "Fork default BF16 EP forward GEMM stages"),
    "gemm_group_size_m": TunableSpec(int, 1, "Fork default grouped GEMM M swizzle group"),
    "capacity": TunableSpec(int, 0, "Optional EP capacity override; 0 means runtime default"),
    "gather_scatter_num_warps": TunableSpec(int, 16, "Gather/scatter index kernel warps"),
}

BF16_TP_TUNABLES = {
    "n_chunks_rs": TunableSpec(int, 32, "Reduce-scatter chunks"),
    "gemm_block_m": TunableSpec(int, 128, "GEMM block M"),
    "gemm_block_n": TunableSpec(int, 128, "GEMM block N"),
    "gemm_block_k": TunableSpec(int, 64, "GEMM block K"),
    "gemm_group_size_m": TunableSpec(int, 1, "Grouped GEMM M swizzle group"),
    "gemm_num_warps": TunableSpec(int, 8, "GEMM launch warps"),
    "gemm_num_stages": TunableSpec(int, 3, "GEMM launch stages"),
}

FP8_EP_TUNABLES = {
    "fp8_rs_enabled": TunableSpec(bool, True, "Enable FP8 reduce-scatter/combine path"),
    "num_combine_sms": TunableSpec(int, 8, "Combine SM allocation"),
    "num_reduce_sms_in_combine": TunableSpec(int, 100, "Reduce SM allocation inside combine"),
    "fp8_fwd_gemm_block_size_n": TunableSpec(int, 128, "Fork default FP8 EP forward GEMM block N"),
    "fp8_fwd_gemm_block_size_k": TunableSpec(int, 128, "Fork default FP8 EP forward GEMM block K"),
    "fp8_fwd_gemm_num_stages": TunableSpec(int, 4, "Fork default FP8 EP forward GEMM stages"),
    "gemm_group_size_m": TunableSpec(int, 1, "Fork default grouped GEMM M swizzle group"),
    "gemm_num_warps": TunableSpec(int, 8, "Fork default FP8 grouped GEMM warps"),
    "num_dispatch_warps": TunableSpec(int, 8, "Dispatch kernel warps"),
    "num_sm": TunableSpec(int, 110, "A2A communication SM allocation"),
    "capacity": TunableSpec(int, 0, "Optional EP capacity override; 0 means runtime default"),
    "gather_scatter_num_warps": TunableSpec(int, 16, "Gather/scatter index kernel warps"),
}

FP8_TP_TUNABLES = {
    "n_chunks_rs": TunableSpec(int, 32, "Reduce-scatter chunks"),
    "gemm_block_m": TunableSpec(int, 128, "FP8 GEMM block M"),
    "gemm_block_n": TunableSpec(int, 128, "FP8 GEMM block N"),
    "gemm_block_k": TunableSpec(int, 128, "FP8 GEMM block K"),
    "gemm_group_size_m": TunableSpec(int, 8, "FP8 grouped GEMM M swizzle group"),
    "gemm_num_warps": TunableSpec(int, 8, "FP8 GEMM launch warps"),
    "gemm_num_stages": TunableSpec(int, 4, "FP8 GEMM launch stages"),
}


def build_td_instance(spec: SchemeSpec, import_target: str, cfg: Any, ctx: Any, data: Any, scheme_cfg: Any) -> SchemeInstance:
    tunables = validate_tunables(spec, scheme_cfg.tunables)

    from moe_bench.tdx.compat import apply
    apply()

    try:
        module = importlib.import_module(import_target)
    except ModuleNotFoundError as exc:
        missing_dependency(f"SERVER-VERIFY: {spec.code} requires migrated {import_target} and clean triton_dist upstream on GPU server")
        raise exc

    bundle = require_data(data)
    topk_ids, topk_weights = routing_local(bundle)
    if spec.parallel == "EP":
        weights = ep_weight_views(bundle, ctx.rank, ctx.world_size)
        if spec.weight_dtype == "bf16":
            run_fn, close_fn = _build_ep_bf16(module, cfg, ctx, bundle, weights, topk_ids, topk_weights, tunables)
        else:
            run_fn, close_fn = _build_ep_fp8(module, cfg, ctx, bundle, weights, topk_ids, topk_weights, tunables)
    else:
        weights = tp_weight_views(bundle, ctx.rank, ctx.world_size)
        run_fn, close_fn = _build_tp(module, spec, cfg, ctx, bundle, weights, tunables)

    if os.environ.get("MOE_BENCH_TDX_AUDIT") == "1":
        import sys as _sys
        for name in sorted(n for n in _sys.modules if n.startswith(("triton_dist", "moe_bench.tdx"))):
            print(f"TDX-AUDIT {name} -> {getattr(_sys.modules[name], '__file__', None)}", flush=True)

    return make_lazy_instance(
        spec,
        tunables,
        run_fn,
        diagnostics={"tdx_import": import_target},
        close_impl=close_fn,
    )


def _run_with_tdx_layer(spec: SchemeSpec, module: Any, cfg: Any, ctx: Any, data: Any, tunables: dict[str, Any]) -> Any:
    bundle = require_data(data)
    topk_ids, topk_weights = routing_local(bundle)
    if spec.parallel == "EP":
        weights = ep_weight_views(bundle, ctx.rank, ctx.world_size)
        if spec.weight_dtype == "bf16":
            return _run_ep_bf16(module, cfg, ctx, bundle, weights, topk_ids, topk_weights, tunables)
        return _run_ep_fp8(module, cfg, ctx, bundle, weights, topk_ids, topk_weights, tunables)
    weights = tp_weight_views(bundle, ctx.rank, ctx.world_size)
    return _run_tp(module, spec, cfg, ctx, bundle, weights, tunables)


def _build_ep_bf16(module: Any, cfg: Any, ctx: Any, bundle: Any, weights: dict[str, Any], topk_ids: Any, topk_weights: Any, tunables: dict[str, Any]) -> Any:
    import torch
    from moe_bench.tdx.layers.ep_moe import EP_MoE
    from moe_bench.tdx.function.ep_moe_fused import TritonDistFusedEpMoeFunction
    from .runtime import shared_ep

    M_local = cfg.shape.M_aligned(ctx.world_size) // ctx.world_size
    moe = EP_MoE(rank=ctx.rank, world_size=ctx.world_size, group=ctx.group)
    moe.top_k = cfg.shape.top_k
    moe.num_experts = cfg.shape.E
    moe.hidden_size = cfg.shape.K
    moe.dtype = torch.bfloat16
    moe.act_fn = torch.nn.functional.silu
    moe.gate = torch.empty(cfg.shape.E, cfg.shape.K, device=ctx.device, dtype=torch.bfloat16)
    moe.gate_up_proj = weights["w1_bf16"]
    moe.down_proj = weights["w2_bf16"]
    moe._init_ctx(ctx.group, max_tokens_per_rank=M_local)

    shared = shared_ep(bundle, cfg, ctx)
    w1 = weights["w1_bf16"]
    w2 = weights["w2_bf16"]

    def run() -> Any:
        result = TritonDistFusedEpMoeFunction.apply(
            cfg.shape.E, topk_weights, topk_ids, bundle.hidden_local, w1, None, w2, ctx.group,
        )
        if shared is not None:
            result = result + shared(bundle.hidden_local)
        return result

    return run, moe.finalize


def _build_ep_fp8(module: Any, cfg: Any, ctx: Any, bundle: Any, weights: dict[str, Any], topk_ids: Any, topk_weights: Any, tunables: dict[str, Any]) -> Any:
    import torch
    from moe_bench.tdx.layers.fp8_ep_moe import FP8_EP_MoE, _quantize_fp8_blockwise
    from moe_bench.tdx.function.ep_moe_fused import TritonDistFusedFp8EpMoeFunction
    from .runtime import shared_ep

    M_local = cfg.shape.M_aligned(ctx.world_size) // ctx.world_size

    fp8_moe = FP8_EP_MoE(rank=ctx.rank, world_size=ctx.world_size, group=ctx.group)
    fp8_moe.top_k = cfg.shape.top_k
    fp8_moe.num_experts = cfg.shape.E
    fp8_moe.hidden_size = cfg.shape.K
    fp8_moe.enable_fp8_rs = tunables.get("fp8_rs_enabled", True)
    fp8_moe._init_ctx(ctx.group, max_tokens_per_rank=M_local)

    shared = shared_ep(bundle, cfg, ctx)
    w1 = weights["w1"]
    w1_scale = weights["w1_scale"]
    w2 = weights["w2"]
    w2_scale = weights["w2_scale"]

    def run() -> Any:
        hidden_fp8, hidden_scale = _quantize_fp8_blockwise(bundle.hidden_local, block_k=128)
        result = TritonDistFusedFp8EpMoeFunction.apply(
            cfg.shape.E, topk_weights, topk_ids, hidden_fp8, hidden_scale,
            w1, w1_scale, None, None, w2, w2_scale, ctx.group,
        )
        if shared is not None:
            result = result + shared(bundle.hidden_local)
        return result

    return run, fp8_moe.finalize


def _build_tp(module: Any, spec: SchemeSpec, cfg: Any, ctx: Any, bundle: Any, weights: dict[str, Any], tunables: dict[str, Any]) -> Any:
    import torch
    import triton
    from .runtime import shared_ep

    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device=ctx.device, dtype=torch.int8)
    triton.set_allocator(alloc_fn)

    M = cfg.shape.M_aligned(ctx.world_size)
    shared = shared_ep(bundle, cfg, ctx)

    if spec.code == "c2":
        from triton_dist.layers.nvidia.tp_moe import TP_MoE
        moe = TP_MoE(rank=ctx.rank, world_size=ctx.world_size, group=ctx.group)
        moe.top_k = cfg.shape.top_k
        moe.num_experts = cfg.shape.E
        moe.hidden_size = cfg.shape.K
        moe.dtype = torch.bfloat16
        moe.act_fn = torch.nn.functional.silu
        moe.gate = bundle.ckpt.gate_weight.to(ctx.device).to(torch.bfloat16)
        moe.gate_up_proj = weights["w1_bf16"].transpose(1, 2).contiguous()
        moe.down_proj = weights["w2_bf16"].transpose(1, 2).contiguous()
        moe._init_ctx(M=M)
        x_3d = bundle.hidden_local.unsqueeze(1).contiguous()

        def run() -> Any:
            out_3d = moe.dist_triton_fwd(x_3d)
            result = out_3d.squeeze(1)
            if shared is not None:
                result = result + shared(bundle.hidden_local)
            return result
        return run, moe.finalize

    cls_name = "FP8_TP_MoE_FP8RS" if spec.code == "c5" else "FP8_TP_MoE"
    layer_module = importlib.import_module("moe_bench.tdx.layers.fp8_tp_moe")
    cls = getattr(layer_module, cls_name)
    layer = cls(
        rank=ctx.rank,
        world_size=ctx.world_size,
        group=ctx.group,
        block_k_quant=128,
        block_n_quant=128,
        gemm_block_m=tunables["gemm_block_m"],
        gemm_block_n=tunables["gemm_block_n"],
        gemm_block_k=tunables["gemm_block_k"],
        gemm_group_size_m=tunables["gemm_group_size_m"],
        gemm_num_warps=tunables["gemm_num_warps"],
        gemm_num_stages=tunables["gemm_num_stages"],
    )
    layer._init_parameters_from_bf16(
        gate_up_proj_bf16=weights["w1_bf16"].transpose(1, 2).contiguous(),
        down_proj_bf16=weights["w2_bf16"].transpose(1, 2).contiguous(),
        num_experts=cfg.shape.E,
        top_k=cfg.shape.top_k,
        hidden_size=cfg.shape.K,
        verbose=(ctx.rank == 0),
    )
    layer._init_ctx(M=M)

    topk_ids_full = bundle.routing.topk_ids_full.contiguous()
    topk_weights_full = bundle.routing.topk_weights_full.contiguous()
    n_chunks = tunables.get("n_chunks_rs", 32)

    def run() -> Any:
        result = layer.dist_triton_fwd(
            bundle.hidden_local, full_topk_ids=topk_ids_full,
            full_topk_weight=topk_weights_full, n_chunks_rs=n_chunks,
        )
        if shared is not None:
            result = result + shared(bundle.hidden_local)
        return result
    return run, layer.finalize
