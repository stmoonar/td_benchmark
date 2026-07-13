#!/usr/bin/env python3
"""Measure exposed preprocessing/quantization costs for b2 and c4 on four ranks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--M", type=int, default=5120)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--E", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()

    import torch
    import torch.distributed as dist

    from moe_bench.context import finalize_context, init_context
    from moe_bench.timing import bench_cuda

    # TD's JIT post-compile hook initializes CUDA modules against NVSHMEM even
    # for layout helpers that do not themselves communicate. Use the same
    # NVSHMEM-enabled context as c4 instead of a plain NCCL-only context.
    td_cfg = SimpleNamespace(schemes=[SimpleNamespace(code="c4")])
    ctx = init_context(td_cfg)
    try:
        if args.M % ctx.world_size:
            raise ValueError(f"M={args.M} must be divisible by world_size={ctx.world_size}")
        generator = torch.Generator(device=ctx.device)
        generator.manual_seed(20260713)
        logits = torch.rand((args.M, args.E), generator=generator, device=ctx.device)
        topk_ids = logits.topk(args.top_k, dim=-1).indices.to(torch.int32).contiguous()
        hidden_local = torch.randn(
            (args.M // ctx.world_size, args.K),
            generator=generator,
            device=ctx.device,
            dtype=torch.bfloat16,
        )

        from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
        from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
        from moe_bench.tdx.kernels.fp8_allgather_group_gemm import sort_topk_ids_align_block_size
        from moe_bench.tdx.kernels.fp8_moe_reduce_rs import calc_gather_scatter_index_v2_triton
        from moe_bench.tdx.layers.fp8_tp_moe import quantize_fp8_blockwise

        measurements: dict[str, dict[str, Any]] = {}

        def measure(name: str, fn: Callable[[], Any]) -> None:
            summary, samples, _ = bench_cuda(fn, warmup=args.warmup, repeat=args.repeat)
            measurements[name] = {"lat_ms": summary, "samples_ms": samples}

        measure("b2_moe_align_block64", lambda: moe_align_block_size(topk_ids, 64, args.E, None))
        measure("b2_moe_align_block128", lambda: moe_align_block_size(topk_ids, 128, args.E, None))
        measure(
            "c4_gateup_sort_layout",
            lambda: sort_topk_ids_align_block_size(
                topk_ids, args.E, ctx.rank, ctx.world_size, ctx.world_size, 128
            ),
        )
        measure(
            "c4_down_gather_scatter_layout",
            lambda: calc_gather_scatter_index_v2_triton(topk_ids, args.E, 128),
        )
        measure(
            "b2_group128_input_quant",
            lambda: moe_kernel_quantize_input(
                hidden_local, None, torch.float8_e4m3fn, False, [128, 128]
            ),
        )
        measure(
            "c4_group128_input_quant",
            lambda: quantize_fp8_blockwise(hidden_local, torch.float8_e4m3fn, 128),
        )

        # Quantized payload + its group128 scales, matching the two collectives
        # that b2 enqueues for FP8 all-gather chunks.
        local_fp8, local_scale = quantize_fp8_blockwise(hidden_local, torch.float8_e4m3fn, 128)
        gathered_fp8 = torch.empty((args.M, args.K), device=ctx.device, dtype=torch.float8_e4m3fn)
        gathered_scale = torch.empty((args.M, args.K // 128), device=ctx.device, dtype=torch.float32)

        def nccl_fp8_and_scale_ag() -> tuple[Any, Any]:
            dist.all_gather_into_tensor(gathered_fp8.view(torch.uint8), local_fp8.view(torch.uint8))
            dist.all_gather_into_tensor(gathered_scale, local_scale)
            return gathered_fp8, gathered_scale

        measure("nccl_full_fp8_plus_scale_allgather", nccl_fp8_and_scale_ag)
        measure(
            "c4_exposed_scale_allgather",
            lambda: dist.all_gather_into_tensor(gathered_scale, local_scale),
        )

        if ctx.rank == 0:
            payload = {
                "shape": {"M": args.M, "K": args.K, "E": args.E, "top_k": args.top_k},
                "rank": ctx.rank,
                "world_size": ctx.world_size,
                "warmup": args.warmup,
                "repeat": args.repeat,
                "CUDA_DEVICE_MAX_CONNECTIONS": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "unset"),
                "measurements": measurements,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(f"Component benchmark: {args.output}")
    finally:
        finalize_context(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
