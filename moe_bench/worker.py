from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist

from .config import RunCfg, run_cfg_from_dict
from .context import DistContext, init_context, finalize_context
from .data.bundle import DataBundle, build_data_bundle
from .logging_util import get_logger, log_diagnostics
from .results import JsonlResultSink
from .schemes import REGISTRY, build_scheme
from .schemes.base import SchemeInstance
from .timing import bench_cuda, summarize_samples
from .verify import compare_outputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MoE Bench v2 worker")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--point-json", required=True)
    args = parser.parse_args(argv)
    point = json.loads(args.point_json)
    run_dir = Path(args.run_dir)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    cfg = run_cfg_from_dict(point["config"])
    ctx = init_context(cfg)
    try:
        return _run_point(run_dir, point, cfg, ctx)
    finally:
        finalize_context(ctx)


def _run_point(run_dir: Path, point: dict[str, Any], cfg: RunCfg, ctx: DistContext) -> int:
    torch.cuda.set_device(ctx.device)

    data = build_data_bundle(cfg, ctx, device=ctx.device)
    logger = get_logger(run_dir=run_dir, rank=ctx.rank, level=cfg.log.level)
    log_diagnostics(logger, cfg.log.diagnostics, lambda: _diagnostics_line(point, cfg, ctx, data))

    sink = JsonlResultSink(run_dir / "results.jsonl") if ctx.rank == 0 else None
    a1_avg = None

    for scheme_cfg in cfg.schemes:
        code = scheme_cfg.code
        spec = REGISTRY[code][0]
        if ctx.rank == 0:
            logger.info(f"Running scheme {code} ({spec.name})...")

        instance = build_scheme(cfg, ctx, data, scheme_cfg)
        try:
            summary, samples, last_output = bench_cuda(
                instance.run, warmup=cfg.warmup, repeat=cfg.repeat
            )

            if cfg.profile.torch_profiler:
                trace_dir = run_dir / "profiles" / "torch"
                trace_dir.mkdir(parents=True, exist_ok=True)
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                ) as prof:
                    for _ in range(3):
                        instance.run()
                    torch.cuda.synchronize()
                prof.export_chrome_trace(str(trace_dir / f"{code}_rank{ctx.rank}.json"))
                if ctx.rank == 0:
                    (trace_dir / f"{code}_top_kernels.txt").write_text(
                        prof.key_averages().table(sort_by="cuda_time_total", row_limit=20),
                        encoding="utf-8",
                    )

            if os.environ.get("MOE_BENCH_NSYS_CAPTURE") == "1":
                _capture_nsys_range(instance, ctx, code)

            verify_result = _reduce_verify_across_ranks(_verify(last_output, data, spec, cfg, ctx), ctx)

            if code == "a1":
                a1_avg = summary["avg"]
            speedup = (a1_avg / summary["avg"]) if a1_avg and summary["avg"] > 0 else None

            if ctx.rank == 0:
                row = _build_row(run_dir, point, cfg, scheme_cfg, summary, samples, verify_result, speedup, ctx)
                assert sink is not None
                sink.append(row)
                pass_str = "PASS" if verify_result["pass"] else "FAIL"
                logger.info(
                    f"  {spec.name:40s} avg={summary['avg']:.3f} ms  "
                    f"min={summary['min']:.3f} ms  med={summary['med']:.3f} ms  "
                    f"{pass_str}  max_abs={verify_result['max_abs']:.6f}  rel_p99={verify_result.get('rel_p99', 0.0):.6f}  cos_sim={verify_result['cos_sim']:.8f}"
                )
        finally:
            try:
                del last_output
            except UnboundLocalError:
                pass
            instance.close()
            del instance
            torch.cuda.empty_cache()

    return 0


def _capture_nsys_range(instance: SchemeInstance, ctx: DistContext, scheme: str) -> None:
    """Capture steady-state iterations when launched under nsys cudaProfilerApi mode."""
    iterations = int(os.environ.get("MOE_BENCH_PROFILE_ITERS", "3"))
    if iterations <= 0:
        raise ValueError("MOE_BENCH_PROFILE_ITERS must be positive")
    if dist.is_initialized():
        dist.barrier()
    torch.cuda.cudart().cudaProfilerStart()
    try:
        for iteration in range(iterations):
            torch.cuda.nvtx.range_push(f"PROFILE_{scheme}_ITERATION_{iteration}")
            try:
                instance.run()
            finally:
                torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
    finally:
        torch.cuda.cudart().cudaProfilerStop()
    if dist.is_initialized():
        dist.barrier()


# 双门阈值：EXP-021 校准值（B1-6）
_DEFAULT_GATES: dict[str, dict[str, float]] = {
    "golden_fp8sim_group128": {"cos_threshold": 0.995, "rel_p99_threshold": 5.0, "scale_tolerance": 0.05},
}


def _verify(output: Any, data: DataBundle, spec: Any, cfg: RunCfg, ctx: DistContext) -> dict[str, Any]:
    if not cfg.verify_enabled:
        return {"max_abs": 0.0, "max_rel": 0.0, "cos_sim": 1.0, "rel_p99": 0.0, "rel_masked_max": 0.0, "scale": 1.0, "pass": True, "vs": "skipped"}

    if spec.weight_dtype != "fp8" or spec.act_quant != "group128":
        raise RuntimeError(
            f"active benchmark scheme {spec.code} violates FP8 contract: "
            f"weight_dtype={spec.weight_dtype}, act_quant={spec.act_quant}"
        )
    golden_full = data.golden.fp8sim_group128
    vs = "golden_fp8sim_group128"

    m_local = output.shape[0]
    golden_local = golden_full[ctx.rank * m_local : (ctx.rank + 1) * m_local]

    if os.environ.get("MOE_BENCH_DUMP_VERIFY") == "1":
        dump_dir = Path(os.environ.get("MOE_BENCH_DUMP_DIR", "results/verify_dump"))
        dump_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "scheme": spec.code,
                "rank": ctx.rank,
                "output": output.detach().float().cpu(),
                "golden_local": golden_local.detach().float().cpu(),
                "golden_full": golden_full.detach().float().cpu(),
            },
            dump_dir / f"{spec.code}_rank{ctx.rank}.pt",
        )

    gates = dict(_DEFAULT_GATES.get(vs, {"cos_threshold": 0.99, "rel_p99_threshold": 5.0, "scale_tolerance": 0.05}))
    gates.update(cfg.verify.tolerances.get(vs, {}))
    result = compare_outputs(output, golden_local, atol=0.2, rtol=0.0, **gates)
    result["vs"] = vs
    return result


def _reduce_verify_across_ranks(result: dict[str, Any], ctx: DistContext) -> dict[str, Any]:
    """Make the persisted verification gate represent every rank, not rank 0 only."""
    if not dist.is_initialized() or ctx.world_size == 1:
        result["world_size"] = ctx.world_size
        return result

    max_keys = ["max_abs", "max_rel", "rel_p99", "rel_masked_max"]
    maxima = torch.tensor([float(result[key]) for key in max_keys], device=ctx.device, dtype=torch.float32)
    dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
    for key, value in zip(max_keys, maxima.cpu().tolist(), strict=True):
        result[key] = float(value)

    minima = torch.tensor(
        [float(result["cos_sim"]), float(result["scale"]), 1.0 if result["pass"] else 0.0],
        device=ctx.device,
        dtype=torch.float32,
    )
    maxima = torch.tensor([float(result["scale"])], device=ctx.device, dtype=torch.float32)
    dist.all_reduce(minima, op=dist.ReduceOp.MIN)
    dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
    result["cos_sim"] = float(minima[0].cpu())
    result["scale_min"] = float(minima[1].cpu())
    result["scale_max"] = float(maxima[0].cpu())
    result["scale"] = max(
        (result["scale_min"], result["scale_max"]),
        key=lambda value: abs(value - 1.0),
    )
    result["pass"] = bool(minima[2].cpu())
    result["world_size"] = ctx.world_size
    return result


def _build_row(
    run_dir: Path, point: dict, cfg: RunCfg, scheme_cfg: Any,
    summary: dict, samples: list[float], verify: dict, speedup: float | None,
    ctx: DistContext,
) -> dict:
    shape = cfg.shape
    return {
        "run_id": run_dir.name,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "point_index": point["index"],
        "point_values": point["values"],
        "scheme": scheme_cfg.code,
        "rank": ctx.rank,
        "world_size": ctx.world_size,
        "latency_aggregation": "per_iteration_max_across_ranks",
        "shape": {
            "M": shape.M,
            "K": shape.K,
            "E": shape.E,
            "top_k": shape.top_k,
            "n_gateup": shape.n_gateup,
            "n_down": shape.n_down,
            "shared": shape.shared_experts,
        },
        "routing": {
            "kind": cfg.routing.imbalance_kind,
            "included_in_timing": False,
            "contract": "gate_and_topk_precomputed",
        },
        "quantization": {
            "activation": {"dtype": "fp8_e4m3fn", "granularity": "group128", "group_size": 128},
            "weight": {"dtype": "fp8_e4m3fn", "granularity": "block", "block_shape": [128, 128]},
        },
        "tunables": scheme_cfg.tunables,
        "shard_level": _shard_level(cfg, scheme_cfg.code),
        "lat_ms": summary,
        "samples_ms": samples,
        "stage_ms": {},
        "verify": verify,
        "speedup_vs_a1": speedup,
    }


def _diagnostics_line(point: dict, cfg: RunCfg, ctx: DistContext, data: DataBundle) -> str:
    return (
        f"point={point['index']} "
        f"rank={ctx.rank}/{ctx.world_size} "
        f"hidden_local={list(data.hidden_local.shape)} "
        f"routing_topk={list(data.routing.topk_ids_local.shape)} "
        f"imbalance={data.routing.stats.imbalance_factor:.4f} "
        f"cv={data.routing.stats.cv:.4f}"
    )


def _shard_level(cfg: RunCfg, scheme_code: str) -> str:
    spec = REGISTRY[scheme_code][0]
    if getattr(spec, "parallel", "TP") == "EP":
        return "L0"
    per_rank_intermediate = cfg.shape.n_down // cfg.dist.nproc
    return "L0" if per_rank_intermediate % 128 == 0 else "L2"


if __name__ == "__main__":
    raise SystemExit(main())
