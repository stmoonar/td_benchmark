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
    torch.cuda.set_device(ctx.device)

    data = build_data_bundle(cfg, ctx, device=ctx.device)
    logger = get_logger(run_dir=run_dir, rank=ctx.rank, level=cfg.log.level)
    log_diagnostics(logger, cfg.log.diagnostics, lambda: _diagnostics_line(point, cfg, ctx, data))

    sink = JsonlResultSink(run_dir / "results.jsonl")
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

            verify_result = _verify(last_output, data, spec, cfg, ctx)

            if code == "a1":
                a1_avg = summary["avg"]
            speedup = (a1_avg / summary["avg"]) if a1_avg and summary["avg"] > 0 else None

            row = _build_row(run_dir, point, cfg, scheme_cfg, summary, samples, verify_result, speedup)
            sink.append(row)

            if ctx.rank == 0:
                pass_str = "PASS" if verify_result["pass"] else "FAIL"
                logger.info(
                    f"  {spec.name:40s} avg={summary['avg']:.3f} ms  "
                    f"min={summary['min']:.3f} ms  med={summary['med']:.3f} ms  "
                    f"{pass_str}  max_abs={verify_result['max_abs']:.6f}  rel_p99={verify_result.get('rel_p99', 0.0):.6f}  cos_sim={verify_result['cos_sim']:.8f}"
                )
        finally:
            instance.close()
            torch.cuda.empty_cache()

    finalize_context()
    return 0


# 双门阈值：EXP-021 校准值（B1-6）
_DEFAULT_GATES: dict[str, dict[str, float]] = {
    "golden_bf16": {"cos_threshold": 0.995, "rel_p99_threshold": 5.0, "scale_tolerance": 0.05},
    "golden_fp8sim_group128": {"cos_threshold": 0.995, "rel_p99_threshold": 5.0, "scale_tolerance": 0.05},
    "golden_fp8sim_rowwise": {"cos_threshold": 0.99, "rel_p99_threshold": 5.0, "scale_tolerance": 0.05},
}


def _verify(output: Any, data: DataBundle, spec: Any, cfg: RunCfg, ctx: DistContext) -> dict[str, Any]:
    if not cfg.verify_enabled:
        return {"max_abs": 0.0, "max_rel": 0.0, "cos_sim": 1.0, "rel_p99": 0.0, "rel_masked_max": 0.0, "scale": 1.0, "pass": True, "vs": "skipped"}

    act_quant = spec.act_quant
    if act_quant == "none":
        golden_full = data.golden.bf16
        vs = "golden_bf16"
    elif act_quant == "group128":
        golden_full = data.golden.fp8sim_group128
        vs = "golden_fp8sim_group128"
    elif act_quant == "rowwise":
        golden_full = data.golden.fp8sim_rowwise
        vs = "golden_fp8sim_rowwise"
    else:
        golden_full = data.golden.bf16
        vs = "golden_bf16"

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


def _build_row(
    run_dir: Path, point: dict, cfg: RunCfg, scheme_cfg: Any,
    summary: dict, samples: list[float], verify: dict, speedup: float | None,
) -> dict:
    shape = cfg.shape
    return {
        "run_id": run_dir.name,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "point_index": point["index"],
        "point_values": point["values"],
        "scheme": scheme_cfg.code,
        "shape": {
            "M": shape.M,
            "K": shape.K,
            "E": shape.E,
            "top_k": shape.top_k,
            "n_gateup": shape.n_gateup,
            "n_down": shape.n_down,
            "shared": shape.shared_experts,
        },
        "routing": {"kind": cfg.routing.imbalance_kind},
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
