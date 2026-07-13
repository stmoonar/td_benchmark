"""Dispatch pre-compute cost: argsort x2, bincount, cumsum vs (M, E, top_k)."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch


def bench_gpu(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return statistics.median(starts[i].elapsed_time(ends[i]) for i in range(repeat))


def main() -> int:
    parser = argparse.ArgumentParser(description="MoE dispatch pre-compute cost")
    parser.add_argument("--output", default="results/microbench_moe_align.jsonl")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for M in [1024, 5120]:
        for E in [64, 128]:
            for top_k in [8, 16]:
                gen = torch.Generator(device=device).manual_seed(42)
                topk_ids = torch.randint(0, E, (M, top_k), generator=gen, device=device, dtype=torch.int32)
                flat = topk_ids.flatten()

                t_sort1 = bench_gpu(lambda: torch.argsort(flat, stable=True), args.warmup, args.repeat)
                order = torch.argsort(flat, stable=True)
                t_sort2 = bench_gpu(lambda: torch.argsort(order, stable=True), args.warmup, args.repeat)
                t_bincount = bench_gpu(lambda: torch.bincount(flat.long(), minlength=E), args.warmup, args.repeat)
                counts = torch.bincount(flat.long(), minlength=E)
                t_cumsum = bench_gpu(lambda: torch.cumsum(counts, dim=0), args.warmup, args.repeat)

                row = {
                    "kind": "microbench", "script": "mb_moe_align.py", "exp_id": "EXP-010",
                    "M": M, "E": E, "top_k": top_k,
                    "argsort1_ms": t_sort1, "argsort2_ms": t_sort2,
                    "bincount_ms": t_bincount, "cumsum_ms": t_cumsum,
                    "total_ms": t_sort1 + t_sort2 + t_bincount + t_cumsum,
                }
                rows.append(row)
                print(json.dumps(row))

    with output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Results written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def main() -> int:
    return run_microbench_plan(
        __file__,
        "EXP-010",
        "dispatch pre-sort/align/pad cost",
        {"M": [1024, 2048, 4096, 6647], "E": [64, 128], "top_k": [8, 16]},
        ["latency_ms", "pad_overhead_ratio", "effective_tokens_per_s"],
        "implement mb_moe_align with server kernels and results.jsonl output",
    )


if __name__ == "__main__":
    raise SystemExit(main())
