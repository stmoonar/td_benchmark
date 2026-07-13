"""Dispatch preprocess cost: token permutation gather + pinned-host readback sync."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description="dispatch prep and host-sync cost")
    parser.add_argument("--output", default="results/microbench_dispatch_prep.jsonl")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for M, K, top_k in [(1024, 4096, 8), (5120, 4096, 8), (4096, 3200, 16)]:
        gen = torch.Generator(device=device).manual_seed(42)
        hidden = torch.randn(M, K, generator=gen, device=device, dtype=torch.bfloat16)
        perm = torch.randperm(M * top_k, generator=gen, device=device)

        def gather():
            return hidden[perm % M]

        for _ in range(args.warmup):
            gather()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeat)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeat)]
        for i in range(args.repeat):
            starts[i].record()
            gather()
            ends[i].record()
        torch.cuda.synchronize()
        gather_ms = statistics.median(starts[i].elapsed_time(ends[i]) for i in range(args.repeat))

        small = torch.randint(0, M, (8,), device=device, dtype=torch.int32)
        pinned = torch.empty(8, dtype=torch.int32, pin_memory=True)
        filler_a = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
        filler_b = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)

        sync_samples = []
        for _ in range(args.warmup):
            torch.mm(filler_a, filler_b)
            pinned.copy_(small, non_blocking=True)
            torch.cuda.synchronize()
        for _ in range(args.repeat):
            torch.mm(filler_a, filler_b)
            t0 = time.perf_counter()
            pinned.copy_(small, non_blocking=True)
            torch.cuda.synchronize()
            sync_samples.append((time.perf_counter() - t0) * 1000.0)
        host_sync_ms = statistics.median(sync_samples)

        row = {
            "kind": "microbench", "script": "mb_dispatch_prep.py", "exp_id": "EXP-011",
            "M": M, "K": K, "top_k": top_k,
            "token_gather_ms": gather_ms, "host_readback_sync_ms": host_sync_ms,
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

from common import run_microbench_plan


def main() -> int:
    return run_microbench_plan(
        __file__,
        "EXP-011",
        "c1/c3 gather-scatter index and token permutation cost",
        {"M": [1024, 2048, 4096, 6647], "E": [64, 128], "top_k": [8, 16]},
        ["latency_ms", "percent_of_e2e", "tokens_per_s"],
        "implement mb_dispatch_prep with TD kernels on GPU server",
    )


if __name__ == "__main__":
    raise SystemExit(main())
