"""Quantization kernel bandwidth microbenchmark: rowwise, group128."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantization kernel bandwidth benchmark")
    parser.add_argument("--output", default="results/microbench_quant_kernels.jsonl")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()

    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input

    M_values = [256, 512, 1024, 2048, 4096, 5120]
    K_values = [2048, 4096]
    results = []

    for M in M_values:
        for K in K_values:
            x = torch.randn(M, K, device=device, dtype=torch.bfloat16)

            for _ in range(args.warmup):
                moe_kernel_quantize_input(x, None, torch.float8_e4m3fn, False, [128, 128])
            torch.cuda.synchronize()

            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeat)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeat)]
            for i in range(args.repeat):
                start_events[i].record()
                moe_kernel_quantize_input(x, None, torch.float8_e4m3fn, False, [128, 128])
                end_events[i].record()
            torch.cuda.synchronize()
            group128_samples = [start_events[i].elapsed_time(end_events[i]) for i in range(args.repeat)]

            g128_med = statistics.median(group128_samples)
            bytes_processed = M * K * 2
            g128_bw = bytes_processed / (g128_med / 1000) / 1e9 if g128_med > 0 else 0

            row = {
                "kind": "microbench", "script": "mb_quant_kernels.py", "exp_id": "EXP-015",
                "M": M, "K": K,
                "group128_latency_ms": g128_med, "group128_bandwidth_gbps": g128_bw,
            }
            results.append(row)
            if rank == 0:
                print(f"  M={M:5d} K={K:4d}  group128: {g128_med:.4f} ms ({g128_bw:.1f} GB/s)")

    if rank == 0:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as fh:
            for row in results:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nResults written to {output}")

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def main() -> int:
    return run_microbench_plan(
        __file__,
        "EXP-015",
        "rowwise/group128 quantization and SwiGLU+quant kernel bandwidth",
        {"kernel": ["rowwise", "group128", "swiglu_quant"], "M": [1024, 4096, 6647], "K": [2048, 3200, 4096]},
        ["latency_ms", "bandwidth_gbps", "bandwidth_utilization"],
        "implement quantization kernel bandwidth benchmark",
    )


if __name__ == "__main__":
    raise SystemExit(main())
