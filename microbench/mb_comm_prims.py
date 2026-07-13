"""Communication primitives microbenchmark: NCCL AG/RS bandwidth vs message size."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist


def main() -> int:
    parser = argparse.ArgumentParser(description="Communication primitive bandwidth sweep")
    parser.add_argument("--output", default="results/microbench_comm_prims.jsonl")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    sizes_bytes = [1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864, 268435456]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    results = []

    for total_bytes in sizes_bytes:
        n_elements = total_bytes // 2  # bf16 = 2 bytes
        local_elements = n_elements // world_size

        local_tensor = torch.randn(local_elements, device=device, dtype=torch.bfloat16)
        full_tensor = torch.empty(n_elements, device=device, dtype=torch.bfloat16)

        # AllGather
        dist.barrier()
        for _ in range(args.warmup):
            dist.all_gather_into_tensor(full_tensor, local_tensor)
        torch.cuda.synchronize()
        dist.barrier()

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeat)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeat)]
        for i in range(args.repeat):
            start_events[i].record()
            dist.all_gather_into_tensor(full_tensor, local_tensor)
            end_events[i].record()
        torch.cuda.synchronize()
        ag_samples = [start_events[i].elapsed_time(end_events[i]) for i in range(args.repeat)]

        # ReduceScatter
        dist.barrier()
        for _ in range(args.warmup):
            dist.reduce_scatter_tensor(local_tensor, full_tensor)
        torch.cuda.synchronize()
        dist.barrier()

        for i in range(args.repeat):
            start_events[i].record()
            dist.reduce_scatter_tensor(local_tensor, full_tensor)
            end_events[i].record()
        torch.cuda.synchronize()
        rs_samples = [start_events[i].elapsed_time(end_events[i]) for i in range(args.repeat)]

        ag_med = statistics.median(ag_samples)
        rs_med = statistics.median(rs_samples)
        ag_bw = (total_bytes * (world_size - 1) / world_size) / (ag_med / 1000) / 1e9 if ag_med > 0 else 0
        rs_bw = (total_bytes * (world_size - 1) / world_size) / (rs_med / 1000) / 1e9 if rs_med > 0 else 0

        row = {
            "kind": "microbench",
            "script": "mb_comm_prims.py",
            "exp_id": "EXP-013",
            "total_bytes": total_bytes,
            "dtype": "bf16",
            "world_size": world_size,
            "ag_latency_ms": ag_med,
            "ag_bandwidth_gbps": ag_bw,
            "rs_latency_ms": rs_med,
            "rs_bandwidth_gbps": rs_bw,
        }
        results.append(row)
        if rank == 0:
            print(f"  {total_bytes:>12d} B  AG: {ag_med:.3f} ms ({ag_bw:.2f} GB/s)  RS: {rs_med:.3f} ms ({rs_bw:.2f} GB/s)")

    if rank == 0:
        with output.open("w", encoding="utf-8") as fh:
            for row in results:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nResults written to {output}")

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
