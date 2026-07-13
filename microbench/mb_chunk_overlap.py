"""Chunk-count sweep for b-group overlap: measures overlap efficiency eta.

eta = (T_a1 - T_b) / T_comm_est
Requires: configs/run_v1.yaml and results/microbench_comm_prims.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def latest_results(root: Path) -> Path:
    runs = sorted(p for p in root.iterdir() if p.is_dir())
    return runs[-1] / "results.jsonl"


def read_med(results: Path, scheme: str) -> float:
    for line in results.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["scheme"] == scheme:
            return row["lat_ms"]["med"]
    raise RuntimeError(f"{scheme} not in {results}")


def comm_ms(comm_jsonl: Path, total_bytes: int, field: str) -> float:
    rows = [json.loads(line) for line in comm_jsonl.read_text(encoding="utf-8").splitlines()]
    best = min(rows, key=lambda r: abs(r["total_bytes"] - total_bytes))
    bw = best[field]
    world = best["world_size"]
    return (total_bytes * (world - 1) / world) / (bw * 1e9) * 1000.0


def main() -> int:
    parser = argparse.ArgumentParser(description="b-group chunk sweep with eta")
    parser.add_argument("--config", default="configs/run_v1.yaml")
    parser.add_argument("--comm-jsonl", default="results/microbench_comm_prims.jsonl")
    parser.add_argument("--output", default="results/microbench_chunk_overlap.jsonl")
    parser.add_argument("--out-root", default="results/mb_chunk_overlap")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    K = 4096

    for M in [1024, 5120]:
        subprocess.run([sys.executable, "-m", "moe_bench.cli", args.config,
                        "--set", "schemes.enabled=[a1]", "--set", f"shape.M={M}",
                        "--set", f"run.output_root={out_root}", "--set", "run.tag=a1_base"], check=True)
        t_a1 = read_med(latest_results(out_root), "a1")

        bytes_bf16 = M * K * 2
        for scheme, n_chunks in [("b1", 2), ("b1", 4), ("b2", 2), ("b2", 4), ("b3", 2), ("b3", 4)]:
            ag_bytes = bytes_bf16
            rs_bytes = bytes_bf16
            t_comm = (comm_ms(Path(args.comm_jsonl), ag_bytes, "ag_bandwidth_gbps")
                      + comm_ms(Path(args.comm_jsonl), rs_bytes, "rs_bandwidth_gbps"))
            subprocess.run([sys.executable, "-m", "moe_bench.cli", args.config,
                            "--set", f"schemes.enabled=[{scheme}]", "--set", f"shape.M={M}",
                            "--set", f"schemes.{scheme}.tunables.n_chunks_gateup={n_chunks}",
                            "--set", f"schemes.{scheme}.tunables.n_chunks_down={n_chunks}",
                            "--set", f"run.output_root={out_root}",
                            "--set", f"run.tag={scheme}_c{n_chunks}"], check=True)
            t_b = read_med(latest_results(out_root), scheme)
            eta = (t_a1 - t_b) / t_comm if t_comm > 0 else 0.0
            row = {
                "kind": "microbench", "script": "mb_chunk_overlap.py", "exp_id": "EXP-016",
                "M": M, "scheme": scheme, "n_chunks": n_chunks,
                "latency_ms": t_b, "a1_latency_ms": t_a1,
                "comm_est_ms": t_comm, "overlap_efficiency_eta": eta,
            }
            rows.append(row)
            print(json.dumps(row))

    with Path(args.output).open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Results written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from common import run_microbench_plan


def main() -> int:
    return run_microbench_plan(
        __file__,
        "EXP-016",
        "b-group chunk count versus overlap efficiency eta",
        {"n_chunks": [1, 2, 4, 8, 16], "scheme": ["b1", "b2", "b3"]},
        ["latency_ms", "overlap_efficiency_eta", "exposed_comm_ms"],
        "implement chunk overlap sweep with eta calculation",
    )


if __name__ == "__main__":
    raise SystemExit(main())
