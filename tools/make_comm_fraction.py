"""Generate the comm-fraction table from a sweep results.jsonl + comm_prims jsonl.

Usage: python tools/make_comm_fraction.py <sweep_results.jsonl> <comm_prims.jsonl>
Writes a markdown table to stdout.
"""

import json
import sys


def comm_ms(rows: list[dict], total_bytes: int, field: str) -> float:
    best = min(rows, key=lambda r: abs(r["total_bytes"] - total_bytes))
    world = best["world_size"]
    return (total_bytes * (world - 1) / world) / (best[field] * 1e9) * 1000.0


def main() -> int:
    sweep = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
    comm = [json.loads(line) for line in open(sys.argv[2], encoding="utf-8")]

    med: dict[tuple[int, str], float] = {}
    K = sweep[0]["shape"]["K"]
    for row in sweep:
        key = (row["shape"]["M"], row["scheme"])
        med.setdefault(key, row["lat_ms"]["med"])

    ms_values = sorted({m for (m, _) in med})
    print("# 通信占比表（v1 sweep x EXP-013 带宽折算）\n")
    print("| M | AG+RS bytes (MB) | T_comm_est (ms) | T_a1 (ms) | 通信占比 | b3 vs a1 | c3 vs a1 |")
    print("|---|---|---|---|---|---|---|")
    for m in ms_values:
        ag_bytes = m * K * 2
        rs_bytes = m * K * 2
        t_comm = comm_ms(comm, ag_bytes, "ag_bandwidth_gbps") + comm_ms(comm, rs_bytes, "rs_bandwidth_gbps")
        t_a1 = med.get((m, "a1"))
        if t_a1 is None:
            continue
        b3 = med.get((m, "b3"))
        c3 = med.get((m, "c3"))
        cells = [
            str(m),
            f"{(ag_bytes + rs_bytes) / 1e6:.1f}",
            f"{t_comm:.3f}",
            f"{t_a1:.3f}",
            f"{t_comm / t_a1:.1%}",
            f"{t_a1 / b3:.2f}x" if b3 else "-",
            f"{t_a1 / c3:.2f}x" if c3 else "-",
        ]
        print("| " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
