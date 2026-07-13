#!/usr/bin/env python3
"""Build a compact b2-vs-c4 comparison from focused investigation runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMES = {"b2", "c4"}


def collect_rows(suite_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for result_path in sorted(suite_dir.rglob("results.jsonl")):
        for line in result_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("scheme") not in SCHEMES:
                continue
            records.append(
                {
                    "run_dir": result_path.parent.name,
                    "point_index": int(row.get("point_index", -1)),
                    "point_values": json.dumps(row.get("point_values", {}), sort_keys=True),
                    "scheme": row["scheme"],
                    "avg_ms": float(row["lat_ms"]["avg"]),
                    "median_ms": float(row["lat_ms"]["med"]),
                    "p95_ms": float(row["lat_ms"]["p95"]),
                    "verify": bool(row.get("verify", {}).get("pass", False)),
                    "world_size": int(row.get("world_size", 0)),
                    "samples": len(row.get("samples_ms", [])),
                }
            )
    return records


def write_csv(records: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_dir", "point_index", "point_values", "scheme", "avg_ms",
        "median_ms", "p95_ms", "verify", "world_size", "samples",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def render_markdown(records: list[dict[str, Any]], components: dict[str, Any] | None = None) -> str:
    grouped: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        key = (record["run_dir"], record["point_index"], record["point_values"])
        grouped[key][record["scheme"]] = record

    lines = [
        "# TD-TP-FP8 vs Chunk Overlap Investigation",
        "",
        "| run | point | values | b2 med ms | c4 med ms | c4 / b2 | verify |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    paired = 0
    for (run_dir, point_index, values), schemes in sorted(grouped.items()):
        b2 = schemes.get("b2")
        c4 = schemes.get("c4")
        if b2 and c4:
            paired += 1
            ratio = c4["median_ms"] / b2["median_ms"] if b2["median_ms"] else float("inf")
            verify = "PASS" if b2["verify"] and c4["verify"] else "FAIL"
            lines.append(
                f"| {run_dir} | {point_index} | `{values}` | {b2['median_ms']:.4f} | "
                f"{c4['median_ms']:.4f} | {ratio:.3f}x | {verify} |"
            )
    if paired == 0:
        lines.append("| n/a | - | no paired b2/c4 rows found | - | - | - | - |")

    lines.extend(
        [
            "",
            "## All focused rows",
            "",
            "| run | point | values | scheme | median ms | p95 ms | samples | verify |",
            "|---|---:|---|---|---:|---:|---:|---|",
        ]
    )
    for record in sorted(
        records,
        key=lambda item: (item["run_dir"], item["point_index"], item["scheme"]),
    ):
        verify = "PASS" if record["verify"] else "FAIL"
        lines.append(
            f"| {record['run_dir']} | {record['point_index']} | `{record['point_values']}` | "
            f"{record['scheme']} | {record['median_ms']:.4f} | {record['p95_ms']:.4f} | "
            f"{record['samples']} | {verify} |"
        )

    if components:
        lines.extend(
            [
                "",
                "## Component benchmarks",
                "",
                "| component | median ms | p95 ms |",
                "|---|---:|---:|",
            ]
        )
        for name, measurement in components.get("measurements", {}).items():
            latency = measurement["lat_ms"]
            lines.append(f"| {name} | {latency['med']:.6f} | {latency['p95']:.6f} |")

    lines.extend(
        [
            "",
            "## Interpretation checklist",
            "",
            "- Compare `shared_experts=0` with `shared_experts=1` to isolate shared-expert scheduling.",
            "- Use the M sweep to distinguish fixed launch/synchronization overhead from throughput limits.",
            "- Use the b2 and c4 chunk sweeps to identify over-fragmentation.",
            "- Use routing sweeps to quantify token-layout and padding sensitivity.",
            "- Inspect torch and Nsight Systems traces before attributing the gap to a single kernel.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", required=True, type=Path)
    args = parser.parse_args()
    suite_dir = args.suite_dir.resolve()
    records = collect_rows(suite_dir)
    analysis_dir = suite_dir / "analysis"
    component_path = analysis_dir / "component_benchmarks.json"
    components = json.loads(component_path.read_text(encoding="utf-8")) if component_path.is_file() else None
    write_csv(records, analysis_dir / "comparison.csv")
    (analysis_dir / "comparison.md").write_text(
        render_markdown(records, components), encoding="utf-8"
    )
    print(f"Focused rows: {len(records)}")
    print(f"Analysis: {analysis_dir}")
    return 0 if records else 2


if __name__ == "__main__":
    raise SystemExit(main())
