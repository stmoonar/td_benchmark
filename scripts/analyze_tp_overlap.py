#!/usr/bin/env python3
"""Build a compact b2-vs-c4 comparison from focused investigation runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


SCHEMES = {"b2", "c4"}
CRITICAL_PATH_FIELDS = ("total_us", "pre_us", "gate_us", "middle_us", "down_us")
NSYS_KERNELS = {
    "b2": {
        "fused_moe_kernel_accumulate",
        "fused_moe_kernel",
        "ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL",
    },
    "c4": {
        "fp8_kernel_consumer_ag_group_gemm",
        "fp8_moe_gather_rs_grouped_gemm_kernel",
        "reduce_topk_reduce_scatter_ring_intra_node_kernel",
    },
}


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


def collect_torch_critical_paths(suite_dir: Path) -> list[dict[str, Any]]:
    """Split each torch GPU trace into comparable end-to-end critical-path stages."""
    records: list[dict[str, Any]] = []
    for trace_path in sorted(suite_dir.rglob("profiles/torch/*_rank*.json")):
        stem = trace_path.stem
        scheme = next((code for code in SCHEMES if stem.startswith(f"{code}_rank")), None)
        if scheme is None:
            continue
        rank = int(stem.rsplit("rank", 1)[1])
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        events = sorted(
            (
                event for event in payload.get("traceEvents", [])
                if event.get("cat") in {"kernel", "gpu_memcpy"}
                and "ts" in event and "dur" in event
            ),
            key=lambda event: float(event["ts"]),
        )
        marker = "moe_align_block_size_kernel" if scheme == "b2" else "_quantize_fp8_blockwise_kernel"
        starts = [index for index, event in enumerate(events) if marker in event.get("name", "")]
        for iteration, start_index in enumerate(starts):
            stop_index = starts[iteration + 1] if iteration + 1 < len(starts) else len(events)
            window = events[start_index:stop_index]
            stages = _decompose_trace_window(scheme, window)
            if stages is None:
                continue
            records.append(
                {
                    "run_dir": trace_path.parents[2].name,
                    "scheme": scheme,
                    "rank": rank,
                    "iteration": iteration,
                    "cold": iteration == 0,
                    **stages,
                }
            )
    return records


def _decompose_trace_window(scheme: str, events: list[dict[str, Any]]) -> dict[str, float] | None:
    if not events:
        return None
    if scheme == "b2":
        gate_start_events = [event for event in events if "ncclDevKernel_AllGather" in event.get("name", "")]
        gate_events = [event for event in events if event.get("name") == "fused_moe_kernel_accumulate"]
        down_events = [event for event in events if event.get("name") == "fused_moe_kernel"]
    else:
        gate_events = [event for event in events if event.get("name") == "fp8_kernel_consumer_ag_group_gemm"]
        gate_start_events = gate_events
        down_events = [event for event in events if event.get("name") == "fp8_moe_gather_rs_grouped_gemm_kernel"]
    if not gate_start_events or not gate_events or not down_events:
        return None

    start = float(events[0]["ts"])
    end = max(float(event["ts"]) + float(event["dur"]) for event in events)
    gate_start = float(gate_start_events[0]["ts"])
    gate_end = max(float(event["ts"]) + float(event["dur"]) for event in gate_events)
    down_start = float(down_events[0]["ts"])
    return {
        "total_us": end - start,
        "pre_us": gate_start - start,
        "gate_us": gate_end - gate_start,
        "middle_us": down_start - gate_end,
        "down_us": end - down_start,
    }


def summarize_torch_critical_paths(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for scheme in sorted(SCHEMES):
        scheme_rows = [record for record in records if record["scheme"] == scheme]
        steady_rows = [record for record in scheme_rows if not record["cold"]] or scheme_rows
        if not steady_rows:
            continue
        summary[scheme] = {
            field: median(float(record[field]) for record in steady_rows)
            for field in CRITICAL_PATH_FIELDS
        }
        summary[scheme]["observations"] = len(steady_rows)
    return summary


def resource_limited_blocks_per_sm(
    *,
    max_blocks_per_sm: int,
    max_warps_per_sm: int,
    max_registers_per_sm: int,
    max_smem_per_sm: int,
    block_threads: int,
    registers_per_thread: int,
    shared_memory: int,
) -> int:
    limits = [max_blocks_per_sm, max_warps_per_sm // math.ceil(block_threads / 32)]
    if registers_per_thread:
        limits.append(max_registers_per_sm // (registers_per_thread * block_threads))
    if shared_memory:
        limits.append(max_smem_per_sm // shared_memory)
    return max(1, min(limits))


def collect_nsys_kernel_resources(suite_dir: Path) -> dict[str, Any]:
    output: dict[str, Any] = {"gpu": {}, "kernels": []}
    for scheme, kernel_names in NSYS_KERNELS.items():
        database = suite_dir / "profiles" / "nsys" / f"{scheme}_steady_state.sqlite"
        if not database.is_file():
            continue
        with sqlite3.connect(database) as connection:
            gpu_row = connection.execute(
                """
                SELECT name, smCount, maxRegistersPerSm, maxShmemPerSm,
                       maxWarpsPerSm, maxBlocksPerSm, computeMajor, computeMinor
                FROM TARGET_INFO_GPU LIMIT 1
                """
            ).fetchone()
            if gpu_row is None:
                continue
            (
                gpu_name, sm_count, max_registers, max_smem, max_warps,
                max_blocks, compute_major, compute_minor,
            ) = gpu_row
            output["gpu"] = {
                "name": gpu_name,
                "sm_count": sm_count,
                "max_registers_per_sm": max_registers,
                "max_smem_per_sm": max_smem,
                "max_warps_per_sm": max_warps,
                "max_blocks_per_sm": max_blocks,
                "compute_capability": f"{compute_major}.{compute_minor}",
            }
            placeholders = ",".join("?" for _ in kernel_names)
            query = f"""
                SELECT strings.value, kernels.registersPerThread, kernels.gridX,
                       kernels.blockX, kernels.staticSharedMemory,
                       kernels.dynamicSharedMemory,
                       AVG(kernels.end - kernels.start) / 1000.0, COUNT(*)
                FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernels
                JOIN StringIds AS strings ON strings.id = kernels.shortName
                WHERE strings.value IN ({placeholders})
                GROUP BY strings.value, kernels.registersPerThread, kernels.gridX,
                         kernels.blockX, kernels.staticSharedMemory,
                         kernels.dynamicSharedMemory
                ORDER BY strings.value
            """
            for row in connection.execute(query, sorted(kernel_names)):
                (
                    name, registers, grid_x, block_x, static_smem,
                    dynamic_smem, average_us, instances,
                ) = row
                shared_memory = int(static_smem) + int(dynamic_smem)
                active_blocks = resource_limited_blocks_per_sm(
                    max_blocks_per_sm=int(max_blocks),
                    max_warps_per_sm=int(max_warps),
                    max_registers_per_sm=int(max_registers),
                    max_smem_per_sm=int(max_smem),
                    block_threads=int(block_x),
                    registers_per_thread=int(registers),
                    shared_memory=shared_memory,
                )
                output["kernels"].append(
                    {
                        "scheme": scheme,
                        "name": name,
                        "average_us": average_us,
                        "instances": instances,
                        "grid_x": grid_x,
                        "block_threads": block_x,
                        "registers_per_thread": registers,
                        "static_smem_bytes": static_smem,
                        "dynamic_smem_bytes": dynamic_smem,
                        "resource_limited_blocks_per_sm": active_blocks,
                        "launch_waves": grid_x / (sm_count * active_blocks),
                    }
                )
    return output


def render_markdown(
    records: list[dict[str, Any]],
    components: dict[str, Any] | None = None,
    critical_paths: dict[str, dict[str, float]] | None = None,
    nsys_resources: dict[str, Any] | None = None,
) -> str:
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
        wait_comparison = components.get("consumer_wait_comparison")
        if wait_comparison:
            lines.extend(
                [
                    "",
                    "Ready/no-wait output parity: "
                    f"max_abs={wait_comparison['max_abs']:.6g}. "
                    f"Removing `dl.wait` changes the median by "
                    f"{wait_comparison['wait_cost_ms']:+.6f} ms "
                    f"({wait_comparison['wait_cost_percent_of_ready']:+.2f}% of ready-wait latency).",
                ]
            )

    if critical_paths:
        lines.extend(
            [
                "",
                "## Torch profile steady critical path",
                "",
                "Cold profiler iteration is excluded. Values are medians across ranks and remaining iterations.",
                "",
                "| scheme | total ms | pre ms | gate-up ms | activation/layout ms | down+RS ms | observations |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for scheme in sorted(critical_paths):
            item = critical_paths[scheme]
            lines.append(
                f"| {scheme} | {item['total_us'] / 1000:.4f} | {item['pre_us'] / 1000:.4f} | "
                f"{item['gate_us'] / 1000:.4f} | {item['middle_us'] / 1000:.4f} | "
                f"{item['down_us'] / 1000:.4f} | {int(item['observations'])} |"
            )
        if "b2" in critical_paths and "c4" in critical_paths:
            b2 = critical_paths["b2"]
            c4 = critical_paths["c4"]
            lines.append(
                f"| c4 - b2 | {(c4['total_us'] - b2['total_us']) / 1000:+.4f} | "
                f"{(c4['pre_us'] - b2['pre_us']) / 1000:+.4f} | "
                f"{(c4['gate_us'] - b2['gate_us']) / 1000:+.4f} | "
                f"{(c4['middle_us'] - b2['middle_us']) / 1000:+.4f} | "
                f"{(c4['down_us'] - b2['down_us']) / 1000:+.4f} | - |"
            )

    if nsys_resources and nsys_resources.get("kernels"):
        gpu = nsys_resources.get("gpu", {})
        lines.extend(
            [
                "",
                "## Nsight launch resources",
                "",
                f"GPU: {gpu.get('name', 'unknown')}, CC {gpu.get('compute_capability', 'unknown')}, "
                f"SMs {gpu.get('sm_count', 'unknown')}, registers/SM {gpu.get('max_registers_per_sm', 'unknown')}, "
                f"shared-memory/SM {gpu.get('max_smem_per_sm', 'unknown')} bytes.",
                "",
                "The active-block estimate is the minimum imposed by block, warp, register, and shared-memory limits.",
                "",
                "| scheme | kernel | avg us | grid | threads | regs/thread | dynamic smem | blocks/SM | waves |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in sorted(nsys_resources["kernels"], key=lambda value: (value["scheme"], value["name"])):
            lines.append(
                f"| {item['scheme']} | `{item['name']}` | {item['average_us']:.1f} | "
                f"{item['grid_x']} | {item['block_threads']} | {item['registers_per_thread']} | "
                f"{item['dynamic_smem_bytes']} | {item['resource_limited_blocks_per_sm']} | "
                f"{item['launch_waves']:.2f} |"
            )

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
    critical_path_records = collect_torch_critical_paths(suite_dir)
    critical_paths = summarize_torch_critical_paths(critical_path_records)
    nsys_resources = collect_nsys_kernel_resources(suite_dir)
    write_csv(records, analysis_dir / "comparison.csv")
    (analysis_dir / "torch_critical_path.json").write_text(
        json.dumps({"summary": critical_paths, "iterations": critical_path_records}, indent=2),
        encoding="utf-8",
    )
    (analysis_dir / "nsys_kernel_resources.json").write_text(
        json.dumps(nsys_resources, indent=2), encoding="utf-8"
    )
    (analysis_dir / "comparison.md").write_text(
        render_markdown(records, components, critical_paths, nsys_resources), encoding="utf-8"
    )
    print(f"Focused rows: {len(records)}")
    print(f"Analysis: {analysis_dir}")
    return 0 if records else 2


if __name__ == "__main__":
    raise SystemExit(main())
