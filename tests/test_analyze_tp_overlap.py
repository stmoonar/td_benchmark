from __future__ import annotations

import json

from scripts.analyze_tp_overlap import (
    collect_rows,
    collect_torch_critical_paths,
    resource_limited_blocks_per_sm,
    render_markdown,
    summarize_torch_critical_paths,
    write_csv,
)


def _row(scheme: str, median: float) -> dict:
    return {
        "point_index": 0,
        "point_values": {"shape.M": 5120},
        "scheme": scheme,
        "lat_ms": {"avg": median, "med": median, "p95": median},
        "verify": {"pass": True},
        "world_size": 4,
        "samples_ms": [median] * 3,
    }


def test_collect_and_render_paired_comparison(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [_row("b2", 2.0), _row("c4", 3.0), _row("a1", 9.0)]
    (run_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    records = collect_rows(tmp_path)
    markdown = render_markdown(records)
    csv_path = tmp_path / "comparison.csv"
    write_csv(records, csv_path)

    assert [record["scheme"] for record in records] == ["b2", "c4"]
    assert "1.500x" in markdown
    assert "a1" not in markdown
    assert "median_ms" in csv_path.read_text(encoding="utf-8")


def test_collects_and_summarizes_torch_critical_path(tmp_path) -> None:
    trace_dir = tmp_path / "run" / "profiles" / "torch"
    trace_dir.mkdir(parents=True)
    events = []
    for offset in (0.0, 1000.0):
        events.extend(
            [
                {"cat": "kernel", "name": "moe_align_block_size_kernel", "ts": offset, "dur": 10.0},
                {"cat": "kernel", "name": "ncclDevKernel_AllGather", "ts": offset + 100.0, "dur": 100.0},
                {"cat": "kernel", "name": "fused_moe_kernel_accumulate", "ts": offset + 150.0, "dur": 300.0},
                {"cat": "kernel", "name": "fused_moe_kernel", "ts": offset + 500.0, "dur": 200.0},
                {"cat": "kernel", "name": "ncclDevKernel_ReduceScatter", "ts": offset + 600.0, "dur": 200.0},
            ]
        )
    (trace_dir / "b2_rank0.json").write_text(
        json.dumps({"traceEvents": events}), encoding="utf-8"
    )

    rows = collect_torch_critical_paths(tmp_path)
    summary = summarize_torch_critical_paths(rows)
    markdown = render_markdown([], critical_paths=summary)

    assert len(rows) == 2
    assert summary["b2"]["total_us"] == 800.0
    assert summary["b2"]["gate_us"] == 350.0
    assert "Torch profile steady critical path" in markdown


def test_resource_limited_blocks_matches_blackwell_trace_shapes() -> None:
    common = {
        "max_blocks_per_sm": 24,
        "max_warps_per_sm": 48,
        "max_registers_per_sm": 65536,
        "max_smem_per_sm": 102400,
        "registers_per_thread": 255,
    }

    c4_blocks = resource_limited_blocks_per_sm(
        **common, block_threads=256, shared_memory=98304
    )
    b2_blocks = resource_limited_blocks_per_sm(
        **common, block_threads=128, shared_memory=49664
    )

    assert c4_blocks == 1
    assert b2_blocks == 2


def test_render_includes_ready_no_wait_comparison() -> None:
    components = {
        "measurements": {},
        "consumer_wait_comparison": {
            "max_abs": 0.0,
            "wait_cost_ms": 0.0125,
            "wait_cost_percent_of_ready": 0.5,
        },
    }

    markdown = render_markdown([], components=components)

    assert "Ready/no-wait output parity" in markdown
    assert "+0.012500 ms" in markdown
