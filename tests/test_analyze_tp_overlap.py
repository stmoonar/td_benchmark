from __future__ import annotations

import json

from scripts.analyze_tp_overlap import collect_rows, render_markdown, write_csv


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
