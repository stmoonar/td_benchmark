import csv
import subprocess
import sys
from pathlib import Path

from moe_bench.results import JsonlResultSink


def _row(scheme, m, med, passed=True):
    return {
        "scheme": scheme,
        "shape": {"M": m},
        "lat_ms": {"med": med},
        "verify": {"pass": passed},
    }


def test_report_table_generator_reads_jsonl_and_writes_derived_csvs(tmp_path):
    run_dir = tmp_path / "run"
    sink = JsonlResultSink(run_dir / "results.jsonl")
    sink.append(_row("a1", 128, 2.0))
    sink.append(_row("b1", 128, 1.0))
    sink.append(_row("c1", 128, 3.0, passed=False))
    out_dir = tmp_path / "report"

    completed = subprocess.run(
        [
            sys.executable,
            "docs/report/plots/generate_summary_tables.py",
            "--runs",
            str(run_dir),
            "--output-dir",
            str(out_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    latency_rows = list(csv.DictReader((out_dir / "latency_by_m_scheme.csv").open(encoding="utf-8")))
    verify_rows = list(csv.DictReader((out_dir / "verify_summary.csv").open(encoding="utf-8")))
    assert latency_rows[1] == {
        "M": "128",
        "scheme": "b1",
        "median_ms": "1.000",
        "speedup_vs_a1": "2.00",
        "verify": "PASS",
    }
    assert {"verify": "PASS", "count": "2"} in verify_rows
    assert {"verify": "FAIL", "count": "1"} in verify_rows


def test_final_report_draft_contains_plan_outline_sections():
    text = Path("docs/report/final_report.md").read_text(encoding="utf-8")

    for heading in [
        "## Background And Goal",
        "## Scheme Matrix",
        "## Fairness Method",
        "## End-To-End Results",
        "## Bottleneck Breakdown",
        "## Deep-Dive Cases",
        "## Work Timeline And Gains",
        "## Conclusions",
    ]:
        assert heading in text
