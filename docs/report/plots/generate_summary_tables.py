from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moe_bench.results import load_runs


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate report tables from MoE Bench JSONL results.")
    parser.add_argument("--runs", required=True, help="Run directory glob or result JSONL glob.")
    parser.add_argument("--output-dir", default="docs/report/generated", help="Directory for derived CSV tables.")
    args = parser.parse_args()

    frame = load_runs(args.runs)
    if frame.empty:
        raise SystemExit(f"no result rows matched {args.runs!r}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_latency_table(frame, output_dir / "latency_by_m_scheme.csv")
    _write_verify_summary(frame, output_dir / "verify_summary.csv")
    return 0


def _write_latency_table(frame, path: Path) -> None:
    rows = []
    for _, row in frame.sort_values(["shape.M", "scheme"]).iterrows():
        median = _median_ms(row)
        baseline = _baseline_for_row(frame, row)
        speedup = baseline / median if baseline and median else None
        rows.append(
            {
                "M": str(int(row["shape.M"])),
                "scheme": str(row["scheme"]),
                "median_ms": f"{median:.3f}",
                "speedup_vs_a1": "n/a" if speedup is None else f"{speedup:.2f}",
                "verify": _verify_status(row),
            }
        )
    _write_csv(path, ["M", "scheme", "median_ms", "speedup_vs_a1", "verify"], rows)


def _write_verify_summary(frame, path: Path) -> None:
    counts: dict[str, int] = {}
    for _, row in frame.iterrows():
        status = _verify_status(row)
        counts[status] = counts.get(status, 0) + 1
    rows = [{"verify": status, "count": str(counts[status])} for status in sorted(counts)]
    _write_csv(path, ["verify", "count"], rows)


def _median_ms(row) -> float:
    for column in ["lat_ms.med", "lat_ms.median", "lat_ms.avg"]:
        if column in row and row[column] == row[column]:
            return float(row[column])
    raise ValueError("result row is missing lat_ms.med, lat_ms.median, and lat_ms.avg")


def _baseline_for_row(frame, row) -> float | None:
    candidates = frame[frame["scheme"] == "a1"]
    if "point_index" in frame.columns and "point_index" in row:
        candidates = candidates[candidates["point_index"] == row["point_index"]]
    else:
        candidates = candidates[candidates["shape.M"] == int(row["shape.M"])]
        if "routing.kind" in frame.columns and "routing.kind" in row:
            candidates = candidates[candidates["routing.kind"] == row["routing.kind"]]
    if candidates.empty:
        return None
    return _median_ms(candidates.iloc[0])


def _verify_status(row) -> str:
    if "verify.status" in row and isinstance(row["verify.status"], str):
        return row["verify.status"]
    return "PASS" if bool(row.get("verify.pass", False)) else "FAIL"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
