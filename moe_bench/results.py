from __future__ import annotations

import glob
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable


class JsonlResultSink:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_results(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_runs(pattern: str | Path) -> Any:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("load_runs requires pandas; install pandas for report/chart analysis.") from exc

    rows: list[dict[str, Any]] = []
    for path in _expand_result_paths(pattern):
        manifest = _read_manifest(path.parent / "manifest.json")
        for row in read_results(path):
            enriched = dict(row)
            enriched["source_path"] = path.as_posix()
            enriched["run_dir"] = path.parent.name
            if manifest:
                enriched["manifest"] = manifest
            rows.append(enriched)
    if not rows:
        return pd.DataFrame()
    return pd.json_normalize(rows)


def generate_summary(rows: Iterable[dict[str, Any]], path: str | Path) -> str:
    materialized = list(rows)
    baseline = next((row for row in materialized if row.get("scheme") == "a1"), None)
    baseline_med = _median_ms(baseline) if baseline else None
    baseline_by_m = _baseline_by_m(materialized)
    verify_counts = _verify_counts(materialized)
    lines = [
        "# MoE Bench Summary",
        "",
        "## Scheme Overview",
        "",
        "| scheme | median_ms | speedup_vs_a1 | verify |",
        "|---|---:|---:|---|",
    ]
    for row in materialized:
        med = _median_ms(row)
        speedup = baseline_med / med if baseline_med is not None and med else None
        verify = _verify_status(row)
        speedup_text = "n/a" if speedup is None else f"{speedup:.2f}x"
        lines.append(f"| {row.get('scheme')} | {med:.3f} | {speedup_text} | {verify} |")
    lines.extend(
        [
            "",
            "## Latency By M And Scheme",
            "",
            "| M | scheme | median_ms | speedup_vs_a1 | verify |",
            "|---:|---|---:|---:|---|",
        ]
    )
    for row in sorted(materialized, key=lambda item: (_shape_m(item), str(item.get("scheme")))):
        med = _median_ms(row)
        baseline_for_m = baseline_by_m.get(_shape_m(row))
        speedup = baseline_for_m / med if baseline_for_m is not None and med else None
        speedup_text = "n/a" if speedup is None else f"{speedup:.2f}x"
        lines.append(f"| {_shape_m(row)} | {row.get('scheme')} | {med:.3f} | {speedup_text} | {_verify_status(row)} |")
    lines.extend(
        [
            "",
            "## Verify Summary",
            "",
            "| verify | count |",
            "|---|---:|",
        ]
    )
    for status, count in sorted(verify_counts.items()):
        lines.append(f"| {status} | {count} |")
    summary = "\n".join(lines) + "\n"
    Path(path).write_text(summary, encoding="utf-8")
    return summary


def write_manifest(run_dir: str | Path, cfg: Any, points: Iterable[Any], argv: list[str] | None, effective_env: dict[str, str] | None = None) -> None:
    path = Path(run_dir) / "manifest.json"
    env_keys = [
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_MAX_CONNECTIONS",
        "NVSHMEM_SYMMETRIC_SIZE",
        "NVSHMEM_REMOTE_TRANSPORT",
        "NVSHMEM_DISABLE_CUDA_VMM",
        "C_INCLUDE_PATH",
        "TRITON_PTXAS_PATH",
        "PYTHONPATH",
    ]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "argv": argv,
        "config": cfg.resolved_dict(),
        "env": {key: os.environ.get(key) for key in env_keys if os.environ.get(key) is not None},
        "configured_env": cfg.env,
        "effective_env": dict(effective_env or {}),
        "cuda_visible_devices": cfg.dist.cuda_visible_devices,
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "dependencies": {
            "torch": _dist_version("torch"),
            "vllm": _dist_version("vllm"),
            "triton": _dist_version("triton"),
            "triton_dist": _dist_version("triton_dist"),
        },
        "sweep_points": [{"index": point.index, "values": point.values} for point in points],
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _median_ms(row: dict[str, Any] | None) -> float | None:
    if row is None:
        return None
    lat = row.get("lat_ms", {})
    if "med" in lat:
        return float(lat["med"])
    if "median" in lat:
        return float(lat["median"])
    if "avg" in lat:
        return float(lat["avg"])
    return None


def _shape_m(row: dict[str, Any]) -> int:
    return int(row.get("shape", {}).get("M", 0))


def _verify_status(row: dict[str, Any]) -> str:
    verify_data = row.get("verify", {})
    return verify_data.get("status") or ("PASS" if verify_data.get("pass") else "FAIL")


def _baseline_by_m(rows: Iterable[dict[str, Any]]) -> dict[int, float]:
    baselines: dict[int, float] = {}
    for row in rows:
        if row.get("scheme") != "a1":
            continue
        med = _median_ms(row)
        if med is not None:
            baselines[_shape_m(row)] = med
    return baselines


def _verify_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = _verify_status(row)
        counts[status] = counts.get(status, 0) + 1
    return counts


def _expand_result_paths(pattern: str | Path) -> list[Path]:
    matches = [Path(item) for item in glob.glob(str(pattern))]
    paths: list[Path] = []
    for match in matches:
        if match.is_dir():
            result_path = match / "results.jsonl"
            if result_path.is_file():
                paths.append(result_path)
        elif match.is_file():
            paths.append(match)
    return sorted(paths)


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _git_sha() -> str | None:
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True)
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _dist_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "missing"
    except Exception:
        return "unknown"
