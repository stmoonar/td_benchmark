from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile


ACTIVE_SCHEMES = {"a1", "a2", "b1", "b2", "b3", "c3", "c4", "c5"}
DISABLED_SCHEMES = {"c1", "c2"}


def validate_suite(suite_dir: Path) -> dict[str, Any]:
    run_dirs = sorted(path.parent for path in suite_dir.rglob("manifest.json"))
    errors: list[str] = []
    run_reports: list[dict[str, Any]] = []
    if not run_dirs:
        errors.append("no manifest.json files found")

    for run_dir in run_dirs:
        manifest = _read_json(run_dir / "manifest.json")
        results_path = run_dir / "results.jsonl"
        rows = _read_jsonl(results_path) if results_path.is_file() else []
        config = manifest.get("config", {})
        enabled = list(config.get("schemes", {}).get("enabled", []))
        nproc = int(config.get("dist", {}).get("nproc", 0))
        repeat = int(config.get("repeat", 0))
        points = list(manifest.get("sweep_points", []))
        expected_rows = len(enabled) * len(points)
        prefix = run_dir.relative_to(suite_dir).as_posix()

        if set(enabled) - ACTIVE_SCHEMES:
            errors.append(f"{prefix}: non-FP8/unknown schemes enabled: {sorted(set(enabled) - ACTIVE_SCHEMES)}")
        if set(enabled) & DISABLED_SCHEMES:
            errors.append(f"{prefix}: BF16 TD schemes must be disabled")
        for env_section in ("env", "configured_env", "effective_env"):
            if "CUDA_DEVICE_MAX_CONNECTIONS" in manifest.get(env_section, {}):
                errors.append(f"{prefix}: CUDA_DEVICE_MAX_CONNECTIONS present in {env_section}")
        if "CUDA_DEVICE_MAX_CONNECTIONS" not in manifest.get("unset_env", []):
            errors.append(f"{prefix}: manifest does not record CUDA_DEVICE_MAX_CONNECTIONS as unset")
        if len(rows) != expected_rows:
            errors.append(f"{prefix}: expected {expected_rows} rank-0 rows, found {len(rows)}")

        key_counts: Counter[tuple[int, str]] = Counter()
        for row in rows:
            scheme = str(row.get("scheme"))
            key_counts[(int(row.get("point_index", -1)), scheme)] += 1
            if scheme not in ACTIVE_SCHEMES:
                errors.append(f"{prefix}: invalid scheme row {scheme}")
            if row.get("rank") != 0 or row.get("world_size") != nproc:
                errors.append(f"{prefix}: {scheme} must be a rank=0/world_size={nproc} aggregate row")
            if row.get("latency_aggregation") != "per_iteration_max_across_ranks":
                errors.append(f"{prefix}: {scheme} lacks per-iteration MAX aggregation")
            if row.get("routing", {}).get("included_in_timing") is not False:
                errors.append(f"{prefix}: {scheme} routing timing contract violated")
            quant = row.get("quantization", {})
            if quant.get("activation", {}).get("group_size") != 128:
                errors.append(f"{prefix}: {scheme} activation is not group128")
            if quant.get("weight", {}).get("block_shape") != [128, 128]:
                errors.append(f"{prefix}: {scheme} weight is not block128x128")
            if row.get("verify", {}).get("vs") != "golden_fp8sim_group128":
                errors.append(f"{prefix}: {scheme} uses a non-group128 golden")
            if not row.get("verify", {}).get("pass", False):
                errors.append(f"{prefix}: {scheme} verification failed")
            actual_samples = len(row.get("samples_ms", []))
            if actual_samples != repeat:
                errors.append(
                    f"{prefix}: {scheme} has wrong sample count "
                    f"(expected {repeat}, found {actual_samples})"
                )

        duplicates = [key for key, count in key_counts.items() if count != 1]
        if duplicates:
            errors.append(f"{prefix}: duplicate/missing aggregate keys: {duplicates[:10]}")
        run_reports.append(
            {
                "run_dir": prefix,
                "enabled": enabled,
                "points": len(points),
                "expected_rows": expected_rows,
                "actual_rows": len(rows),
                "all_verify_pass": bool(rows) and all(row.get("verify", {}).get("pass", False) for row in rows),
            }
        )

    return {"valid": not errors, "errors": errors, "runs": run_reports}


def package_suite(suite_dir: Path, output: Path) -> tuple[Path, str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(suite_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=(Path(suite_dir.name) / path.relative_to(suite_dir)).as_posix())
    digest = _sha256(output)
    checksum_path = Path(str(output) + ".sha256")
    checksum_path.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    return checksum_path, digest


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate rank-0 aggregate benchmark artifacts and create a ZIP")
    parser.add_argument("--suite-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--package-invalid",
        action="store_true",
        help="Create the diagnostic ZIP even if validation fails.",
    )
    args = parser.parse_args()

    suite_dir = args.suite_dir.resolve()
    report = validate_suite(suite_dir)
    report_path = suite_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not report["valid"]:
        for error in report["errors"]:
            print(f"ERROR: {error}")
        print(f"Validation report: {report_path}")
        if not args.package_invalid:
            return 1
        print("WARNING: packaging invalid suite for diagnosis")

    checksum_path, digest = package_suite(suite_dir, args.output.resolve())
    print(f"Packaged runs: {len(report['runs'])} valid={report['valid']}")
    print(f"ZIP: {args.output.resolve()}")
    print(f"SHA256: {digest}")
    print(f"Checksum file: {checksum_path}")
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
