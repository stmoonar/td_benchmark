import json
from pathlib import Path

from scripts.package_server_results import package_suite, validate_suite


def _write_valid_run(suite_dir: Path) -> None:
    run_dir = suite_dir / "run"
    run_dir.mkdir(parents=True)
    manifest = {
        "config": {
            "repeat": 2,
            "dist": {"nproc": 4},
            "schemes": {"enabled": ["a1"]},
        },
        "env": {},
        "configured_env": {},
        "effective_env": {},
        "unset_env": ["CUDA_DEVICE_MAX_CONNECTIONS"],
        "sweep_points": [{"index": 0, "values": {}}],
    }
    row = {
        "point_index": 0,
        "scheme": "a1",
        "rank": 0,
        "world_size": 4,
        "latency_aggregation": "per_iteration_max_across_ranks",
        "routing": {"included_in_timing": False},
        "quantization": {
            "activation": {"group_size": 128},
            "weight": {"block_shape": [128, 128]},
        },
        "verify": {"pass": True, "vs": "golden_fp8sim_group128"},
        "samples_ms": [1.0, 1.1],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_validate_and_package_server_suite(tmp_path):
    suite_dir = tmp_path / "suite"
    _write_valid_run(suite_dir)

    report = validate_suite(suite_dir)
    checksum_path, digest = package_suite(suite_dir, tmp_path / "suite.zip")

    assert report["valid"] is True
    assert report["runs"][0]["actual_rows"] == 1
    assert (tmp_path / "suite.zip").is_file()
    assert checksum_path.read_text(encoding="utf-8").startswith(digest)


def test_validation_rejects_duplicate_rank_rows(tmp_path):
    suite_dir = tmp_path / "suite"
    _write_valid_run(suite_dir)
    results = suite_dir / "run" / "results.jsonl"
    results.write_text(results.read_text(encoding="utf-8") * 2, encoding="utf-8")

    report = validate_suite(suite_dir)

    assert report["valid"] is False
    assert any("expected 1 rank-0 rows, found 2" in error for error in report["errors"])
