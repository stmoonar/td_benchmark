import json

import pytest

from moe_bench.results import JsonlResultSink, generate_summary, load_runs, read_results
from moe_bench.results import write_manifest
from moe_bench.config import expand_sweep, load_config


def _row(scheme, med, passed=True):
    return {
        "run_id": "run",
        "ts": "2026-07-04T00:00:00Z",
        "scheme": scheme,
        "shape": {"M": 128, "K": 64, "E": 8, "top_k": 2, "n_gateup": 32, "n_down": 16, "shared": 1},
        "routing": {"kind": "none"},
        "tunables": {},
        "lat_ms": {"avg": med, "med": med, "min": med, "p95": med, "std": 0.0},
        "verify": {"pass": passed},
    }


def test_jsonl_sink_appends_and_reads_rows(tmp_path):
    path = tmp_path / "results.jsonl"
    sink = JsonlResultSink(path)

    sink.append(_row("a1", 2.0))
    sink.append(_row("b1", 1.0))

    assert [row["scheme"] for row in read_results(path)] == ["a1", "b1"]
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(raw_lines[0])["lat_ms"]["med"] == 2.0


def test_generate_summary_writes_speedup_against_a1_and_verify_status(tmp_path):
    rows = [_row("a1", 2.0), _row("b1", 1.0), _row("c1", 4.0, passed=False)]

    summary = generate_summary(rows, tmp_path / "summary.md")

    assert "| a1 | 2.000 | 1.00x | PASS |" in summary
    assert "| b1 | 1.000 | 2.00x | PASS |" in summary
    assert "| c1 | 4.000 | 0.50x | FAIL |" in summary
    assert (tmp_path / "summary.md").read_text(encoding="utf-8") == summary


def test_generate_summary_preserves_server_verify_status(tmp_path):
    row = _row("a1", 0.0, passed=False)
    row["verify"] = {"status": "SERVER-VERIFY", "pass": False}

    summary = generate_summary([row], tmp_path / "summary.md")

    assert "| a1 | 0.000 | n/a | SERVER-VERIFY |" in summary


def test_generate_summary_includes_shape_matrix_and_verify_rollup(tmp_path):
    a1_128 = _row("a1", 2.0)
    b1_128 = _row("b1", 1.0)
    a1_512 = _row("a1", 8.0)
    b1_512 = _row("b1", 4.0, passed=False)
    a1_512["shape"]["M"] = 512
    b1_512["shape"]["M"] = 512
    b1_512["verify"] = {"status": "SERVER-VERIFY", "pass": False}

    summary = generate_summary([a1_128, b1_128, a1_512, b1_512], tmp_path / "summary.md")

    assert "## Latency By M And Scheme" in summary
    assert "| M | scheme | median_ms | speedup_vs_a1 | verify |" in summary
    assert "| 128 | b1 | 1.000 | 2.00x | PASS |" in summary
    assert "| 512 | b1 | 4.000 | 2.00x | SERVER-VERIFY |" in summary
    assert "## Verify Summary" in summary
    assert "| PASS | 3 |" in summary
    assert "| SERVER-VERIFY | 1 |" in summary


def test_generate_summary_does_not_mix_baselines_across_routing_points(tmp_path):
    a1_none = _row("a1", 2.0)
    b1_none = _row("b1", 1.0)
    a1_zipf = _row("a1", 4.0)
    b1_zipf = _row("b1", 2.0)
    for row in [a1_none, b1_none]:
        row["point_index"] = 0
        row["routing"]["kind"] = "none"
    for row in [a1_zipf, b1_zipf]:
        row["point_index"] = 1
        row["routing"]["kind"] = "zipf"

    summary = generate_summary([a1_none, b1_none, a1_zipf, b1_zipf], tmp_path / "summary.md")

    assert summary.count("| b1 | 1.000 | 2.00x | PASS |") == 2
    assert summary.count("| b1 | 2.000 | 2.00x | PASS |") == 2
    assert "| b1 | 1.000 | 4.00x | PASS |" not in summary


def test_write_manifest_captures_runtime_env_and_resolved_config(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    cfg = load_config("configs/smoke.yaml", ["run.tag=manifest"])
    points = expand_sweep(cfg)

    effective_env = {"CUDA_VISIBLE_DEVICES": "0", **cfg.env}
    write_manifest(tmp_path, cfg, points, argv=["configs/smoke.yaml"], effective_env=effective_env)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["tag"] == "manifest"
    assert manifest["runtime"]["python"]
    assert manifest["runtime"]["platform"]
    assert manifest["env"]["CUDA_VISIBLE_DEVICES"] == "7"
    assert manifest["effective_env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert manifest["effective_env"]["TRITON_PTXAS_PATH"] == "/usr/local/cuda/bin/ptxas"
    assert manifest["dependencies"]["vllm"] in {"missing", "unknown"}


def test_load_runs_normalizes_result_jsonl_from_run_directories(tmp_path):
    pytest.importorskip("pandas")
    first = tmp_path / "run_a"
    second = tmp_path / "run_b"
    JsonlResultSink(first / "results.jsonl").append(_row("a1", 1.0))
    JsonlResultSink(second / "results.jsonl").append(_row("c3", 0.5))

    frame = load_runs(str(tmp_path / "run_*"))
    records = frame.sort_values("scheme").to_dict("records")

    assert [record["scheme"] for record in records] == ["a1", "c3"]
    assert records[0]["run_dir"] == "run_a"
    assert records[0]["source_path"].endswith("run_a/results.jsonl")
    assert records[0]["lat_ms.med"] == 1.0
    assert records[1]["shape.M"] == 128


def test_load_runs_accepts_result_file_globs(tmp_path):
    pytest.importorskip("pandas")
    JsonlResultSink(tmp_path / "r1.jsonl").append(_row("a1", 1.0))
    JsonlResultSink(tmp_path / "r2.jsonl").append(_row("b1", 2.0))

    frame = load_runs(str(tmp_path / "*.jsonl"))

    assert set(frame["scheme"]) == {"a1", "b1"}
