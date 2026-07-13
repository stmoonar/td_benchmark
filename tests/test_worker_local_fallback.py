import json
from pathlib import Path

from moe_bench.cli import main as cli_main


def test_cli_non_dry_run_writes_server_verify_result_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(Path.cwd())
    output_root = tmp_path / "results"

    exit_code = cli_main(
        [
            "configs/smoke.yaml",
            "--set",
            f"run.output_root={output_root.as_posix()}",
            "--set",
            "run.tag=worker_local",
            "--set",
            "schemes.enabled=[a1,b1,c3]",
            "--set",
            "log.diagnostics=true",
        ]
    )

    assert exit_code == 0
    run_dirs = list(output_root.glob("*_worker_local"))
    assert len(run_dirs) == 1
    results_path = run_dirs[0] / "results.jsonl"
    summary_path = run_dirs[0] / "summary.md"
    resolved_config_path = run_dirs[0] / "config.resolved.yaml"
    manifest = json.loads((run_dirs[0] / "manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]

    assert resolved_config_path.is_file()
    assert (run_dirs[0] / "profiles").is_dir()
    assert (run_dirs[0] / "profiles" / "torch").is_dir()
    assert (run_dirs[0] / "profiles" / "intra").is_dir()
    assert (run_dirs[0] / "profiles" / "ncu").is_dir()
    assert [row["scheme"] for row in rows] == ["a1", "b1", "c3"]
    assert all(row["verify"]["status"] == "SERVER-VERIFY" for row in rows)
    assert all(row["data"]["hidden_local_shape"] == [256, 64] for row in rows)
    assert all(row["data"]["golden_bf16_shape"] == [256, 64] for row in rows)
    assert rows[0]["scheme_meta"]["name"] == "vLLM-TP-Serial"
    assert manifest["effective_env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert manifest["effective_env"]["TRITON_PTXAS_PATH"] == "/usr/local/cuda/bin/ptxas"
    assert summary_path.is_file()
    log_text = (run_dirs[0] / "logs" / "rank0.log").read_text(encoding="utf-8")
    assert "[rank0][INFO]" in log_text
    assert "point=0" in log_text
    assert "hidden_local=[256, 64]" in log_text
    assert "routing_per_owner_counts=" in log_text
