import json
import subprocess
import sys
from pathlib import Path


def test_all_microbench_scripts_support_dry_run_json_plan():
    exp_ids = []
    for path in sorted(Path("microbench").glob("mb_*.py")):
        completed = subprocess.run(
            [sys.executable, str(path), "--dry-run"],
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, completed.stderr
        payload = json.loads(completed.stdout)
        assert payload["kind"] == "microbench"
        assert payload["script"] == path.name
        assert payload["server_verify"] is True
        assert payload["sweep"]
        assert payload["exp_id"].startswith("EXP-")
        assert payload["result_schema"]["kind"] == "microbench"
        assert {"metric", "unit", "aggregation"} <= set(payload["result_schema"])
        assert payload["metrics"]
        assert payload["recommended_command"][0] == "torchrun"
        exp_ids.append(payload["exp_id"])

    assert sorted(exp_ids) == [f"EXP-{index:03d}" for index in range(10, 17)]


def test_microbench_non_dry_run_writes_jsonl_rows(tmp_path):
    output = tmp_path / "microbench.jsonl"
    completed = subprocess.run(
        [
            sys.executable,
            "microbench/mb_group_gemm.py",
            "--output",
            str(output),
            "--warmup",
            "1",
            "--repeat",
            "2",
            "--max-cases",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["kind"] == "microbench"
    assert rows[0]["status"] == "SERVER-VERIFY"
    assert len(rows[0]["samples_ms"]) == 2
    assert rows[0]["metrics"]["latency_ms"] >= 0
