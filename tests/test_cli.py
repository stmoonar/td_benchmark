from pathlib import Path

from moe_bench.cli import build_worker_command, dry_run_lines, env_for_run
from moe_bench.config import expand_sweep, load_config


def test_build_worker_command_uses_torchrun_module_worker_and_point_json(tmp_path):
    cfg = load_config("configs/smoke.yaml", ["dist.nproc=2", "dist.cuda_visible_devices=0,1", "dist.master_port=29991"])
    point = expand_sweep(cfg)[0]

    command = build_worker_command(cfg, point, tmp_path)

    assert command[:4] == [
        "torchrun",
        "--nproc_per_node=2",
        "--master_port=29991",
        "-m",
    ]
    assert "moe_bench.worker" in command
    assert "--run-dir" in command
    assert str(tmp_path) in command
    assert "--point-json" in command


def test_dry_run_lines_include_injected_env_and_command(tmp_path):
    cfg = load_config(
        "configs/smoke.yaml",
        ["run.tag=dry", "env.CUDA_DEVICE_MAX_CONNECTIONS=1", "dist.nproc=2", "dist.cuda_visible_devices=4,5"],
    )

    lines = dry_run_lines(cfg, tmp_path)

    text = "\n".join(lines)
    assert "run_dir=" in text
    assert "CUDA_VISIBLE_DEVICES=4,5" in text
    assert "CUDA_DEVICE_MAX_CONNECTIONS=1" in text
    assert "torchrun --nproc_per_node=" in text
    assert "moe_bench.worker" in text
    assert Path(tmp_path).name in text


def test_env_for_run_disables_libuv_on_windows_torchrun():
    cfg = load_config("configs/smoke.yaml")

    env = env_for_run(cfg)

    assert env["USE_LIBUV"] == "0"
