from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

import yaml

from .config import RunCfg, SweepPoint, expand_sweep, load_config, run_cfg_from_dict
from .results import generate_summary, read_results, write_manifest
from . import worker


_UNSET_FOR_RUN = {"CUDA_DEVICE_MAX_CONNECTIONS"}


def build_worker_command(cfg: RunCfg, point: SweepPoint, run_dir: str | Path) -> list[str]:
    point_json = json.dumps({"index": point.index, "values": point.values, "config": point.cfg.resolved_dict()}, ensure_ascii=False)
    return [
        "torchrun",
        f"--nproc_per_node={cfg.dist.nproc}",
        f"--master_port={cfg.dist.master_port}",
        "-m",
        "moe_bench.worker",
        "--run-dir",
        str(run_dir),
        "--point-json",
        point_json,
    ]


def env_for_run(cfg: RunCfg) -> dict[str, str]:
    env = dict(cfg.env)
    env["CUDA_VISIBLE_DEVICES"] = cfg.dist.cuda_visible_devices
    env.setdefault("USE_LIBUV", "0")
    new_root = str(Path(__file__).resolve().parent.parent)
    existing = os.environ.get("PYTHONPATH", "")
    parts = [new_root] + ([existing] if existing else [])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def dry_run_lines(cfg: RunCfg, run_dir: str | Path | None = None) -> list[str]:
    target_dir = Path(run_dir) if run_dir is not None else make_run_dir(cfg, create=False)
    lines = [f"run_dir={target_dir}"]
    lines.append("unset: " + ",".join(sorted(_UNSET_FOR_RUN)))
    env = env_for_run(cfg)
    lines.append("env:")
    for key in sorted(env):
        lines.append(f"  {key}={env[key]}")
    lines.append("commands:")
    for point in expand_sweep(cfg):
        lines.append("  " + _quote_command(build_worker_command(cfg, point, target_dir)))
    return lines


def make_run_dir(cfg: RunCfg, create: bool = True) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.output_root) / f"{stamp}_{cfg.tag}"
    if create:
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        for profile_kind in ["torch", "intra", "ncu"]:
            (run_dir / "profiles" / profile_kind).mkdir(parents=True, exist_ok=True)
    return run_dir


def write_resolved_config(cfg: RunCfg, run_dir: str | Path) -> None:
    path = Path(run_dir) / "config.resolved.yaml"
    path.write_text(yaml.safe_dump(cfg.resolved_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MoE Bench v2 launcher")
    parser.add_argument("config")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--isolate-schemes", action="store_true",
                        help="Run each scheme in a separate torchrun process (required for mixing NVSHMEM schemes)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    run_dir = make_run_dir(cfg, create=not args.dry_run)
    if args.dry_run:
        for line in dry_run_lines(cfg, run_dir):
            print(line)
        return 0

    write_resolved_config(cfg, run_dir)
    points = expand_sweep(cfg)
    run_env = env_for_run(cfg)
    write_manifest(
        run_dir,
        cfg,
        points,
        argv=list(argv) if argv is not None else list(sys.argv[1:]),
        effective_env=run_env,
        unset_env=_UNSET_FOR_RUN,
    )
    child_env = os.environ.copy()
    for key in _UNSET_FOR_RUN:
        child_env.pop(key, None)
    child_env.update(run_env)
    exit_code = 0

    if args.isolate_schemes:
        for point in points:
            for scheme_cfg in point.cfg.schemes:
                isolated_raw = point.cfg.resolved_dict()
                isolated_raw["schemes"] = {"enabled": [scheme_cfg.code], scheme_cfg.code: {"tunables": scheme_cfg.tunables}}
                isolated_point = SweepPoint(index=point.index, cfg=_build_run_cfg_from_module(isolated_raw), values=point.values)
                command = build_worker_command(cfg, isolated_point, run_dir)
                completed = subprocess.run(command, env=child_env, check=False)
                exit_code = max(exit_code, completed.returncode)
    else:
        for point in points:
            command = build_worker_command(cfg, point, run_dir)
            if (shutil.which(command[0]) is None or os.name == "nt") and cfg.dist.nproc == 1:
                completed_returncode = worker.main(["--run-dir", str(run_dir), "--point-json", command[-1]])
            else:
                completed = subprocess.run(command, env=child_env, check=False)
                completed_returncode = completed.returncode
            exit_code = max(exit_code, completed_returncode)
            if completed_returncode != 0:
                break

    results_path = run_dir / "results.jsonl"
    if results_path.exists():
        generate_summary(read_results(results_path), run_dir / "summary.md")
    return exit_code


def _build_run_cfg_from_module(raw: dict) -> 'RunCfg':
    return run_cfg_from_dict(raw)


def _quote_command(command: Sequence[str]) -> str:
    return " ".join(_quote_part(part) for part in command)


def _quote_part(part: str) -> str:
    if not part or any(ch.isspace() for ch in part):
        return '"' + part.replace('"', '\\"') + '"'
    return part


if __name__ == "__main__":
    raise SystemExit(main())
