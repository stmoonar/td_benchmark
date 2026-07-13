#!/usr/bin/env python3
"""Select the first fully idle four-GPU group in benchmark priority order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence


GPU_GROUPS: tuple[tuple[int, ...], ...] = (
    (8, 10, 12, 14),
    (9, 11, 13, 15),
    (0, 2, 4, 8),
    (1, 3, 5, 7),
)


@dataclass(frozen=True)
class GpuState:
    index: int
    uuid: str
    memory_used_mb: int
    utilization_percent: int


def parse_gpu_rows(output: str) -> dict[int, GpuState]:
    states: dict[int, GpuState] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise RuntimeError(f"unexpected nvidia-smi GPU row: {line!r}")
        index, uuid, memory_used, utilization = fields
        state = GpuState(
            index=int(index),
            uuid=uuid,
            memory_used_mb=int(memory_used),
            utilization_percent=int(utilization),
        )
        states[state.index] = state
    return states


def parse_busy_uuids(output: str) -> set[str]:
    return {
        line.strip()
        for line in output.splitlines()
        if line.strip().startswith(("GPU-", "MIG-"))
    }


def select_idle_group(
    states: dict[int, GpuState],
    busy_uuids: set[str],
    *,
    max_memory_mb: int,
    max_util_percent: int,
    groups: Sequence[Sequence[int]] = GPU_GROUPS,
) -> tuple[int, ...]:
    for group in groups:
        if all(
            index in states
            and states[index].uuid not in busy_uuids
            and states[index].memory_used_mb <= max_memory_mb
            and states[index].utilization_percent <= max_util_percent
            for index in group
        ):
            return tuple(group)
    raise RuntimeError("no complete idle GPU group is available")


def _run_nvidia_smi(arguments: Iterable[str]) -> str:
    command = ["nvidia-smi", *arguments]
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except FileNotFoundError as exc:
        raise RuntimeError("nvidia-smi was not found") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or f"exit code {exc.returncode}"
        raise RuntimeError(f"{' '.join(command)} failed: {detail}") from exc


def collect_gpu_state() -> tuple[dict[int, GpuState], set[str]]:
    gpu_output = _run_nvidia_smi(
        (
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        )
    )
    process_output = _run_nvidia_smi(
        (
            "--query-compute-apps=gpu_uuid",
            "--format=csv,noheader,nounits",
        )
    )
    return parse_gpu_rows(gpu_output), parse_busy_uuids(process_output)


def _format_snapshot(states: dict[int, GpuState], busy_uuids: set[str]) -> str:
    lines = ["GPU snapshot (index: memory MiB, utilization %, compute process):"]
    for index in sorted(states):
        state = states[index]
        lines.append(
            f"  {index}: {state.memory_used_mb} MiB, "
            f"{state.utilization_percent}%, "
            f"{'yes' if state.uuid in busy_uuids else 'no'}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-memory-mb", type=int, default=1024)
    parser.add_argument("--max-util-percent", type=int, default=10)
    args = parser.parse_args(argv)

    if args.max_memory_mb < 0 or not 0 <= args.max_util_percent <= 100:
        parser.error("idle thresholds must be non-negative and utilization <= 100")

    try:
        states, busy_uuids = collect_gpu_state()
        selected = select_idle_group(
            states,
            busy_uuids,
            max_memory_mb=args.max_memory_mb,
            max_util_percent=args.max_util_percent,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if "states" in locals():
            print(_format_snapshot(states, busy_uuids), file=sys.stderr)
        return 2

    print(",".join(map(str, selected)))
    print(
        "Selected idle GPU group "
        f"{selected} (memory <= {args.max_memory_mb} MiB, "
        f"utilization <= {args.max_util_percent}%, no compute process)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
