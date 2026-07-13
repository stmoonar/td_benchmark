from __future__ import annotations

import pytest

from scripts.select_idle_gpus import (
    GPU_GROUPS,
    GpuState,
    parse_busy_uuids,
    parse_gpu_rows,
    select_idle_group,
)


def _states(indices=range(16)) -> dict[int, GpuState]:
    return {
        index: GpuState(index, f"GPU-{index}", 0, 0)
        for index in indices
    }


def test_parse_nvidia_smi_rows() -> None:
    output = "8, GPU-aaa, 12, 0\n10, GPU-bbb, 512, 7\n"
    states = parse_gpu_rows(output)
    assert states[8] == GpuState(8, "GPU-aaa", 12, 0)
    assert states[10].utilization_percent == 7


def test_parse_busy_uuids_ignores_status_text() -> None:
    output = "GPU-aaa\nNo running processes found\nMIG-bbb\n"
    assert parse_busy_uuids(output) == {"GPU-aaa", "MIG-bbb"}


def test_selects_first_priority_group_when_all_are_idle() -> None:
    selected = select_idle_group(
        _states(), set(), max_memory_mb=1024, max_util_percent=10
    )
    assert selected == GPU_GROUPS[0]


def test_compute_process_forces_next_priority_group() -> None:
    selected = select_idle_group(
        _states(), {"GPU-8"}, max_memory_mb=1024, max_util_percent=10
    )
    assert selected == GPU_GROUPS[1]


def test_memory_and_utilization_thresholds_are_enforced() -> None:
    states = _states()
    states[10] = GpuState(10, "GPU-10", 1025, 0)
    states[9] = GpuState(9, "GPU-9", 0, 11)
    selected = select_idle_group(
        states, set(), max_memory_mb=1024, max_util_percent=10
    )
    assert selected == GPU_GROUPS[2]


def test_missing_gpu_makes_group_unavailable() -> None:
    with pytest.raises(RuntimeError, match="no complete idle GPU group"):
        select_idle_group(
            _states((8, 10, 12)),
            set(),
            max_memory_mb=1024,
            max_util_percent=10,
        )
