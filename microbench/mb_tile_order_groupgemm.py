"""SERVER-VERIFY: benchmark tile order and expert locality effects on FP8 GroupGEMM."""

from __future__ import annotations

from common import run_microbench_plan


def main() -> int:
    return run_microbench_plan(
        __file__,
        "EXP-012",
        "tile order and expert locality effects on FP8 GroupGEMM",
        {"order": ["expert_major", "phase_major", "half_phase"], "M": [2048, 4096, 6647]},
        ["latency_ms", "tflops", "l2_hit_rate"],
        "implement tile-order GroupGEMM benchmark with NCU hooks",
    )


if __name__ == "__main__":
    raise SystemExit(main())
