"""Distributed communication primitive sweep plan."""

from common import run_microbench_plan


def main() -> int:
    return run_microbench_plan(
        __file__,
        "EXP-013",
        "FP8 all-gather, reduce-scatter, and all-to-all primitives",
        {"dtype": ["fp8"], "bytes": [1048576, 4194304, 16777216, 67108864]},
        ["latency_ms", "bandwidth_gbps"],
        "run distributed communication primitives on the GPU server",
    )


if __name__ == "__main__":
    raise SystemExit(main())
