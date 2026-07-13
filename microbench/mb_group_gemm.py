"""FP8 GroupGEMM distribution sweep plan."""

from common import run_microbench_plan


def main() -> int:
    return run_microbench_plan(
        __file__,
        "EXP-014",
        "FP8 GroupGEMM efficiency versus token distribution",
        {"dtype": ["fp8"], "distribution": ["uniform", "zipf", "hotspot"], "tile": [64, 128, 256]},
        ["latency_ms", "tflops", "tile_utilization"],
        "run the FP8 GroupGEMM distribution sweep on the GPU server",
    )


if __name__ == "__main__":
    raise SystemExit(main())
