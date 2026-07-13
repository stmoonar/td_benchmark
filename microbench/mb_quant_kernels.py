"""Group128 FP8 quantization kernel sweep plan."""

from common import run_microbench_plan


def main() -> int:
    return run_microbench_plan(
        __file__,
        "EXP-015",
        "group128 token quantization and SwiGLU+group128 kernel bandwidth",
        {"kernel": ["group128", "swiglu_group128"], "M": [1024, 4096, 6647], "K": [2048, 3200, 4096]},
        ["latency_ms", "bandwidth_gbps", "bandwidth_utilization"],
        "run group128 quantization kernels on the GPU server",
    )


if __name__ == "__main__":
    raise SystemExit(main())
