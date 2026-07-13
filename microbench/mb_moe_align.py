"""MoE token alignment sweep plan."""

from common import run_microbench_plan


def main() -> int:
    return run_microbench_plan(
        __file__,
        "EXP-010",
        "dispatch pre-sort, align, and pad cost",
        {"M": [1024, 2048, 4096, 6647], "E": [64, 128], "top_k": [8, 16]},
        ["latency_ms", "pad_overhead_ratio", "effective_tokens_per_s"],
        "run MoE alignment kernels on the GPU server",
    )


if __name__ == "__main__":
    raise SystemExit(main())
