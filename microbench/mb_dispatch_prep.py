"""EP dispatch preprocessing sweep plan."""

from common import run_microbench_plan


def main() -> int:
    return run_microbench_plan(
        __file__,
        "EXP-011",
        "c3 gather-scatter index and token permutation cost",
        {"M": [1024, 2048, 4096, 6647], "E": [64, 128], "top_k": [8, 16]},
        ["latency_ms", "percent_of_e2e", "tokens_per_s"],
        "run c3 dispatch preprocessing kernels on the GPU server",
    )


if __name__ == "__main__":
    raise SystemExit(main())
