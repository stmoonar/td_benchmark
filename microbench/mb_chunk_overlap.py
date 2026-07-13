"""Chunk-count sweep plan for b-group overlap efficiency."""

from common import run_microbench_plan


def main() -> int:
    return run_microbench_plan(
        __file__,
        "EXP-016",
        "b-group chunk count versus overlap efficiency eta",
        {"n_chunks": [1, 2, 4, 8, 16], "scheme": ["b1", "b2", "b3"]},
        ["latency_ms", "overlap_efficiency_eta", "exposed_comm_ms"],
        "run the chunk sweep on the GPU server after the end-to-end suite",
    )


if __name__ == "__main__":
    raise SystemExit(main())
