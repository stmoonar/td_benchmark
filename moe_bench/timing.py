from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from statistics import mean, median, pstdev
from typing import Any, Callable, Iterator


@dataclass
class StageTimer:
    elapsed_ms: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.elapsed_ms[name] = self.elapsed_ms.get(name, 0.0) + (time.perf_counter() - start) * 1000.0


def summarize_samples(samples_ms: list[float]) -> dict[str, float]:
    if not samples_ms:
        return {"avg": 0.0, "min": 0.0, "med": 0.0, "p95": 0.0, "std": 0.0}
    ordered = sorted(samples_ms)
    p95_index = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
    return {
        "avg": mean(samples_ms),
        "min": min(samples_ms),
        "med": median(samples_ms),
        "p95": ordered[p95_index],
        "std": pstdev(samples_ms) if len(samples_ms) > 1 else 0.0,
    }


def bench_cpu(fn: Callable[[], None], warmup: int, repeat: int) -> tuple[dict[str, float], list[float]]:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return summarize_samples(samples), samples


def bench_cuda(fn: Callable[[], Any], warmup: int, repeat: int) -> tuple[dict[str, float], list[float], Any]:
    """CUDA event-based timing: barrier -> warmup -> sync+barrier -> per-iter events -> sync.

    Returns (summary_dict, samples_ms, last_output).
    """
    import torch
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    if dist.is_initialized():
        dist.barrier()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    last_output = None

    for i in range(repeat):
        start_events[i].record()
        last_output = fn()
        end_events[i].record()

    torch.cuda.synchronize()

    samples: list[float] = []
    for i in range(repeat):
        samples.append(start_events[i].elapsed_time(end_events[i]))

    return summarize_samples(samples), samples, last_output
