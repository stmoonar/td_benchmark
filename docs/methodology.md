# Methodology

This document is the phase-two working copy of REFACTOR_PLAN.md section 9.

## Timing Discipline
Use warmup >= 20 and repeat >= 50 for report-grade runs. Report median as the primary latency and keep average only as context. Multi-rank latency uses the slowest rank.

## Analysis Order
Start with end-to-end latency, then stage timing, then profiler top kernels, then single-kernel NCU. Do not profile every kernel before the previous layer identifies the dominant cost.

## Upper Bounds
For communication, compute `bytes / measured_bandwidth`. For compute, compare effective FLOPs against the relevant GEMM ceiling and discount padding waste.

## Overlap Metrics
Use `eta = (T_serial - T_overlap) / min(T_comm, T_compute)` and report exposed communication separately.

## Controlled Experiments
Change one variable at a time. Keep padding, FLOPs, and input data aligned or quantify the mismatch explicitly.
