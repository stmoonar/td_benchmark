# Methodology

This document is the phase-two working copy of REFACTOR_PLAN.md section 9.

## Timing Discipline
Use warmup >= 20 and repeat >= 50 for report-grade runs. Report median as the primary latency and keep average only as context. Multi-rank latency uses the slowest rank.

The implementation records one CUDA-event sample per iteration on every rank,
then performs an element-wise distributed MAX before computing summary
statistics. Only rank 0 persists the aggregated row.

## Fairness Contract
- Active schemes are `a1,a2,b1,b2,b3,c3,c4,c5`; BF16 Triton-distributed
  schemes `c1,c2` are disabled.
- Gate projection, softmax, and top-k routing are precomputed before timing for
  every active scheme. Scheme-specific transport of already-computed routing
  tensors remains part of that scheme's communication path.
- Token activations use FP8 E4M3 with one scale per contiguous group of 128
  elements. Weights use FP8 E4M3 with one scale per 128x128 block.
- `CUDA_DEVICE_MAX_CONNECTIONS` is explicitly removed from worker environments
  so CUDA's runtime default is used.

## Analysis Order
Start with end-to-end latency, then stage timing, then profiler top kernels, then single-kernel NCU. Do not profile every kernel before the previous layer identifies the dominant cost.

## Upper Bounds
For communication, compute `bytes / measured_bandwidth`. For compute, compare effective FLOPs against the relevant GEMM ceiling and discount padding waste.

## Overlap Metrics
Use `eta = (T_serial - T_overlap) / min(T_comm, T_compute)` and report exposed communication separately.

## Controlled Experiments
Change one variable at a time. Keep padding, FLOPs, and input data aligned or quantify the mismatch explicitly.
