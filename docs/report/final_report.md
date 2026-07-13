# MoE Bench FP8 Comparison Report

This report is intentionally reset. Historical numbers mixed rank-local timing,
different quantization contracts, and different run lengths; they must not be
used as the result of the revised benchmark.

## Background And Goal

Compare active FP8 MoE implementations under one reproducible computation,
quantization, routing, timing, and environment contract.

## Scheme Matrix

Active schemes: `a1`, `a2`, `b1`, `b2`, `b3`, `c3`, `c4`, `c5`.
BF16 Triton-distributed schemes `c1` and `c2` are disabled.

## Fairness Method

- Routing gate/softmax/top-k is precomputed outside timing.
- Token activations are FP8 E4M3 group128.
- Weights are FP8 E4M3 block128x128.
- Each latency sample is the MAX of the same iteration across all ranks.
- Only rank 0 writes the aggregate result row.
- `CUDA_DEVICE_MAX_CONNECTIONS` is unset so the runtime default applies.
- Report-grade runs use at least 20 warmups and 50 measured iterations.

## End-To-End Results

Pending a clean remote run with `bash scripts/run_all.sh`.

## Bottleneck Breakdown

Pending remote profiler and microbenchmark artifacts.

## Deep-Dive Cases

Pending v1/und/gen shape and routing-imbalance sweeps.

## Work Timeline And Gains

Pending validated remote result artifacts.

## Conclusions

No performance winner is declared until the revised artifact validator passes
and the generated ZIP has been reviewed.
