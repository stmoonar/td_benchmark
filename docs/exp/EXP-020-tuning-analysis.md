# EXP-020: c-Group Performance Gap Analysis

> ⚠ 本文的根因分析已被 REWORK-B 推翻：真实根因是 `build_td_instance` 在 `run()` 闭包内重复初始化
> EP_MoE/TP_MoE 层（每次 bench 迭代都重做 NVSHMEM 分配 + 内核编译），而非 upstream group_gemm 慢。
> 修复后 c1 从 31ms 降到 5.97ms。原文保留如下（superseded）。

> Date: 2026-07-05 / Machine: TENCENT64 16x RTX PRO 5000 72GB Blackwell / GPUs 9,11,13,15
> Config: configs/run_v1_production.yaml (warmup=50/200, repeat=50)
> 数据目录已归档至 `results/archive_pre_rework_b/`。注意：这些数字产生于校验门修复（EXP-021）与计时污染修复（EXP-024）之前，仅作过程证据，不得引用为性能结论。

## TL;DR
c-group schemes run 3-6x slower than old benchmark's fork results. Root cause: upstream `triton_dist.kernels.nvidia.group_gemm` has different autotuning/performance characteristics than the fork's tuned version, despite our migrated code passing the same GEMM config parameters (BLOCK_SIZE_N=256, K=64, stages=3).

## Background and Hypothesis
Hypothesis: c-group slowdown is from insufficient Triton warmup/autotuning.
Result: Rejected — performance at warmup=50 vs 200 is identical (31.2ms). The slowdown is structural.

## Experiment Design
- Fixed: v1 shape, seed=42, all tunable params at fork defaults
- Variable: warmup count (5, 50, 100, 200)
- Control: a1 scheme timing (stable at 5.63ms across all warmup settings)

## Results

| Scheme | Old fork (ms) | New bench (ms) | Ratio |
|--------|--------------|----------------|-------|
| c1 TD-EP-BF16 | 9.5 | 31.2 | 3.3x |
| c2 TD-TP-BF16 | 5.7 | 40.4 | 7.1x |
| c3 TD-EP-FP8 | ~8-10 | 29.9 | ~3x |
| c4 TD-TP-FP8 | N/A | 60.5 | — |
| c5 TD-TP-FP8-RS | N/A | 60.9 | — |

## Analysis
The migrated code (`moe_bench/tdx/`) correctly passes GEMM config parameters (BLOCK_SIZE_N=256, BF16_BLOCK_K=64, stages=3) to `EpAll2AllFusedOp`. However, the actual GroupGEMM kernel comes from the upstream `triton_dist.kernels.nvidia.group_gemm.moe_grouped_gemm()`, which:
1. May have different Triton autotune configurations
2. Doesn't include fork-specific optimizations (e.g., fused SwiGLU+quantize kernel, improved dispatch warp allocation from commits `3653b8c`, `69ab14b`)
3. Uses upstream's kernel launch parameters that weren't tuned for sm120

The fork's 28 commits since baseline include kernel-level changes to:
- GEMM tile/warp/stage configurations
- Dispatch/combine warp allocations
- Fused SwiGLU+FP8 quantize kernel
- SM allocation for communication

## Conclusion and Next Steps
To close the gap, the options are:
1. **Install the fork as triton_dist** (quickest, breaks "clean upstream only" requirement)
2. **Port fork's group_gemm kernel into tdx** (adds another ~1000 lines to maintain)
3. **Accept the gap** and note it in the report as "upstream vs optimized fork" comparison

Recommendation: For this benchmark's purpose of comparing schemes, the relative ordering matters more than absolute numbers. All schemes use the same upstream, so the comparison is fair within the new bench. Report both old (fork) and new (upstream) numbers for context.
