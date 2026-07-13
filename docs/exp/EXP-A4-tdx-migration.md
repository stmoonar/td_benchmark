# EXP-A4: tdx migration audit draft
> Date: 2026-07-04 / machine: local Windows CPU / repo: new_start / TD fork: `60bcb68767d21d22bcc4129d7f25446cf954e225` / TD upstream: `1b9dc71a0a58585ac99766d739bf08ec60de4ae7`

## TL;DR
A4 local skeleton is in place: `moe_bench.tdx.compat`, c1-c5 scheme specs, import-safe builders, and copied fork-changed Python files under `moe_bench/tdx`. File-level runtime parity still needs server execution because Triton/NVSHMEM/vLLM are not available in this local Windows environment.

## Migration Inputs
Fork path: `C:\Users\stmoonar\Desktop\files\forks\Triton-distributed`

Expected upstream path: `C:\Users\stmoonar\Desktop\files\forks\Triton-distributed-upstream`

Baseline commit: `1b9dc71a0a58585ac99766d739bf08ec60de4ae7`

Observed fork HEAD: `60bcb68767d21d22bcc4129d7f25446cf954e225`

## Changed Python Files From Fork Diff
| status | path |
|---|---|
| M | `python/triton_dist/function/nvidia/common.py` |
| M | `python/triton_dist/function/nvidia/ep_moe_fused.py` |
| M | `python/triton_dist/jit.py` |
| M | `python/triton_dist/kernels/common_ops.py` |
| A | `python/triton_dist/kernels/nvidia/benchmark_dot_scaled.py` |
| A | `python/triton_dist/kernels/nvidia/benchmark_fp8_vs_fp16_gemm.py` |
| M | `python/triton_dist/kernels/nvidia/ep_all2all_fused.py` |
| A | `python/triton_dist/kernels/nvidia/fp8_allgather_group_gemm.py` |
| A | `python/triton_dist/kernels/nvidia/fp8_moe_reduce_rs.py` |
| M | `python/triton_dist/kernels/nvidia/memory_ops.py` |
| M | `python/triton_dist/kernels/nvidia/moe_utils.py` |
| A | `python/triton_dist/kernels/nvidia/swiglu_quantize_fp8.py` |
| M | `python/triton_dist/layers/nvidia/ep_a2a_fused_layer.py` |
| M | `python/triton_dist/layers/nvidia/ep_moe.py` |
| M | `python/triton_dist/layers/nvidia/fp8_ep_moe.py` |
| A | `python/triton_dist/layers/nvidia/fp8_tp_moe.py` |
| M | `python/triton_dist/nv_utils.py` |

## Local Work Completed
`moe_bench.tdx.compat.apply()` now applies the required upstream runtime patches when the clean `triton_dist` modules are importable: `nv_utils.get_ptxas` injection, `jit.nvidia_stages_inspection_hook` wrapping, and `__fence` to `fence` aliasing. It still records detect-and-skip actions when the relevant local module is unavailable.

c1-c5 scheme builders are registered and import-safe. Their `run()` paths call `compat.apply()`, materialize DataBundle weights/routing through the new scheme abstraction, and call a `run_moe_bench_scheme` adapter when present before falling back to the migrated layer/function entry points. Kernel parity and real stage timing remain `SERVER-VERIFY`.

Changed fork Python files listed above are copied into `moe_bench/tdx`; see `docs/tdx_migration_manifest.json` for source-target mapping.

## SERVER-VERIFY Items
| file | verify |
|---|---|
| `moe_bench/tdx/compat.py` | Run against clean upstream Triton-distributed and confirm each detect-skip action is either unnecessary or replaced by a safe patch. |
| `moe_bench/tdx/kernels/` | Run imported kernels against clean upstream dependencies and verify rewritten imports resolve correctly. |
| `moe_bench/tdx/function/` | Run EP MoE functions against clean upstream dependencies and verify parity against fork behavior. |
| `moe_bench/tdx/layers/` | Run c1-c5 layers against clean upstream dependencies and verify parity against fork behavior. |
| `moe_bench/schemes/td_*.py` | Replace import-safe builder placeholders with real DataBundle shard materialization and GPU parity checks. |

## Blockers
No local CUDA/Triton/NVSHMEM/vLLM runtime is available. Server S3 must execute c1-c5 parity before deleting `SERVER-VERIFY` markers.
