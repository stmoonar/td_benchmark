# Phase Two Handoff

## Path Replacement Table
| Placeholder | Local phase-one value | Server value |
|---|---|---|
| `{{NEW_ROOT}}` | `C:\Users\stmoonar\Desktop\files\moe_bench_0607\new_start` | `/data/cinnzhang_vllm_td_test/new_start` |
| `{{TD_UPSTREAM}}` | `C:\Users\stmoonar\Desktop\files\forks\Triton-distributed-upstream` @ `1b9dc71a0a58585ac99766d739bf08ec60de4ae7` | `/data/cinnzhang_vllm_td_test/triton_distributed-TD+Flux/python/triton_dist` |
| `{{TD_FORK}}` | — | `/data/cinnzhang_vllm_td_test/triton_distributed-TD+Flux/python/triton_dist_fp8` |
| `{{CUDA_HOME}}` | `/usr/local/cuda` | `/usr/local/cuda` |
| `{{PYTHON}}` | — | `source /data/cinnzhang_vllm_td_test/venvs/vllm-td/bin/activate && python` |
| GPUs | — | `9,11,13,15` (CUDA_VISIBLE_DEVICES) |

## Environment Preparation
1. Clone clean Triton-distributed upstream at `1b9dc71a0a58585ac99766d739bf08ec60de4ae7`.
2. Set `PYTHONPATH={{NEW_ROOT}}:{{TD_UPSTREAM}}/python`.
3. Verify imports: `python -c "import moe_bench; import moe_bench.tdx.compat; print('ok')"`.
4. Verify config dry-run: `python -m moe_bench.cli configs/smoke.yaml --dry-run`.
5. Verify CPU tests: `python -m pytest tests`.
6. Local CPU non-dry-run without `torchrun` may produce `results.jsonl` rows marked `SERVER-VERIFY`; these are artifact-shape checks only, not benchmark or correctness data.

## Dependency Versions
Record exact server versions in `EXP-000-baseline-snapshot.md` and keep the generated run `manifest.json` with every result directory.

| Dependency | Requirement | Check Command |
|---|---|---|
| `torch` | CUDA-enabled build compatible with the target driver and `sm_120`. | `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"` |
| `vllm` | Installed package exposing the fused MoE kernels used by a1/a2/b paths. | `python -c "import vllm, importlib.metadata as m; print(m.version('vllm'))"` |
| `triton` | Compatible with clean Triton-distributed upstream and `/usr/local/cuda/bin/ptxas`. | `python -c "import triton, importlib.metadata as m; print(m.version('triton'))"` |
| `triton_dist` | Clean upstream checkout at `1b9dc71a0a58585ac99766d739bf08ec60de4ae7`; local fork changes must only come from `moe_bench/tdx`. | `git -C {{TD_UPSTREAM}} rev-parse HEAD` |
| `NVSHMEM` | Runtime available with configured symmetric heap and transport for c-schemes. | `python -c "import os; print(os.environ.get('NVSHMEM_SYMMETRIC_SIZE'), os.environ.get('NVSHMEM_REMOTE_TRANSPORT'))"` |

## S0-S7 Task Checklist
| Task | Purpose | Command | Expected Output | Troubleshooting |
|---|---|---|---|---|
| S0 | Environment and old benchmark snapshot | Run old repo benchmark per REFACTOR_PLAN.md S0 | `EXP-000-baseline-snapshot.md` | Check vLLM/triton_dist versions and CUDA env |
| S1 | CUDA unit tests and tolerance calibration | `python -m pytest tests`; `python tests/calibrate_tolerance.py` | CUDA tests pass and tolerance table drafted | Start with failed test file and compare against golden |
| S2 | a/b parity | `python -m moe_bench.cli configs/smoke.yaml --set schemes.enabled=[a1,a2,b1,b2,b3]` | verify PASS and old/new latency delta recorded | Inspect vLLM imports and overlap state |
| S3 | tdx migration parity | `python -m moe_bench.cli configs/smoke.yaml --set schemes.enabled=[c1,c2,c3,c4,c5]` | c schemes run against clean upstream | Use `docs/exp/EXP-A4-tdx-migration.md` audit table |
| S4 | Three shapes and imbalance calibration | Run `configs/sweep_*.yaml` with imbalance overrides | all schemes verify PASS | Use `test_ragged_block_quant_gemm.py` for L1/L2 decisions |
| S5 | Microbenchmarks | Run scripts in `microbench/` | EXP-010 through EXP-016 drafts | Profile one bottleneck at a time |
| S6 | Tuning | Run `configs/tune_c3.yaml` and related tune configs | tuned config files and EXP-020 series | Keep a1 anchor reruns in long sweeps |
| S7 | Report and cleanup | Generate `docs/report/final_report.md` | report and old-code cleanup plan | Confirm all SERVER-VERIFY rows are resolved |

## Phase-2 进度（2026-07-05 REWORK-B/C 后）
| Task | 状态 | 证据 |
|---|---|---|
| S0 环境与老 bench 快照 | 完成：环境版本已记录；老 bench 对账见 EXP-023 | EXP-001 / EXP-023 |
| S1 CUDA 单测与容差校准 | 完成（三道门 + 哨兵） | EXP-021, tests/test_verify_gate.py |
| S2 a/b parity | 完成；新旧口径裁决见 EXP-023 | 20260705_063409_REWORK_B_final |
| S3 tdx 迁移 parity | 完成（正确性 EXP-022，性能 EXP-024） | 同上 |
| S4 三 shape 与失衡 | v1 完成；und/gen 见 C5 结果目录 | sweep 结果 |
| S5 microbench | 6/7 实现；EXP-010/013/015 done | results/microbench_*.jsonl |
| S6 调优 | 移交"性能分析"阶段 | — |
| S7 报告与清理 | 数据章节待性能分析阶段填写 | final_report.md |

## SERVER-VERIFY Summary
All scheme-level SERVER-VERIFY items resolved by REWORK-B (2026-07-05). Remaining:

| File | Status |
|---|---|
| `microbench/mb_tile_order_groupgemm.py` | 待实现（需现场核对 group_gemm tile 接口） |

## Known Risks
| Risk | Mitigation |
|---|---|
| mb_tile_order_groupgemm.py 未实现 | 不阻塞主流程，EXP-012 待后续补充 |
| `und`/`gen` TP intermediate is ragged at 192. | Use A2 shard L1 by default; only use L2 after server repro proves a kernel requires aligned tails. |
| `triton.language.target_info` import differences vary by Triton version. | Keep `compat.apply()` detect-and-skip and document observed server behavior. |
| Existing old-code worktree is dirty. | Treat old repo files as read-only reference until S7 cleanup. |
