# EXP-024: c 组 3-6 倍性能差根因与修复（B4）

> Date: 2026-07-05 / GPUs 9,11,13,15 / 数据: results/20260705_063409_REWORK_B_final

## TL;DR
根因是**计时污染**，不是 kernel 慢：修复前 schemes/td_common.py 的 build_td_instance 把
compat.apply()、importlib.import_module、权重视图构造、EP_MoE/TP_MoE 层构造、_init_ctx
（含 NVSHMEM 对称堆分配）全部写在被计时的 run() 闭包里，bench_cuda 的每次迭代都重做一遍。
修复：全部 setup 移到 build 期，run() 闭包只保留前向调用（EP-FP8 的激活量化属于方案本身成本，保留在计时内）。

## 前后对比（v1, M=5120, warmup=50, repeat=50）
| 方案 | 修复前 med (ms) | 修复后 med (ms) | 备注 |
|---|---:|---:|---|
| c1 | 31.2 | 5.98 | |
| c2 | 40.4 | 6.00 | sm120 可正常运行 |
| c3 | 29.9 | 4.57 | 现为全场最快，1.23x vs a1 |
| c4 | 60.5 | 4.97 | |
| c5 | 60.9 | 4.97 | |

## 对 EXP-020 的处置
EXP-020 曾把差距归因于"upstream group_gemm 缺少 fork 优化"。该归因错误：
fork 对共享 group_gemm.py 零改动（仅新增 fp8_allgather_group_gemm.py），
所引 commit 3653b8c/69ab14b 改的是已迁移的 function/nvidia/ep_moe_fused.py。
EXP-020 已加 superseded 声明，本文为正式结论。

## 残留事项
与老 bench（fork 安装态）的同卡横向对账归 EXP-023；
老报告 c1=9.5ms 的卡号不明，不作为对账基准。
