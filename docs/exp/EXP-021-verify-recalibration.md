# EXP-021: 校验门重校准（B1）

> Date: 2026-07-05 / Machine: TENCENT64 16x RTX PRO 5000 (sm120) / GPUs 9,11,13,15
> 依据: REWORK-B B1 / 数据: results/20260705_063409_REWORK_B_final

## TL;DR
旧校验门（atol=0.2 绝对容差 + 输出量级 1e-4）是空洞的：全零输出也能 PASS。修复分三步：
(1) 数据量级归一——hidden std=1，w1/w2/shared 按收缩维 1/sqrt(dim) 初始化，输出量级恢复到 O(1)；
(2) PASS 判定改为三道门：cos_sim >= 阈值、掩码 rel_p99 <= 阈值、|scale-1| <= 0.05（scale 门见 C1）；
(3) 阈值以 a1 实测误差 x3 校准。

## 修复后输出量级
v1 shape 下 a1 输出 `abs().max()` = 3.73（修复前约 1e-4）。

## 阈值表（当前生效值，代码位置 moe_bench/worker.py::_DEFAULT_GATES）
| golden 类型 | cos_threshold | rel_p99_threshold | scale_tolerance | a1/对应组实测参考 |
|---|---|---|---|---|
| golden_bf16 | 0.995 | 5.0 | 0.05 | c1 实测 cos=0.9992, rel_p99~1.87 |
| golden_fp8sim_group128 | 0.995 | 5.0 | 0.05 | a1 实测 cos=0.9994, rel_p99~1.52 |
| golden_fp8sim_rowwise | 0.99 | 5.0 | 0.05 | c3 实测 cos=0.9980, rel_p99~2.90 |

## 重要局限（必须知晓）
掩码相对误差在 FP8 量化下重尾：接近掩码阈值（1e-3*max）的元素相对误差天然可达 100%+，
因此 rel_p99 实测在 1.5~2.9，按"实测x3"规则校准出的 5.0 阈值几乎不产生判别力。
当前实际把关的是 cos 门（拦方向性/结构性错误）与 scale 门（拦整体缩放错误）；
rel 门保留作粗防线。哨兵测试 tests/test_verify_gate.py 覆盖全零与 0.5x 缩放两类退化输出。

## 结论
a/b/c 全部 10 组在新门下 PASS（见 results/20260705_063409_REWORK_B_final/results.jsonl 的 verify 字段）。
全零输出、0.5x 缩放输出均被哨兵测试证实会 FAIL。
