# EXP-022: c 组正确性根因与修复（B2）

> Date: 2026-07-05 / GPUs 9,11,13,15 / 数据: results/20260705_063409_REWORK_B_final

## TL;DR
c1-c5 cos_sim~0.33 的根因是 **H2 部分和：方案输出漏加 shared expert**。
golden（moe_bench/verify.py::golden_moe）包含 shared expert 贡献，而修复前
schemes/td_common.py 的 c 组 run() 只返回路由专家部分。补上 shared_ep 后全部通过新校验门。

## 证据链
1. 修复前特征：全部 c 组 cos~0.333、max_abs~6e-5（当时输出量级 ~1e-4，误差与信号同阶）——
   与"缺一个加性分量"的假设一致；与 EXP-001 当时"EP 输出零位多"的解释不一致
   （若两边零位相同，cos 应~1，该解释已在 EXP-001 文首标记为错误）。
2. 修复位置：moe_bench/schemes/td_common.py 的 _build_ep_bf16 / _build_ep_fp8 / _build_tp，
   run() 返回前统一 `result = result + shared(bundle.hidden_local)`（shared_ep 构造）。
3. 修复后校验（v1, M=5120，三道门全过）：

| 方案 | 对比 golden | cos_sim | rel_p99 |
|---|---|---|---|
| c1 | golden_bf16 | 0.9992 | ~1.87 |
| c2 | golden_bf16 | 0.9990 | ~1.93 |
| c3 | golden_fp8sim_rowwise | 0.9980 | ~2.90 |
| c4 | golden_fp8sim_group128 | 0.9994 | ~1.52 |
| c5 | golden_fp8sim_group128 | 0.9994 | ~1.52 |

## 附注（对报告的披露要求）
当前 c 组 shared expert 为**串行**追加（fused kernel 结束后加一次），与 a1 口径一致、对比公平，
但存在把 shared 放到并行流与融合 kernel 重叠的优化空间。
最终报告的方案说明里必须带这句披露。
