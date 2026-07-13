# EXP-014: token 分布不均对 GroupGEMM 的影响

> Date: 2026-07-05 / 单卡 RTX PRO 5000 / vLLM fused_experts, FP8 group128 / 数据: results/microbench_group_gemm.jsonl

## TL;DR
失衡对 GroupGEMM 的伤害随 M 增大而消失：
- M=1024：zipf +15.6%（0.743->0.859ms）、hotspot +13.1%（0.840ms），失衡度 max/mean~7.7/6.8；
- M=5120：zipf +0.1%、hotspot +1.6%——几乎无感。
原因：大 M 时每个专家的 tile 数足够多，SM 波次能被填满，热点专家只是排队更久而非空转。

## 数据
| M | 分布 | latency (ms) | max/mean 失衡度 |
|---|---|---|---|
| 1024 | uniform | 0.743 | 1.16 |
| 1024 | zipf | 0.859 | 7.72 |
| 1024 | hotspot | 0.840 | 6.80 |
| 5120 | uniform | 3.297 | 1.10 |
| 5120 | zipf | 3.301 | 7.73 |
| 5120 | hotspot | 3.349 | 6.75 |

## 这组数字回答的问题
"要不要为路由失衡做专门优化？"——大 batch 场景（M>=4096）GroupGEMM 层面不需要；
失衡的真正代价在 EP 通信侧（热点专家所在 rank 的收发字节放大），该部分由 sweep 的
imbalance 配置（routing.imbalance=zipf/hotspot）在整层口径下测，属性能分析阶段。
