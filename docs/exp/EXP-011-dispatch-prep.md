# EXP-011: dispatch prep 开销（token 搬运 + pinned host 回读同步）

> Date: 2026-07-05 / 单卡 RTX PRO 5000 / 数据: results/microbench_dispatch_prep.jsonl

## TL;DR
两笔开销性质完全不同：
- token permutation gather 是带宽项，随 M 线性增长：M=1024 时 0.030ms -> M=5120 时 0.362ms（gen 4096x3200xtopk16 为 0.400ms）。
- **pinned host 回读同步 ~0.56ms，与 M 无关**——这是 EP 路线 preprocess 里 CPU 必须读 num_recv_tokens
  才能分配缓冲的同步点，且必须等流上已排队的 kernel 排空（测量含中等 GEMM 垫底，为真实代价上界）。
  对 c3 的 4.57ms 端到端占 ~12%，是 tile-level 路线最大的一笔固定票价（大于 EXP-010 的 0.2ms 前置链）。

## 数据
| M | K | top_k | token_gather (ms) | host_readback_sync (ms) |
|---|---|---|---|---|
| 1024 | 4096 | 8 | 0.030 | 0.562 |
| 5120 | 4096 | 8 | 0.362 | 0.562 |
| 4096 | 3200 | 16 | 0.400 | 0.562 |

## 这组数字回答的问题
"EP 融合路线的 preprocess 到底贵在哪？"——不贵在排序/计数（EXP-010, 0.2ms），
贵在 host 同步点（0.56ms 常数）。优化方向应是消除/隐藏该同步（capacity 预分配、异步两段式），
而不是优化排序 kernel。小 M 时该常数占比放大，是 c 组小 M 负收益（M=128 时 c3 仅 0.44x）的主要构成之一。
