# EXP-010: dispatch 前置计算开销（argsort*2 / bincount / cumsum）

> Date: 2026-07-05 / 单卡 RTX PRO 5000 / 数据: results/microbench_moe_align.jsonl

## TL;DR
EP dispatch 的 host 侧前置链（两次 argsort + bincount + cumsum）合计 **0.19-0.22ms**，
且对 M（1024->5120）、E（64->128）、top_k（8->16）都不敏感——launch-bound 的固定票价。
对 c3 的 4.57ms 端到端约占 4-5%：单层视角可接受，但它是 tile-level 路线不可省的部分。

## 数据
| M | E | top_k | total (ms) |
|---|---|---|---|
| 1024 | 64 | 8 | 0.188 |
| 1024 | 128 | 16 | 0.196 |
| 5120 | 64 | 8 | 0.213 |
| 5120 | 128 | 16 | 0.219 |

分项：argsort2 最贵（~0.083-0.086ms），argsort1/bincount ~0.046-0.048ms，cumsum ~0.012ms。
完整表：`python tools/jsonl_to_md.py results/microbench_moe_align.jsonl M E top_k argsort1_ms argsort2_ms bincount_ms cumsum_ms total_ms`
