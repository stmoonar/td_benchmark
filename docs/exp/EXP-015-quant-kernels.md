# EXP-015: group128 量化 kernel 开销

> Date: 2026-07-05 / 单卡 RTX PRO 5000 / 数据: results/microbench_quant_kernels.jsonl

## TL;DR
group128 激活量化延迟在 19-29us：小 M 时是 ~19-20us 的启动地板（launch-bound），
大 M 时逼近显存带宽（M=5120,K=4096 达 1429 GB/s）。
结论：FP8 路线每次量化的"量化税"约 20-30us——对 ms 级的 MoE 层可接受，
但 chunk 化后每 chunk 都要量化一次，chunk 数过大时量化税线性放大。

## 关键点
| M | K | latency (us) | 带宽 (GB/s) |
|---|---|---|---|
| 256 | 2048 | 19.5 | 54 |
| 1024 | 4096 | 20.8 | 403 |
| 4096 | 4096 | 25.2 | 1332 |
| 5120 | 4096 | 29.4 | 1429 |

完整表：`python tools/jsonl_to_md.py results/microbench_quant_kernels.jsonl M K group128_latency_ms group128_bandwidth_gbps`
