# EXP-013: NCCL 通信原语带宽（AG/RS x 消息大小）

> Date: 2026-07-05 / 4 GPUs (9,11,13,15), PCIe 5.0, NCCL / 数据: results/microbench_comm_prims.jsonl

## TL;DR
AG/RS 带宽随消息大小上升，在 ~64MB 后饱和于 **32 GB/s**；小消息延迟地板 ~20us。
v1 shape 的 BF16 hidden AG（M=5120*K=4096*2B = 40MB）按 32 GB/s 折算约 0.94ms 纯传输，
是端到端 5.6ms 的 ~17%——这是 overlap 路线的收益上限依据。

## 关键点
| total_bytes | AG (ms / GB/s) | RS (ms / GB/s) |
|---|---|---|
| 1 KB | 0.020 / 0.04 | 0.020 / 0.04 |
| 1 MB | 0.052 / 15.1 | 0.051 / 15.5 |
| 4 MB | 0.125 / 25.3 | 0.128 / 24.6 |
| 16 MB | 0.414 / 30.4 | 0.413 / 30.4 |
| 64 MB | 1.588 / 31.7 | 1.589 / 31.7 |
| 256 MB | 6.290 / 32.0 | 6.263 / 32.1 |

半带宽点在 1-4MB 之间：切 chunk 时单 chunk 字节数低于 ~4MB 会明显吃不满带宽。
完整表：`python tools/jsonl_to_md.py results/microbench_comm_prims.jsonl total_bytes ag_latency_ms ag_bandwidth_gbps rs_latency_ms rs_bandwidth_gbps`

## 这组数字回答的问题
"AG 1.2-1.5ms 是实现慢还是链路慢？"——40MB/32GB/s~0.94ms + 启动开销即实测值，慢在链路，不在实现。
