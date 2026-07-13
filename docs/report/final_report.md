# MoE Bench v2 — Single-Layer MoE Performance Comparison Report

## 1. Background and Objective

Fair comparison of 10 MoE single-layer forward implementations on 4x RTX PRO 5000 72GB Blackwell (sm120, PCIe).

**Hardware:** GPUs 9,11,13,15 from 16x NVIDIA RTX PRO 5000 72GB Blackwell
**Software:** PyTorch 2.10.0+cu128, Triton 3.4.0, vLLM 0.1.0, triton_dist upstream @ 1b9dc71a
**Data source:** results/20260705_073111_sweep_v1_share (REWORK-B/C verified)

## 2. Scheme Matrix

| Code | Name | Family | Parallel | Weight | Act Quant |
|------|------|--------|----------|--------|-----------|
| a1 | vLLM-TP-Serial | baseline | TP | FP8 | group128 |
| a2 | vLLM-EP-Naive | baseline | EP | FP8 | group128 |
| b1 | Overlap-BF16 | chunk_overlap | TP | FP8 | group128 |
| b2 | Overlap-FP8AG | chunk_overlap | TP | FP8 | group128 |
| b3 | Overlap-FP8AG-FP8RS | chunk_overlap | TP | FP8 | group128 |
| c1 | TD-EP-BF16 | td_fused | EP | BF16 | none |
| c2 | TD-TP-BF16 | td_fused | TP | BF16 | none |
| c3 | TD-EP-FP8 | td_fused | EP | FP8 | rowwise |
| c4 | TD-TP-FP8 | td_fused | TP | FP8 | group128 |
| c5 | TD-TP-FP8-RS | td_fused | TP | FP8 | group128 |

## 3. Methodology

- **Unified checkpoint:** All schemes share globally-quantized FP8 weights (128x128 block)
- **Unified routing:** Same gate matrix -> same token-to-expert assignment
- **Golden verification:** Three-gate判定: cos_sim / masked rel_p99 / least-squares scale (EXP-021)
- **Timing:** CUDA event pairs, barrier-synchronized, warmup=50, repeat=50
- **Process isolation:** --isolate-schemes for NVSHMEM state separation
- 校验判定为三道门（cos_sim / 掩码 rel_p99 / 最小二乘 scale）；rel 门因 FP8 掩码相对误差重尾而阈值宽松，实际判别力主要来自 cos 门与 scale 门（EXP-021）。
- c 组（TD 融合方案）的 shared expert 为串行追加，与 a1 口径一致；存在并行流重叠的进一步优化空间（EXP-022 附注）。

## 4. Results: v1 Shape (M=5120, K=4096, E=64, top_k=8) — 数据: E_final (post shared expert fix)

| Scheme | med (ms) | vs a1 | Verify(三道门) |
|--------|----------|-------|--------|
| c3 TD-EP-FP8 | 4.42 | **1.25x** | PASS |
| c4 TD-TP-FP8 | 4.94 | 1.12x | PASS |
| c5 TD-TP-FP8-RS | 4.94 | 1.12x | PASS |
| b3 Overlap-FP8AG-FP8RS | 5.26 | **1.05x** | PASS |
| b2 Overlap-FP8AG | 5.29 | 1.05x | PASS |
| a1 vLLM-TP-Serial | 5.54 | 1.00x | PASS |
| b1 Overlap-BF16 | 5.75 | 0.96x | PASS |
| c1 TD-EP-BF16 | 5.85 | 0.95x | PASS |
| c2 TD-TP-BF16 | 5.97 | 0.93x | PASS |
| a2 vLLM-EP-Naive | 6.35 | 0.87x | PASS |

### 4.1 M sweep（med ms）

| M | a1 | b3 | c3 | c4/c5 | c3 vs a1 |
|---|---|---|---|---|---|
| 128 | 0.73 | 1.70 | 1.66 | 0.87 | 0.44x |
| 512 | 0.94 | 1.72 | 1.75 | 0.96 | 0.54x |
| 1024 | 1.32 | 1.73 | 2.00 | 1.38 | 0.66x |
| 2048 | 2.35 | 2.53 | 2.61 | 2.20 | 0.90x |
| 3072 | 3.40 | 3.57 | 3.27 | 3.09 | **1.04x** |
| 4096 | 4.47 | 4.66 | 3.92 | 4.01 | 1.14x |
| 5120 | 5.63 | 5.84 | 4.56 | 4.98 | 1.23x |

c3 盈亏平衡点 ~ M=3072；c4/c5 ~ M=2048。小 M 时 c 组受 preprocess 固定票价
（EXP-010 前置链 0.2ms + EXP-011 host 同步 0.56ms）与 NVSHMEM 初始化路径拖累。

## 5. Communication Fraction

| M | AG+RS (MB) | T_comm_est (ms) | T_a1 (ms) | 通信占比 | c3 vs a1 |
|---|---|---|---|---|---|
| 1024 | 16.8 | 0.505 | 1.32 | 38% | 0.66x |
| 3072 | 50.3 | 1.242 | 3.40 | 37% | 1.04x |
| 5120 | 83.9 | 2.069 | 5.63 | 37% | 1.23x |

通信在 v1 shape 上占 ~37% (PCIe 32 GB/s)。c3 能赢是因为 tile-level fusion 把通信完全隐藏。

## 6. Key Findings

1. **c3 (TD-EP-FP8) 全场最快**：M=5120 时 4.42ms = 1.25x vs a1
2. **b2/b3 (FP8 overlap) 也优于 a1**：1.05x，FP8 AG 路径节省通信带宽有效
3. **Tile-level fusion 盈亏平衡点 ~ M=3072**：小 M 时 preprocess 固定票价使 c 组亏损
4. **Token 分布不均对 GroupGEMM 伤害随 M 增大消失**（EXP-014：M=5120 时 zipf 仅 +0.1%）
5. **Shared expert 实现质量对性能关键**：旧 fused_experts 高层调用比预构建底层 kernel 慢 0.57ms（EXP-026）

## 7. Conclusion

For 4x RTX PRO 5000 on PCIe at v1 shape (M=5120):
- **c3 EP-FP8 融合路径全面领先** (1.25x vs a1)
- **b2/b3 FP8 overlap 也有效** (1.05x，FP8 通信减半补偿了 chunk 开销)
- **小 M (<1024): a1 serial baseline 最优**（c 组固定票价占比过大）
- **BF16 路径 (b1/c1/c2) 不优于 FP8 路径**
