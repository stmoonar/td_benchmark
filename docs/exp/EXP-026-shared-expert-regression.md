# EXP-026: shared expert 实现退化修复

> Date: 2026-07-05 / GPUs 9,11,13,15

## TL;DR
b3 迁移退化的真实根因是 `StaticSharedExpert` 实现退化——每次调用走完整 `fused_experts`
高层接口（每次重做 config 查找 + moe_align + 内部量化），且 FP8 输入被先反量化回 BF16
再让 fused_experts 重新量化（违反 overlap 模块"FP8 直通"设计）。

修复：移植老版预构建实现（一次性 config/align/buffer 分配，FP8 输入直接使用）。

## 修复前后对照（v1 M=5120, warmup=50, repeat=50）

| Scheme | 修复前 (ms) | 修复后 (ms) | 变化 | vs a1 (修复后) |
|--------|------------|------------|------|----------------|
| a1 | 5.625 | 5.535 | -90us | 1.00x |
| a2 | 6.380 | 6.351 | -29us | 0.87x |
| b1 | 5.847 | 5.753 | -94us | 0.96x |
| b2 | 5.865 | **5.291** | **-574us** | **1.05x** |
| b3 | 5.842 | **5.262** | **-580us** | **1.05x** |
| c1 | 5.987 | 5.861 | -126us | 0.94x |
| c3 | 4.538 | 4.386 | -152us | **1.26x** |
| c4 | 4.966 | 4.936 | -30us | 1.12x |

## b3 chunk 配置对照
| Config (gateup, down) | 修复后 (ms) | vs a1 |
|---|---|---|
| (2, 4) 默认 | 5.262 | 1.05x |
| (1, 4) | 5.049 | 1.10x |
| (4, 4) | 5.633 | 0.98x |

老实现同卡 b3=5.259ms (1.05x)。修复后 (2,4)=5.262ms — **完全恢复**。

## b3 口径终判
**单口径**：b3 已修复，新实现与老实现性能一致（5.262 vs 5.259ms, Δ<0.1%）。
EXP-025 的"双口径披露"不再需要。

## 根因解析
- overlap 模块本身 (`moe_bench/overlap/`) 两边逐字节一致，EXP-025 的"separate-dequant 未迁移"猜测不成立
- 真实退化在 shared_expert.py：老版用 `dispatch_fused_moe_kernel` 直接调底层 kernel，新版用 `fused_experts` 高层 API
- 关键性能差异：(1) FP8 输入直通 vs dequant+requant (2) 预构建 runtime vs 每次调用重做 config/align
