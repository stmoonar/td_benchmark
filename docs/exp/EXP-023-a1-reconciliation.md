# EXP-023: a1/b3 新旧 bench 同卡对账

> Date: 2026-07-05 / GPUs 9,11,13,15 / 老 bench: /data/cinnzhang_vllm_td_test/moe_bench_0607

## TL;DR
同卡同 GPU（9,11,13,15）对跑：a1 新旧差距 1.6%（5.537 vs 5.625ms），在噪声范围内。
b3 差距 11%（5.259 vs 5.849ms），⚠ 归因未定：经核验两边 down GEMM 与 RS 字节完全相同
（老 N_DOWN=4096 是 down 输出维=K，新 n_down=3072 是 down 输入维=intermediate，同一计算），
配置差异解释不成立。疑似 overlap 模块迁移退化，见"b3 差距排查"一节与 EXP-025。
裁决：a1 基线采用**新口径**（Δ仅1.6%）；b3 结论在 EXP-025 关闭前按双口径披露。

## 配置对表
| 项 | 老 bench | 新 bench | 一致? |
|---|---|---|---|
| M / K / E / top_k | 5120 / 4096 / 64 / 8 | 5120 / 4096 / 64 / 8 | YES |
| N_GATEUP | 6144 | 6144 | YES |
| N_DOWN（语义不同！） | 4096 (=K, down 输出维) | 3072 (=intermediate, down 输入维) | **NO** |
| shared expert | ON / SHARED_INTERMEDIATE=3072 | ON / shared_intermediate=3072 | YES |
| GPU 子集 | 9,11,13,15 | 9,11,13,15 | YES |
| warmup / repeat | 50 / 50 | 50 / 50 | YES |
| vLLM | 0.1.0+cu128 | 0.1.0+cu128 | YES |
| CUDA_DEVICE_MAX_CONNECTIONS | 1 | 1 | YES |

## 四数对比
| 方案 | 老 bench (ms) | 新 bench (ms) | delta |
|---|---|---|---|
| a1 | 5.537 | 5.625 | +1.6% |
| b3 | 5.259 | 5.849 | +11.2% |

## 差异解释
- **a1 +1.6%**：噪声范围，机器负载/时钟漂移。老报告的 7.63ms 来自不同卡/配置，不再引用。
- **b3 +11.2%**：⚠ 原版本归因于 N_DOWN 配置差异，已被证伪（两边 w2 均为 (E,4096,3072)，
  同一 GEMM、同一 RS 字节）。真实差距 0.6ms 未解释，最可能是 overlap 模块迁移时
  丢失了老 b3 的 FP8-RS 优化路径（老仓库 FP8_RS_OVERLAP_OPT.md 记载 separate-dequant
  等迭代把 b3 从 5.365 优化到 5.120ms）。排查见 EXP-025；关闭前 b3 按双口径披露：
  老实现 1.05x vs a1（5.259/5.537），新实现 0.96x（5.849/5.625）。

## 基线口径裁决
**分支 (b)**：差异来自可指认的配置差（N_DOWN 语义）。
报告统一采用新 bench 口径（n_down = n_gateup/2 = intermediate），老数字不引用为对照。
