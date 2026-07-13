# REWORK-D：REWORK-C 收尾清单（小尾巴，不引入新范围）

> 交给执行模型在 GPU 服务器上完成。这是 REWORK-C 的**剩余项清单 + 一处事实错误修正**，全部完成即整改阶段终结，直接进入性能分析与经验分享。
> 执行须知与红线同 REWORK-C（原代码块做锚点搜索、【待填】才填、不删 results、每任务单独 commit `D<N>: ...`）。
> 依赖顺序：D1（修正）最先；D2 排查有界不发散；D3/D4/D5 可并行；D6 之后 D7；D8 收官回归。

## 现状（2026-07-05 审计 new_start07051555 后）

已完成：C0 归档、C1 scale 门（10 组 scale 0.998–0.999）、C2 文档链主体、C3 对跑（a1 之谜解决：老 a1 同卡 5.537ms，Δ1.6%）、C5 的 v1 M sweep（700 行全 PASS）、C6 的 test_a5/HANDOFF。
剩余：**EXP-023 的 b3 归因是错的（D1 修正）**；mb_chunk_overlap 未跑、mb_tile_order 未实现；EXP-011/012/014/016 文档缺；und/gen sweep 与 comm_fraction.md 缺；final_report.md 一字未动。

---

## D1 修正 EXP-023 的 b3 归因（事实错误，最优先）

**错误内容**：EXP-023 称 b3 +11% 差距来自"老 N_DOWN=4096 使 down GEMM 更大、RS 字节更多"。经核验老仓库 `benchmarks/data.py`：老 bench `prepare_moe_weights(E, N_down=4096, gateup//2=3072)` 生成 w2 = (E, 4096, 3072)，新 bench w2 = (E, K=4096, n_down=3072)——**两边是同一个 3072→4096 GEMM，RS 都传 K=4096 宽的输出，计算与通信字节完全相同**，只是参数命名语义不同。b3 的 0.6ms 差距真实存在且未解释。

### D1-1 EXP-023 文本替换

**原文**（TL;DR 段中这两句）：

```markdown
b3 差距 11%（5.259 vs 5.849ms），主要归因于配置差异（老 N_DOWN=4096 vs 新 n_down=3072）。
裁决：采用分支 (b) —— 差异来自可指认的配置差。报告统一采用**新口径**。
```

**替换为**：

```markdown
b3 差距 11%（5.259 vs 5.849ms），⚠ 归因未定：经核验两边 down GEMM 与 RS 字节完全相同
（老 N_DOWN=4096 是 down 输出维=K，新 n_down=3072 是 down 输入维=intermediate，同一计算），
配置差异解释不成立。疑似 overlap 模块迁移退化，见"b3 差距排查"一节与 EXP-025。
裁决：a1 基线采用**新口径**（Δ仅1.6%）；b3 结论在 EXP-025 关闭前按双口径披露。
```

**原文**（"差异解释"整节，从 `## 差异解释` 到 `## 基线口径裁决` 之前）替换为：

```markdown
## 差异解释
- **a1 +1.6%**：噪声范围，机器负载/时钟漂移。老报告的 7.63ms 来自不同卡/配置，不再引用。
- **b3 +11.2%**：⚠ 原版本归因于 N_DOWN 配置差异，已被证伪（两边 w2 均为 (E,4096,3072)，
  同一 GEMM、同一 RS 字节）。真实差距 0.6ms 未解释，最可能是 overlap 模块迁移时
  丢失了老 b3 的 FP8-RS 优化路径（老仓库 FP8_RS_OVERLAP_OPT.md 记载 separate-dequant
  等迭代把 b3 从 5.365 优化到 5.120ms）。排查见 EXP-025；关闭前 b3 按双口径披露：
  老实现 1.05x vs a1（5.259/5.537），新实现 0.96x（5.849/5.625）。
```

### D1-2 同步修正两处引用

- `docs/exp/BLOCKERS.md` "已解决"表中 `a1 新旧口径差异` 一行，解决方式改为：`a1 差 1.6% 属噪声，采用新口径（EXP-023）；b3 的 0.6ms 差距归因转入活跃区 EXP-025。` 并在"活跃"表加一行：`| b3 新实现比老实现慢 0.6ms（1.05x→0.96x 翻转） | chunk overlap 结论受影响 | EXP-025 排查，关闭前双口径披露 |`
- `docs/exp/INDEX.md` EXP-023 行 Summary 改为：`a1 对账完成（Δ1.6%，采用新口径）；b3 差距归因证伪，转 EXP-025。`

---

## D2 b3 迁移退化排查（EXP-025，有界，最多做完下面清单就收）

目的：解释或修复 0.6ms。**收敛规则写死**：清单做完仍未定位 → 在 EXP-025 记录已排除项，b3 双口径披露（D7 有现成语句），继续后面任务，不再回头。

1. **RS 模式核对**：老 b3 的最优实现是 separate-dequant（FP8_RS_OVERLAP_OPT.md：separate 5.120 vs packed 5.192）。
   `grep -rn "separate\|packed" {{OLD_BENCH}}/moe_overlap {{OLD_BENCH}}/benchmarks/overlap_cases.py` 找到模式开关及默认值；
   `grep -rn "separate\|packed" moe_bench/overlap moe_bench/schemes/overlap_*.py` 对照新侧是否有该开关、默认值是否一致。不一致 → 改成老默认，跑 D2-4 验证。
2. **文件级 diff**：`diff -r {{OLD_BENCH}}/moe_overlap moe_bench/overlap`（忽略 import 行差异）。凡逻辑差异逐条记录：哪个函数、老新行为、是否影响 b3 的 FP8 RS 路径。
3. **chunk 配置矩阵**：新 b3 分别用 `n_chunks_gateup/down` = (2,4)、(4,4)、(2,8)、(1,4) 各跑 v1 单点（`--set schemes.b3.tunables.n_chunks_gateup=... --set schemes.b3.tunables.n_chunks_down=...`），确认 (2,4) 确为新实现最优、排除"只是没调好"。
4. **验证跑**：每改一项只跑 `--set "schemes.enabled=[a1,b3]"` v1 单点，b3 目标 ≤5.4ms（老 5.259 + 噪声余量）。
5. 产出 `docs/exp/EXP-025-b3-migration-regression.md`：排查项 → 结论（修复了 / 排除了 / 未定位），末行明确写"b3 报告口径：单口径（已修复）或双口径（未修复）"。INDEX 加行。

---

## D3 跑 mb_chunk_overlap + EXP-016（脚本已在位）

```bash
cd {{NEW_ROOT}}
export CUDA_VISIBLE_DEVICES=9,11,13,15
python microbench/mb_chunk_overlap.py     # 产出 results/microbench_chunk_overlap.jsonl
```

注意：若 D2 修复了 b3，先修再跑本项（否则 b3 的 η 曲线要重采）。`EXP-016-chunk-overlap.md` 结构照 EXP-010（TL;DR / 表格用 `python tools/jsonl_to_md.py results/microbench_chunk_overlap.jsonl M scheme n_chunks latency_ms overlap_efficiency_eta exposed_comm_ms` / "回答什么问题"）。TL;DR 必须回答：最优 chunk 数是多少、η 峰值多少、η 为负出现在哪些点（负值如实保留——这是经验分享 §5 负收益区间的直接证据）。

## D4 实现并跑 mb_tile_order_groupgemm + EXP-012

按 REWORK-C C4-2 的六步执行（先 grep 真实签名再写；三种顺序=专家有序/randperm 打乱/半 phase 分组；UND 与 GEN 两 shape；warmup≥20 repeat≥50）。**sanity 锚点**：老口径 UND 322.8/401.0/329.7µs 的相对关系（打乱明显最慢、半 phase 接近有序）必须复现，绝对值可不同；不复现 → 停写 BLOCKERS，EXP-012 状态记 blocked，不硬凑。完成后：删脚本 docstring 的 `SERVER-VERIFY` 字样，`tests/test_a5_handoff.py` 把 `mb_tile_order_groupgemm.py` 从 PENDING 挪进 IMPLEMENTED（PENDING 清空）。

## D5 补写 EXP-011 / EXP-014（数据已在 jsonl，正文预写好，整块照抄）

`docs/exp/EXP-011-dispatch-prep.md`：

```markdown
# EXP-011: dispatch prep 开销（token 搬运 + pinned host 回读同步）

> Date: 2026-07-05 / 单卡 RTX PRO 5000 / 数据: results/microbench_dispatch_prep.jsonl

## TL;DR
两笔开销性质完全不同：
- token permutation gather 是带宽项，随 M 线性增长：M=1024 时 0.030ms → M=5120 时 0.362ms（gen 4096×3200×topk16 为 0.400ms）。
- **pinned host 回读同步 ≈ 0.56ms，与 M 无关**——这是 EP 路线 preprocess 里 CPU 必须读 num_recv_tokens
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
```

`docs/exp/EXP-014-group-gemm-dist.md`：

```markdown
# EXP-014: token 分布不均对 GroupGEMM 的影响

> Date: 2026-07-05 / 单卡 RTX PRO 5000 / vLLM fused_experts, FP8 group128 / 数据: results/microbench_group_gemm.jsonl

## TL;DR
失衡对 GroupGEMM 的伤害随 M 增大而消失：
- M=1024：zipf +15.6%（0.743→0.859ms）、hotspot +13.1%（0.840ms），失衡度 max/mean≈7.7/6.8；
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
"要不要为路由失衡做专门优化？"——大 batch 场景（M≥4096）GroupGEMM 层面不需要；
失衡的真正代价在 EP 通信侧（热点专家所在 rank 的收发字节放大），该部分由 sweep 的
imbalance 配置（routing.imbalance=zipf/hotspot）在整层口径下测，属性能分析阶段。
```

INDEX 对应两行状态改 done。

## D6 und/gen sweep + comm_fraction.md

```bash
cd {{NEW_ROOT}}
python -m moe_bench.cli configs/sweep_und.yaml --isolate-schemes --set run.tag=sweep_und_share
python -m moe_bench.cli configs/sweep_gen.yaml --isolate-schemes --set run.tag=sweep_gen_share
```

失败的组如实记 BLOCKERS（und 的 TP 组应走 L2 padding，结果行 `shard_level` 应为 L2——顺带验证）。

**新建 `tools/make_comm_fraction.py`（整块照抄）**，然后 `python tools/make_comm_fraction.py results/20260705_073111_sweep_v1_share/results.jsonl results/microbench_comm_prims.jsonl > docs/report/comm_fraction.md`：

```python
"""Generate the comm-fraction table from a sweep results.jsonl + comm_prims jsonl.

Usage: python tools/make_comm_fraction.py <sweep_results.jsonl> <comm_prims.jsonl>
Writes a markdown table to stdout.
"""

import json
import sys


def comm_ms(rows: list[dict], total_bytes: int, field: str) -> float:
    best = min(rows, key=lambda r: abs(r["total_bytes"] - total_bytes))
    world = best["world_size"]
    return (total_bytes * (world - 1) / world) / (best[field] * 1e9) * 1000.0


def main() -> int:
    sweep = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
    comm = [json.loads(line) for line in open(sys.argv[2], encoding="utf-8")]

    med: dict[tuple[int, str], float] = {}
    K = sweep[0]["shape"]["K"]
    for row in sweep:
        key = (row["shape"]["M"], row["scheme"])
        med.setdefault(key, row["lat_ms"]["med"])

    ms_values = sorted({m for (m, _) in med})
    print("# 通信占比表（v1 sweep × EXP-013 带宽折算）\n")
    print("| M | AG+RS bytes (MB) | T_comm_est (ms) | T_a1 (ms) | 通信占比 | b3 vs a1 | c3 vs a1 |")
    print("|---|---|---|---|---|---|---|")
    for m in ms_values:
        ag_bytes = m * K * 2          # BF16 hidden AG
        rs_bytes = m * K * 2          # BF16 输出 RS
        t_comm = comm_ms(comm, ag_bytes, "ag_bandwidth_gbps") + comm_ms(comm, rs_bytes, "rs_bandwidth_gbps")
        t_a1 = med.get((m, "a1"))
        if t_a1 is None:
            continue
        b3 = med.get((m, "b3"))
        c3 = med.get((m, "c3"))
        cells = [
            str(m),
            f"{(ag_bytes + rs_bytes) / 1e6:.1f}",
            f"{t_comm:.3f}",
            f"{t_a1:.3f}",
            f"{t_comm / t_a1:.1%}",
            f"{t_a1 / b3:.2f}x" if b3 else "-",
            f"{t_a1 / c3:.2f}x" if c3 else "-",
        ]
        print("| " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## D7 final_report.md 更新（目前一字未动，是对外风险最大项）

1. **方法论节**追加两句披露（REWORK-C C6-4 原文，整块照抄）：

```markdown
- 校验判定为三道门（cos_sim / 掩码 rel_p99 / 最小二乘 scale）；rel 门因 FP8 掩码相对误差重尾而阈值宽松，实际判别力主要来自 cos 门与 scale 门（EXP-021）。
- c 组（TD 融合方案）的 shared expert 为串行追加，与 a1 口径一致；存在并行流重叠的进一步优化空间（EXP-022 附注）。
```

2. **§4 v1 结果表**整体替换为（数据源 `results/20260705_073111_sweep_v1_share`，M=5120 行；若 D2 修复了 b3 则用修复后的重测值更新 b3 行）：

```markdown
## 4. Results: v1 Shape (M=5120, K=4096, E=64, top_k=8) — 数据: 20260705_073111_sweep_v1_share

| Scheme | med (ms) | vs a1 | Verify(三道门) |
|--------|----------|-------|--------|
| c3 TD-EP-FP8 | 4.56 | **1.23x** | PASS |
| c5 TD-TP-FP8-RS | 4.97 | 1.13x | PASS |
| c4 TD-TP-FP8 | 4.98 | 1.13x | PASS |
| a1 vLLM-TP-Serial | 5.63 | 1.00x | PASS |
| b3 Overlap-FP8AG-FP8RS | 5.84 | 0.96x ⚠ | PASS |
| b1 Overlap-BF16 | 5.85 | 0.96x | PASS |
| b2 Overlap-FP8AG | 5.86 | 0.96x | PASS |
| c1 TD-EP-BF16 | 6.00 | 0.94x | PASS |
| c2 TD-TP-BF16 | 6.01 | 0.94x | PASS |
| a2 vLLM-EP-Naive | 6.38 | 0.88x | PASS |

⚠ b3 存在迁移退化疑点：老实现同卡实测 5.26ms（1.05x vs a1），见 EXP-023/EXP-025。
```

3. **新增 §4.1 M sweep 小节**（数据同上目录，表格照抄）：

```markdown
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

c3 盈亏平衡点 ≈ M=3072；c4/c5 ≈ M=2048。小 M 时 c 组受 preprocess 固定票价
（EXP-010 前置链 0.2ms + EXP-011 host 同步 0.56ms）与 NVSHMEM 初始化路径拖累。
```

4. §5（und）沿用现有 + 加一行"完整 und/gen sweep 见 sweep_und_share / sweep_gen_share（D6）"；旧数字段落（引用 archive 前目录的）删除或移入"历史参照"附录并注明不可引用。

## D8 收官回归

```bash
cd {{NEW_ROOT}}
python -m pytest tests -q                                  # 0 failed
grep -rn "SERVER-VERIFY" microbench | wc -l                # D4 完成后应为 0（common.py 的模板引用除外，如有则注明）
grep -rn "待填" docs/exp/ | wc -l                          # 0
python -m moe_bench.cli configs/smoke.yaml --isolate-schemes   # 10 组 PASS 收官
```

INDEX 终态自查：EXP-A4/001/010/011/012/013/014/015/016/020/021/022/023/024/025 全部在列、状态如实（blocked 也算如实）。HANDOFF"Phase-2 进度"表 S4 行补 und/gen 结果目录、S5 行改 7/7。

## 完成判据

| # | 判据 | 证明物 |
|---|---|---|
| D1 | EXP-023 修正文本落盘；BLOCKERS/INDEX 同步 | docs/exp |
| D2 | EXP-025 落盘且末行有明确口径结论（修复 或 双口径） | EXP-025 |
| D3 | chunk_overlap jsonl + EXP-016（η 峰值与负值点写明） | results + docs |
| D4 | tile_order jsonl + EXP-012（相对关系复现或如实 blocked）；test_a5 PENDING 清空 | results + docs + tests |
| D5 | EXP-011/014 落盘，INDEX 更新 | docs/exp |
| D6 | und/gen sweep 目录 + comm_fraction.md | results + docs/report |
| D7 | final_report 披露语句 + 新 §4/§4.1 + 旧数字清理 | docs/report |
| D8 | pytest 全绿、grep 三查通过、smoke 收官 PASS | 终端输出记录 |

全绿后整改阶段终结。交接物清单见 REWORK-C 末节，其中"b 组负收益"素材按 EXP-025 的口径结论取用（修复→单口径；未修复→双口径叙述：老实现 +5% / 新实现 −4%，差异为迁移退化非物理规律）。
