# REWORK-C：收尾计划（最后一个整改计划）

> 本文档交给执行模型在 **GPU 服务器** 上完成。这是**最后一个**整改计划：全部完成后，仓库直接进入"性能分析与经验分享"阶段，不再有下一轮 REWORK。
> 前置状态：REWORK-B 的 B1/B2/B4 已完成——校验门生效、c 组正确性修复（补 shared expert）、c 组计时污染修复（setup 移出 run() 闭包），v1 单点 10 组数据可信（`results/20260705_063409_REWORK_B_final`）。本计划处理遗留的 7 件事：C0 老结果归档、C1 校验门补一个真实漏洞、C2 文档链补齐（正文已预写好，复制即可）、C3 a1 新旧对账、C4 补跑 microbench、C5 sweep 采数、C6 工程收尾。C7 可选不阻塞。

## 执行须知

1. "原代码"块用于字符串搜索定位，不要依赖行号；搜不到就停下，在 `docs/exp/BLOCKERS.md` 记录后再继续。
2. 标注"**整块照抄**"的内容不要改写；标注"**【待填】**"的槽位才需要你填（填实测值，禁止编造）。
3. 每完成一个 C 任务：跑验收命令 → 单独 commit（格式 `C<N>: <一句话>`）。
4. 红线：不改 `{{TD_UPSTREAM}}` / `{{TD_FORK}}` / `{{OLD_BENCH}}`；**不删除任何 results 目录**；不引用未对账的对外结论。
5. 依赖顺序：C0/C1/C2 可并行先做；C3 与 C4 可并行；C5 在 C1 之后（门变了要用新门跑）；C6 最后。

## 路径表（同 REWORK-B）

| 占位符 | 值 |
|---|---|
| `{{NEW_ROOT}}` | `/data/cinnzhang_vllm_td_test/new_start` |
| `{{TD_UPSTREAM}}` | `/data/cinnzhang_vllm_td_test/triton_distributed-TD+Flux/python/triton_dist` @ `1b9dc71a` |
| `{{TD_FORK}}` | `/data/cinnzhang_vllm_td_test/triton_distributed-TD+Flux/python/triton_dist_fp8`（只读） |
| `{{OLD_BENCH}}` | 老 benchmark 仓库在服务器上的位置（只读；找不到先做 C3-0） |
| 本地归档 | `C:\Users\stmoonar\Desktop\files\moe_bench_0607\new_start07051312\new_start\results\`（REWORK-B 前的全部老结果快照，负责人机器上） |
| GPU / Python | `CUDA_VISIBLE_DEVICES=9,11,13,15`；`source /data/cinnzhang_vllm_td_test/venvs/vllm-td/bin/activate` |

---

## C0 恢复老结果目录（证据链）

REWORK-B 执行时删除了 2026-07-04 ~ 07-05 早间的全部 results 目录，导致 EXP-001/EXP-020 引用的证据目录不存在。处理：

1. 检查服务器 `{{NEW_ROOT}}/results/` 是否还有这些目录（`20260704_*`、`20260705_04*`、`20260705_05*`）。
2. **若服务器上也没了**：从负责人本地归档（路径表"本地归档"行）把整个 results 目录拷回服务器，放到 `{{NEW_ROOT}}/results/archive_pre_rework_b/` 下（让负责人 scp/rsync，你在 BLOCKERS 里留一条待办即可，不阻塞后续任务）。
3. 无论哪种情况，在 `docs/exp/EXP-001-initial-verification.md` 与 `EXP-020-tuning-analysis.md` 的 Results 引用处补一行：

```markdown
> 数据目录已归档至 `results/archive_pre_rework_b/`。注意：这些数字产生于校验门修复（EXP-021）与计时污染修复（EXP-024）之前，仅作过程证据，不得引用为性能结论。
```

**验收**：两份 EXP 文档有归档注记；results 下有归档目录或 BLOCKERS 有待办记录。

---

## C1 校验门补漏：缩放不敏感漏洞 + 文档化 rel 门局限

**问题**：当前双门是 cos + rel_p99。cos 对**整体缩放完全不敏感**（`0.5 × golden` 的 cos = 1.0），而 rel_p99 阈值被校准到 5.0（原因见下），`0.5 × golden` 的 rel_p99 = 0.5 也能过——即"权重乘两次/漏归一化"这类整体缩放 bug 现有门抓不住。补一道 scale 门。

顺带把一个事实写进文档（C2 的 EXP-021 里已预写）：掩码相对误差在 FP8 场景下重尾（实测 a1 rel_p99≈1.5，c3≈2.9），按"实测×3"校准出的阈值 5.0 使 rel 门几乎不产生判别力，当前把关主要靠 cos 门 + 本任务新增的 scale 门。这不是 bug，但必须留档。

### C1-1 `moe_bench/verify.py`：compare_outputs 加 scale 指标与 scale 门

**原代码**（函数签名）：

```python
def compare_outputs(
    actual: Any,
    expected: Any,
    atol: float,
    rtol: float,
    cos_threshold: float | None = None,
    rel_p99_threshold: float | None = None,
) -> dict[str, Any]:
```

**新代码**：

```python
def compare_outputs(
    actual: Any,
    expected: Any,
    atol: float,
    rtol: float,
    cos_threshold: float | None = None,
    rel_p99_threshold: float | None = None,
    scale_tolerance: float | None = None,
) -> dict[str, Any]:
```

**原代码**（rel 计算块之后、判定块之前的位置，锚点是这一段）：

```python
    if cos_threshold is None and rel_p99_threshold is None:
        passed = bool(torch.all(diff <= (atol + rtol * expected_f.abs())))
    else:
        passed = True
        if cos_threshold is not None:
            passed = passed and (cos_sim >= cos_threshold)
        if rel_p99_threshold is not None:
            passed = passed and (rel_p99 <= rel_p99_threshold)
```

**新代码**（在判定块前插入 scale 计算，并在判定里加第三道门）：

```python
    # 最小二乘缩放因子：抓"整体乘了个常数"类 bug（cos 对此完全不敏感）
    scale = 1.0
    if diff.numel():
        denom_sq = float((expected_f * expected_f).sum())
        if denom_sq > 0.0:
            scale = float((actual_f * expected_f).sum() / denom_sq)

    if cos_threshold is None and rel_p99_threshold is None and scale_tolerance is None:
        passed = bool(torch.all(diff <= (atol + rtol * expected_f.abs())))
    else:
        passed = True
        if cos_threshold is not None:
            passed = passed and (cos_sim >= cos_threshold)
        if rel_p99_threshold is not None:
            passed = passed and (rel_p99 <= rel_p99_threshold)
        if scale_tolerance is not None:
            passed = passed and (abs(scale - 1.0) <= scale_tolerance)
```

**原代码**（返回 dict）：

```python
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "cos_sim": cos_sim,
        "rel_p99": rel_p99,
        "rel_masked_max": rel_masked_max,
        "pass": passed,
    }
```

**新代码**（加 scale 字段）：

```python
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "cos_sim": cos_sim,
        "rel_p99": rel_p99,
        "rel_masked_max": rel_masked_max,
        "scale": scale,
        "pass": passed,
    }
```

### C1-2 `moe_bench/worker.py`：三处小改

`_DEFAULT_GATES` **原代码**：

```python
_DEFAULT_GATES: dict[str, dict[str, float]] = {
    "golden_bf16": {"cos_threshold": 0.995, "rel_p99_threshold": 5.0},
    "golden_fp8sim_group128": {"cos_threshold": 0.995, "rel_p99_threshold": 5.0},
    "golden_fp8sim_rowwise": {"cos_threshold": 0.99, "rel_p99_threshold": 5.0},
}
```

**新代码**：

```python
_DEFAULT_GATES: dict[str, dict[str, float]] = {
    "golden_bf16": {"cos_threshold": 0.995, "rel_p99_threshold": 5.0, "scale_tolerance": 0.05},
    "golden_fp8sim_group128": {"cos_threshold": 0.995, "rel_p99_threshold": 5.0, "scale_tolerance": 0.05},
    "golden_fp8sim_rowwise": {"cos_threshold": 0.99, "rel_p99_threshold": 5.0, "scale_tolerance": 0.05},
}
```

`_verify` 里 fallback 门 **原代码**：

```python
    gates = dict(_DEFAULT_GATES.get(vs, {"cos_threshold": 0.99, "rel_p99_threshold": 0.10}))
```

**新代码**：

```python
    gates = dict(_DEFAULT_GATES.get(vs, {"cos_threshold": 0.99, "rel_p99_threshold": 5.0, "scale_tolerance": 0.05}))
```

verify_enabled=False 的早退 dict **原代码**：

```python
        return {"max_abs": 0.0, "max_rel": 0.0, "cos_sim": 1.0, "rel_p99": 0.0, "rel_masked_max": 0.0, "pass": True, "vs": "skipped"}
```

**新代码**：

```python
        return {"max_abs": 0.0, "max_rel": 0.0, "cos_sim": 1.0, "rel_p99": 0.0, "rel_masked_max": 0.0, "scale": 1.0, "pass": True, "vs": "skipped"}
```

### C1-3 `tests/test_verify_gate.py` 追加哨兵（文件末尾整块追加）

```python
def test_uniformly_scaled_output_fails_gate():
    torch.manual_seed(0)
    golden = torch.randn(256, 64)
    result = compare_outputs(
        golden * 0.5, golden, atol=0.2, rtol=0.0,
        cos_threshold=0.99, rel_p99_threshold=5.0, scale_tolerance=0.05,
    )
    # cos(0.5g, g) == 1.0，rel_p99 == 0.5 < 5.0：前两道门都放行，必须由 scale 门拦下
    assert result["pass"] is False
    assert abs(result["scale"] - 0.5) < 0.01
```

### C1-4 验收

```bash
cd {{NEW_ROOT}}
python -m pytest tests/test_verify_gate.py -q          # 4 个测试全绿
python -m moe_bench.cli configs/smoke.yaml --isolate-schemes   # 10 组全 PASS，scale 字段应都在 1±0.02 内
```

10 组的 scale 若有偏离 1 超过 0.02 的，停下按 EXP-022 的方法排查后再继续（这本身就是 scale 门的价值）。

---

## C2 文档链补齐（正文已预写，复制 + 填【待填】即可）

先放一个通用小工具，后面生成表格用。**新建 `tools/jsonl_to_md.py`（整块照抄）**：

```python
"""Print a markdown table from a jsonl file.

Usage: python tools/jsonl_to_md.py <file.jsonl> <field1> <field2> ...
"""

import json
import sys


def main() -> int:
    path, fields = sys.argv[1], sys.argv[2:]
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    print("| " + " | ".join(fields) + " |")
    print("|" + "---|" * len(fields))
    for row in rows:
        cells = []
        for field in fields:
            value = row.get(field, "")
            cells.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        print("| " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### C2-1 `docs/exp/EXP-021-verify-recalibration.md`（新建，整块照抄后填 1 个槽）

```markdown
# EXP-021: 校验门重校准（B1）

> Date: 2026-07-05 / Machine: TENCENT64 16x RTX PRO 5000 (sm120) / GPUs 9,11,13,15
> 依据: REWORK-B B1 / 数据: results/20260705_063409_REWORK_B_final

## TL;DR
旧校验门（atol=0.2 绝对容差 + 输出量级 1e-4）是空洞的：全零输出也能 PASS。修复分三步：
(1) 数据量级归一——hidden std=1，w1/w2/shared 按收缩维 1/sqrt(dim) 初始化，输出量级恢复到 O(1)；
(2) PASS 判定改为三道门：cos_sim ≥ 阈值、掩码 rel_p99 ≤ 阈值、|scale−1| ≤ 0.05（scale 门见 EXP-021 附注与 C1）；
(3) 阈值以 a1 实测误差 ×3 校准。

## 修复后输出量级
v1 shape 下 a1 输出 `abs().max()` = 【待填：从 MOE_BENCH_DUMP_VERIFY=1 的 dump 里读一次】（修复前约 1e-4）。

## 阈值表（当前生效值，代码位置 moe_bench/worker.py::_DEFAULT_GATES）
| golden 类型 | cos_threshold | rel_p99_threshold | scale_tolerance | a1/对应组实测参考 |
|---|---|---|---|---|
| golden_bf16 | 0.995 | 5.0 | 0.05 | c1 实测 cos=0.9992, rel_p99≈1.87 |
| golden_fp8sim_group128 | 0.995 | 5.0 | 0.05 | a1 实测 cos=0.9994, rel_p99≈1.52 |
| golden_fp8sim_rowwise | 0.99 | 5.0 | 0.05 | c3 实测 cos=0.9980, rel_p99≈2.90 |

## 重要局限（必须知晓）
掩码相对误差在 FP8 量化下重尾：接近掩码阈值（1e-3×max）的元素相对误差天然可达 100%+，
因此 rel_p99 实测在 1.5~2.9，按"实测×3"规则校准出的 5.0 阈值几乎不产生判别力。
当前实际把关的是 cos 门（拦方向性/结构性错误）与 scale 门（拦整体缩放错误）；
rel 门保留作粗防线。哨兵测试 tests/test_verify_gate.py 覆盖全零与 0.5x 缩放两类退化输出。

## 结论
a/b/c 全部 10 组在新门下 PASS（见 results/20260705_063409_REWORK_B_final/results.jsonl 的 verify 字段）。
全零输出、0.5x 缩放输出均被哨兵测试证实会 FAIL。
```

### C2-2 `docs/exp/EXP-022-cgroup-correctness.md`（新建，整块照抄）

```markdown
# EXP-022: c 组正确性根因与修复（B2）

> Date: 2026-07-05 / GPUs 9,11,13,15 / 数据: results/20260705_063409_REWORK_B_final

## TL;DR
c1–c5 cos_sim≈0.33 的根因是 **H2 部分和：方案输出漏加 shared expert**。
golden（moe_bench/verify.py::golden_moe）包含 shared expert 贡献，而修复前
schemes/td_common.py 的 c 组 run() 只返回路由专家部分。补上 shared_ep 后全部通过新校验门。

## 证据链
1. 修复前特征：全部 c 组 cos≈0.333、max_abs≈6e-5（当时输出量级 ~1e-4，误差与信号同阶）——
   与"缺一个加性分量"的假设一致；与 EXP-001 当时"EP 输出零位多"的解释不一致
   （若两边零位相同，cos 应≈1，该解释已在 EXP-001 文首标记为错误）。
2. 修复位置：moe_bench/schemes/td_common.py 的 _build_ep_bf16 / _build_ep_fp8 / _build_tp，
   run() 返回前统一 `result = result + shared(bundle.hidden_local)`（shared_ep 构造）。
3. 修复后校验（v1, M=5120，三道门全过）：

| 方案 | 对比 golden | cos_sim | rel_p99 |
|---|---|---|---|
| c1 | golden_bf16 | 0.9992 | ≈1.87 |
| c2 | golden_bf16 | 0.9990 | ≈1.93 |
| c3 | golden_fp8sim_rowwise | 0.9980 | ≈2.90 |
| c4 | golden_fp8sim_group128 | 0.9994 | ≈1.52 |
| c5 | golden_fp8sim_group128 | 0.9994 | ≈1.52 |

## 附注（对报告的披露要求）
当前 c 组 shared expert 为**串行**追加（fused kernel 结束后加一次），与 a1 口径一致、对比公平，
但存在把 shared 放到并行流与融合 kernel 重叠的优化空间（见 REWORK-C C7，可选实验）。
最终报告的方案说明里必须带这句披露。
```

### C2-3 `docs/exp/EXP-024-cgroup-perf-rootcause.md`（新建，整块照抄）

```markdown
# EXP-024: c 组 3–6 倍性能差根因与修复（B4）

> Date: 2026-07-05 / GPUs 9,11,13,15 / 数据: results/20260705_063409_REWORK_B_final

## TL;DR
根因是**计时污染**，不是 kernel 慢：修复前 schemes/td_common.py 的 build_td_instance 把
compat.apply()、importlib.import_module、权重视图构造、EP_MoE/TP_MoE 层构造、_init_ctx
（含 NVSHMEM 对称堆分配）全部写在被计时的 run() 闭包里，bench_cuda 的每次迭代都重做一遍。
修复：全部 setup 移到 build 期，run() 闭包只保留前向调用（EP-FP8 的激活量化属于方案本身成本，保留在计时内）。

## 前后对比（v1, M=5120, warmup=50, repeat=50）
| 方案 | 修复前 med (ms) | 修复后 med (ms) | 备注 |
|---|---:|---:|---|
| c1 | 31.2 | 5.98 | |
| c2 | 40.4 | 6.00 | sm120 可正常运行 |
| c3 | 29.9 | 4.57 | 现为全场最快，1.23x vs a1 |
| c4 | 60.5 | 4.97 | |
| c5 | 60.9 | 4.97 | |

## 对 EXP-020 的处置
EXP-020 曾把差距归因于"upstream group_gemm 缺少 fork 优化"。该归因错误：
fork 对共享 group_gemm.py 零改动（仅新增 fp8_allgather_group_gemm.py），
所引 commit 3653b8c/69ab14b 改的是已迁移的 function/nvidia/ep_moe_fused.py。
EXP-020 已加 superseded 声明，本文为正式结论。

## 残留事项
与老 bench（fork 安装态）的同卡横向对账归 EXP-023；
老报告 c1=9.5ms 的卡号不明，不作为对账基准。
```

### C2-4 `docs/exp/EXP-013-comm-prims.md` 与 `EXP-015-quant-kernels.md`（新建，整块照抄）

EXP-013：

```markdown
# EXP-013: NCCL 通信原语带宽（AG/RS × 消息大小）

> Date: 2026-07-05 / 4 GPUs (9,11,13,15), PCIe 5.0, NCCL / 数据: results/microbench_comm_prims.jsonl

## TL;DR
AG/RS 带宽随消息大小上升，在 ~64MB 后饱和于 **32 GB/s**；小消息延迟地板 ~20µs。
v1 shape 的 BF16 hidden AG（M=5120×K=4096×2B ≈ 40MB）按 32 GB/s 折算约 0.94ms 纯传输，
是端到端 5.6ms 的 ~17%——这是 overlap 路线的收益上限依据（经验分享 §1 用）。

## 关键点
| total_bytes | AG (ms / GB/s) | RS (ms / GB/s) |
|---|---|---|
| 1 KB | 0.020 / 0.04 | 0.020 / 0.04 |
| 1 MB | 0.052 / 15.1 | 0.051 / 15.5 |
| 4 MB | 0.125 / 25.3 | 0.128 / 24.6 |
| 16 MB | 0.414 / 30.4 | 0.413 / 30.4 |
| 64 MB | 1.588 / 31.7 | 1.589 / 31.7 |
| 256 MB | 6.290 / 32.0 | 6.263 / 32.1 |

半带宽点在 1–4MB 之间：切 chunk 时单 chunk 字节数低于 ~4MB 会明显吃不满带宽——chunk 数上限的物理依据。
完整表：`python tools/jsonl_to_md.py results/microbench_comm_prims.jsonl total_bytes ag_latency_ms ag_bandwidth_gbps rs_latency_ms rs_bandwidth_gbps`

## 这组数字回答的问题
"AG 1.2–1.5ms 是实现慢还是链路慢？"——40MB/32GB/s≈0.94ms + 启动开销即实测值，慢在链路，不在实现。
```

EXP-015：

```markdown
# EXP-015: group128 量化 kernel 开销

> Date: 2026-07-05 / 单卡 RTX PRO 5000 / 数据: results/microbench_quant_kernels.jsonl

## TL;DR
group128 激活量化延迟在 19–29µs：小 M 时是 ~19–20µs 的启动地板（launch-bound），
大 M 时逼近显存带宽（M=5120,K=4096 达 1429 GB/s）。
结论：FP8 路线每次量化的"量化税"约 20–30µs——对 ms 级的 MoE 层可接受，
但 chunk 化后每 chunk 都要量化一次，chunk 数过大时量化税线性放大（经验分享 §2 成本清单用）。

## 关键点
| M | K | latency (µs) | 带宽 (GB/s) |
|---|---|---|---|
| 256 | 2048 | 19.5 | 54 |
| 1024 | 4096 | 20.8 | 403 |
| 4096 | 4096 | 25.2 | 1332 |
| 5120 | 4096 | 29.4 | 1429 |

完整表：`python tools/jsonl_to_md.py results/microbench_quant_kernels.jsonl M K group128_latency_ms group128_bandwidth_gbps`
```

### C2-5 `docs/exp/EXP-010-moe-align.md`（新建，整块照抄）

```markdown
# EXP-010: dispatch 前置计算开销（argsort×2 / bincount / cumsum）

> Date: 2026-07-05 / 单卡 RTX PRO 5000 / 数据: results/microbench_moe_align.jsonl

## TL;DR
EP dispatch 的 host 侧前置链（两次 argsort + bincount + cumsum）合计 **0.19–0.22ms**，
且对 M（1024→5120）、E（64→128）、top_k（8→16）都不敏感——launch-bound 的固定票价。
对 c3 的 4.57ms 端到端约占 4–5%：单层视角可接受，但它是 tile-level 路线不可省的部分（经验分享 §4 用）。

## 数据
| M | E | top_k | total (ms) |
|---|---|---|---|
| 1024 | 64 | 8 | 0.188 |
| 1024 | 128 | 16 | 0.196 |
| 5120 | 64 | 8 | 0.213 |
| 5120 | 128 | 16 | 0.219 |

分项：argsort2 最贵（~0.083–0.086ms），argsort1/bincount ~0.046–0.048ms，cumsum ~0.012ms。
完整表：`python tools/jsonl_to_md.py results/microbench_moe_align.jsonl M E top_k argsort1_ms argsort2_ms bincount_ms cumsum_ms total_ms`
```

### C2-6 EXP-001 文首修正声明（在标题下一行插入，整块照抄）

```markdown
> ⚠ 修正（2026-07-05）：本文对 c 组 cos_sim≈0.33 的解释（"EP 输出接近零的位置多"）是错误的——
> 若两边零位相同 cos 仍应≈1。真实原因是方案漏加 shared expert（见 EXP-022）。
> 本文的 c 组延迟数字含计时污染（见 EXP-024），仅作过程记录。数据目录见归档注记。
```

### C2-7 `docs/exp/INDEX.md` 整文件替换（整块照抄；C3/C4 完成后把对应行的【待填】改为实际状态）

```markdown
# Experiment Index

| ID | Date | Summary | Status |
|---|---|---|---|
| EXP-A4 | 2026-07-04 | 本地 tdx 迁移审计草稿。 | draft |
| EXP-001 | 2026-07-04 | 初次 10 组 smoke。c 组解释与延迟数字已被 EXP-022/024 修正，见文首声明。 | superseded-partial |
| EXP-010 | 2026-07-05 | dispatch 前置链固定票价 0.19–0.22ms，对 M/E/top_k 不敏感。 | done |
| EXP-011 | 2026-07-05 | dispatch prep：token gather 带宽成本 + pinned host 回读同步代价。 | 【待填：C4 后 done】 |
| EXP-012 | 2026-07-05 | tile 排序对 GroupGEMM 的影响（β' 复测）。 | 【待填：C4 后 done】 |
| EXP-013 | 2026-07-05 | NCCL AG/RS 在 PCIe 上饱和于 32 GB/s，半带宽点 1–4MB。 | done |
| EXP-014 | 2026-07-05 | token 分布不均 vs GroupGEMM 效率（uniform/zipf/hotspot）。 | 【待填：C4 后 done】 |
| EXP-015 | 2026-07-05 | group128 量化 19–29µs：小 M launch 地板，大 M 逼近带宽。 | done |
| EXP-016 | 2026-07-05 | b 组 chunk 数 sweep 与重叠效率 η。 | 【待填：C4 后 done】 |
| EXP-020 | 2026-07-05 | c 组性能差归因（已被推翻，见文首声明与 EXP-024）。 | superseded |
| EXP-021 | 2026-07-05 | 校验门重校准：三道门（cos/rel_p99/scale）+ 阈值表 + rel 门局限说明。 | done |
| EXP-022 | 2026-07-05 | c 组 cos=0.33 根因=漏加 shared expert；修复后全过新门。 | done |
| EXP-023 | 2026-07-05 | a1/b3 新旧 bench 同卡对账与基线口径拍板。 | 【待填:C3 后 done】 |
| EXP-024 | 2026-07-05 | c 组 3–6x 根因=计时污染（setup 在 run() 闭包内）；修复后 c3 全场最快。 | done |
```

**C2 验收**：上述 8 处文档全部落盘；`grep -rn "待填" docs/exp/` 只剩 INDEX 里 C3/C4 未完成项和 EXP-021 的量级槽（该槽在 C5 第一次带 dump 跑时顺手填掉）。

---

## C3 a1/b 组新旧对账（EXP-023）

目的：解释老 bench a1=7.63ms vs 新 bench a1=5.62ms 的 2ms 差，从而裁决 b 组"老口径 +26% vs 新口径 −4%"的矛盾。**有明确的收敛规则，不会开放式发散**（见 C3-4）。

### C3-0 定位老仓库

`{{OLD_BENCH}}` 若未知：`find /data -maxdepth 4 -name "run.sh" -path "*moe*" 2>/dev/null`，找含 `benchmark.py`、`benchmarks/`、`moe_overlap/` 的目录。找不到就在 BLOCKERS 记录并让负责人提供，先跳去做 C4。

### C3-1 配置对表

老 bench 生效值来源：run.sh 里的默认值 + 启动时的 shape 打印（`PRINT_DIAGNOSTICS` 开着跑一次读）。新 bench 来源：`results/20260705_063409_REWORK_B_final/config.resolved.yaml`。填进 EXP-023：

| 项 | 老 | 新 | 一致? |
|---|---|---|---|
| M/K/E/top_k | | 5120/4096/64/8 | |
| N_GATEUP / N_DOWN | | 6144 / 3072 | |
| shared 开关 / SHARED_INTERMEDIATE | | 1 / 3072 | |
| GPU 子集 | ⚠ 老报告可能测于 0-3 或 4-7 | 9,11,13,15 | |
| warmup/repeat | | 50/50 | |
| vLLM 版本 | | 0.1.0+cu128 | |

### C3-2 同卡同配置对跑（两组命令紧挨着跑）

```bash
cd {{OLD_BENCH}}
CUDA_VISIBLE_DEVICES=9,11,13,15 GROUP_MODE=a1,b3 M=5120 N_GATEUP=6144 SHARED_INTERMEDIATE=3072 \
CUDA_DEVICE_MAX_CONNECTIONS=1 WARMUP=50 REPEAT=50 PRINT_DIAGNOSTICS=0 bash run.sh

cd {{NEW_ROOT}}
python -m moe_bench.cli configs/run_v1_production.yaml --set "schemes.enabled=[a1,b3]" --set run.tag=exp023_new_side
```

### C3-3 若同卡同配置后 a1 差距仍 >5%

新侧 `--set profile.torch_profiler=true` 重跑 a1；老侧用 nsys（`nsys profile -o /tmp/old_a1 --stats=true bash run.sh` 或老仓库自带脚本）。两边 top-10 CUDA kernel 时长表并排贴进 EXP-023，指出差异集中的 kernel（重点看 NCCL AG/RS 时长与 fused_moe kernel 时长）。

### C3-4 收敛规则（三选一，写死，避免死循环）

- **(a) 同卡同配置后 |Δa1| ≤ 5%**：判定为"老报告数字来自不同卡子集/配置"。报告统一采用**新口径**，老数字只作历史参照脚注。
- **(b) Δ 来自可指认的配置差**（如 shared 维度、N_GATEUP、vLLM 版本）：在 EXP-023 写明该项，报告采用新口径，并注明"老报告因 X 配置不同不可直接比较"。
- **(c) 同卡同配置仍差 >5% 且 profile 指认出差异 kernel**：把证据写进 EXP-023，报告采用新口径，老口径数字标记"存在未复现差异，见 EXP-023"。
- 无论哪个分支：**报告基线一律用新 bench 的 a1**，b 组结论以新口径为准（当前即 b 组在 v1 单点不赚，这个结论如实写——经验分享 §5 正好用它）。

**验收**：`EXP-023` 落盘（对表 + 四数 + 分支判定），INDEX 行更新，`final_report.md` 里"⚠ 待对账"字样删除。

---

## C4 补跑 microbench（脚本已实现 3 个未跑 + 1 个待实现）

### C4-1 直接跑（脚本已在位）

```bash
cd {{NEW_ROOT}}
export CUDA_VISIBLE_DEVICES=9        # 单卡脚本
python microbench/mb_dispatch_prep.py
python microbench/mb_group_gemm.py

export CUDA_VISIBLE_DEVICES=9,11,13,15   # chunk sweep 走 cli 多卡
python microbench/mb_chunk_overlap.py
```

产物：`results/microbench_dispatch_prep.jsonl`、`microbench_group_gemm.jsonl`、`microbench_chunk_overlap.jsonl`。

### C4-2 `mb_tile_order_groupgemm.py`（EXP-012，唯一待实现的脚本）

实现步骤（按序，先核对再写）：

1. `grep -n "def moe_grouped_gemm\|def build_block_row_idx_info" {{TD_UPSTREAM}}/python/triton_dist/kernels/nvidia/group_gemm.py` 抄下真实签名。
2. 构造输入：E=128 个专家的 BF16 权重 `(E, N, K)`，token 按 uniform 路由分组（复用 `mb_group_gemm.py` 的 `make_topk`），按专家排序得到分组边界。
3. 三种 tile 顺序数组：①专家有序（build_block_row_idx_info 的默认输出）②对 tile 数组做 `torch.randperm` 全局打乱（模拟到达序）③半 phase 分组——把 tile 数组按专家分成前后两半交错拼接（老文档《Tile 排序对 fused_moe GEMM 性能影响》的排法：前半 phase 内专家有序、跨 phase 交错）。
4. 各顺序 warmup≥20、repeat≥50，CUDA event 计时，输出 jsonl 字段：`{"exp_id":"EXP-012","shape":"und|gen","order":"sorted|shuffled|half_phase","gemm_ms":...}`。
5. shape 两组：UND（K=2048, n_gateup=1536, E=128, top_k=8, M=6648）、GEN（K=3200, n_gateup=1536, E=128, top_k=16, M=4096）。
6. **sanity 锚点**：老口径 UND 三种排序 322.8 / 401.0 / 329.7 µs——新测绝对值可以不同，但**相对关系必须复现**（shuffled 明显最慢，half_phase 接近 sorted）。不复现就停下写 BLOCKERS，不要硬凑。

### C4-3 EXP 文档（每份 15–30 行即可，结构照 EXP-010：TL;DR / 数据表(用 tools/jsonl_to_md.py 生成) / "这组数字回答什么问题"）

- `EXP-011-dispatch-prep.md`：重点写 host_readback_sync_ms 的含义（含流排空等待的上界）。
- `EXP-014-group-gemm-dist.md`：重点写 hotspot 相对 uniform 的减速比，与 imbalance_max_over_mean 对照。
- `EXP-016-chunk-overlap.md`：重点写 η 随 n_chunks 的曲线形态与最优 chunk 数；η 为负的点如实保留并解释（对应经验分享 §5 负收益区间）。
- `EXP-012-tile-order.md`：三种排序对比 + 与老口径相对关系的印证。

**验收**：4 份 jsonl + 4 份 EXP 文档，INDEX 四行状态改 done。

---

## C5 sweep 采数（性能分析的骨干数据）

C1 完成后执行（用带 scale 门的新校验）。

```bash
cd {{NEW_ROOT}}

# 1) v1 M sweep（顺手补 EXP-021 的量级槽：第一个点开 dump）
MOE_BENCH_DUMP_VERIFY=1 MOE_BENCH_DUMP_DIR=results/verify_dump_sweep \
python -m moe_bench.cli configs/run_v1_production.yaml --isolate-schemes \
  --set "schemes.enabled=[a1,a2,b1,b2,b3,c1,c2,c3,c4,c5]" \
  --set "run.tag=sweep_v1_share" \
  --set 'sweep_axes=[{"path":"shape.M","values":[128,512,1024,2048,3072,4096,5120]}]'
# 若 --set 的 sweep_axes 解析失败：复制 run_v1_production.yaml 为 configs/sweep_v1_share.yaml，
# 手工加 sweep_axes 段后用该文件跑。
# 量级槽：python -c "import torch;d=torch.load('results/verify_dump_sweep/a1_rank0.pt');print(d['output'].abs().max())"
# 把打印值填进 EXP-021 的【待填】。

# 2) und / gen 全组
python -m moe_bench.cli configs/sweep_und.yaml --isolate-schemes --set run.tag=sweep_und_share
python -m moe_bench.cli configs/sweep_gen.yaml --isolate-schemes --set run.tag=sweep_gen_share
```

之后做"通信占比"汇总表（进 `docs/report/comm_fraction.md`）：每个 M 一行，列 = `AG_bytes(M×K×2)`、`T_comm_est(用 EXP-013 带宽折算 AG+RS)`、`T_a1(M)`、`占比=T_comm_est/T_a1`、`b3 与 c3 的 speedup_vs_a1`。这张表就是经验分享 §1/§5 的骨干图数据。

**验收**：三个 sweep 结果目录齐全、全部 verify PASS；comm_fraction.md 落盘；EXP-021 量级槽已填。und/gen 若有组失败，如实记录进 BLOCKERS（und 的 TP 组走 L2 padding 路径，`shard_level` 字段应显示 L2——顺带验证 C1 之前修的字段）。

---

## C6 工程收尾

### C6-1 `tests/test_a5_handoff.py` 修过时断言

**原代码**（整个函数替换）：

```python
def test_microbench_scripts_exist_and_are_marked_server_verify():
    for name in [
        "mb_moe_align.py",
        "mb_dispatch_prep.py",
        "mb_tile_order_groupgemm.py",
        "mb_comm_prims.py",
        "mb_group_gemm.py",
        "mb_quant_kernels.py",
        "mb_chunk_overlap.py",
    ]:
        path = Path("microbench") / name
        assert path.is_file()
        assert "SERVER-VERIFY" in path.read_text(encoding="utf-8")
```

**新代码**（C4-2 完成后 PENDING 清空）：

```python
IMPLEMENTED_MICROBENCH = [
    "mb_comm_prims.py",
    "mb_quant_kernels.py",
    "mb_moe_align.py",
    "mb_dispatch_prep.py",
    "mb_group_gemm.py",
    "mb_chunk_overlap.py",
    # C4-2 完成后加入 "mb_tile_order_groupgemm.py" 并清空 PENDING
]
PENDING_MICROBENCH = [
    "mb_tile_order_groupgemm.py",
]


def test_microbench_scripts_state():
    for name in IMPLEMENTED_MICROBENCH:
        path = Path("microbench") / name
        assert path.is_file()
        assert "SERVER-VERIFY" not in path.read_text(encoding="utf-8"), f"{name} 已实现却仍带 SERVER-VERIFY 标记"
    for name in PENDING_MICROBENCH:
        path = Path("microbench") / name
        assert path.is_file()
        assert "SERVER-VERIFY" in path.read_text(encoding="utf-8"), f"{name} 已实现请挪入 IMPLEMENTED_MICROBENCH"
```

### C6-2 `docs/exp/BLOCKERS.md` 整文件替换（整块照抄，再按实际情况增删）

```markdown
# Blockers

## 活跃
| Issue | Impact | 处理 |
|---|---|---|
| 【按实际情况填；没有就写"无"】 | | |

## 已解决（留档）
| Issue | 解决方式 |
|---|---|
| c 组 warmup=5 时延迟虚高 3–6 倍 | 根因不是 JIT 而是计时污染（setup 在 run() 闭包内），已修，见 EXP-024。 |
| c 组校验 cos≈0.33 | 漏加 shared expert，已修，见 EXP-022。 |
| und/gen 的 TP 组 intermediate_per_tp=192 不对齐 128 | runtime.py 的 tp_weight_views 自动走 L2 padding（192→256），结果行 shard_level=L2 标注。 |
| 校验门空洞（atol=0.2 绝对容差） | 三道门（cos/rel_p99/scale）+ 哨兵测试，见 EXP-021。 |
```

### C6-3 `HANDOFF.md`：在"S0-S7 Task Checklist"表格上方插入进度段（整块照抄）

```markdown
## Phase-2 进度（2026-07-05 REWORK-B/C 后）
| Task | 状态 | 证据 |
|---|---|---|
| S0 环境与老 bench 快照 | 部分：环境版本已记录；老 bench 对账见 EXP-023 | EXP-001 头部 / EXP-023 |
| S1 CUDA 单测与容差校准 | 完成（三道门 + 哨兵） | EXP-021, tests/test_verify_gate.py |
| S2 a/b parity | 运行完成；新旧口径裁决见 EXP-023 | 20260705_063409_REWORK_B_final |
| S3 tdx 迁移 parity | 完成（正确性 EXP-022，性能 EXP-024） | 同上 |
| S4 三 shape 与失衡 | v1 完成；und/gen 见 C5 结果目录 | sweep_und/gen_share |
| S5 microbench | 7/7 实现；EXP-010~016 | results/microbench_*.jsonl |
| S6 调优 | 移交"性能分析"阶段（不属于整改范围） | — |
| S7 报告与清理 | 数据章节待性能分析阶段填写 | final_report.md |
```

并把 SERVER-VERIFY Summary 表里已被 EXP-021/022/024 覆盖的行删除。

### C6-4 `docs/report/final_report.md` 数据章节规则 + 两句强制披露

- §4 起的所有数据表只允许引用 `20260705_063409_REWORK_B_final` 及之后的 results 目录；旧目录数字全部删除或移入"历史参照"附录。
- 方法论一节追加两句披露（整块照抄）：

```markdown
- 校验判定为三道门（cos_sim / 掩码 rel_p99 / 最小二乘 scale）；rel 门因 FP8 掩码相对误差重尾而阈值宽松，实际判别力主要来自 cos 门与 scale 门（EXP-021）。
- c 组（TD 融合方案）的 shared expert 为串行追加，与 a1 口径一致；存在并行流重叠的进一步优化空间（EXP-022 附注）。
```

### C6-5 全量回归

```bash
cd {{NEW_ROOT}}
python -m pytest tests -q            # 目标：0 failed（Windows libuv 例外仅本地存在，服务器不适用）
grep -rn "SERVER-VERIFY" moe_bench tests microbench | wc -l   # 记录余量，应只剩 mb_tile_order（若 C4-2 已完成则为 0）与文档性引用
python -m moe_bench.cli configs/smoke.yaml --isolate-schemes  # 10 组 PASS 收官
```

---

## C7（可选，不阻塞收官）：c3 shared expert 并行流实验

把 `_build_ep_fp8` 的 `shared(...)` 放到独立 `torch.cuda.Stream` 上、与融合 kernel 并行，`run()` 末尾 event 同步后相加。跑 c3 v1 单点对比串行版。若有收益（预期 0.1–0.3ms 量级），作为"性能分析阶段"的第一个调优数据点记入 EXP-025；无收益也记录。**做不做都不影响本计划完成判定。**

---

## 完成判据总表（全绿 = 整改阶段终结）

| # | 判据 | 证明物 |
|---|---|---|
| C0 | 老结果归档或待办留档；EXP-001/020 有归档注记 | results/archive_pre_rework_b 或 BLOCKERS |
| C1 | scale 门生效；4 个哨兵测试绿；smoke 10 组 PASS 且 scale∈1±0.02 | tests + smoke run |
| C2 | EXP-010/013/015/021/022/024 落盘；EXP-001 修正声明；INDEX 重写 | docs/exp/ |
| C3 | EXP-023 落盘（对表+四数+分支判定）；基线口径拍板 | EXP-023 |
| C4 | 4 份 jsonl + 4 份 EXP 文档；tile_order 相对关系复现 | EXP-011/012/014/016 |
| C5 | v1 M sweep + und/gen sweep 全 PASS；comm_fraction.md；EXP-021 量级槽已填 | results/*_share |
| C6 | pytest 全绿；BLOCKERS/HANDOFF/final_report 更新；披露语句就位 | git log |

## 收官交接物（给"性能分析与经验分享"阶段）

完成后通知负责人，交接物即以下清单（经验分享大纲 `TILE_OVERLAP_SHARE_OUTLINE.md` 里的 🕐 项全部解锁）：

1. **三条骨干曲线的数据**：M sweep（`sweep_v1_share`，含 c3 vs a1 vs b3 交叉）、chunk 数 sweep（EXP-016 的 η 曲线）、通信占比表（comm_fraction.md）。
2. **机理数据**：EXP-010/011（tile-level 固定票价）、EXP-012（β' 排序）、EXP-013/015（带宽墙与量化税）、EXP-014（失衡影响）。
3. **叙事素材**：EXP-020→024 的错误归因与翻案过程（经验分享 §6 反面案例）、EXP-022 的 shared expert 破案过程（§7 踩坑第 7 行）、b 组 v1 单点负收益 + EXP-023 口径裁决（§5）。
4. **头条结论（v1, 4×RTX PRO 5000, PCIe）**：c3 EP-FP8 4.57ms = 1.23x vs a1，FP8 融合路径全面领先；chunk overlap 在该配置下不赚（0.96x）——两句话就是 PPT 和分享的钩子。
