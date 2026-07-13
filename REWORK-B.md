# REWORK-B：第二阶段数据可信化整改计划（含逐行代码改法）

> 本文档交给执行模型在 **GPU 服务器** 上完成。目标：让 benchmark 产出的每一个数字都经得起追问。
> 在 B1–B4 全部完成之前，**当前 results/ 与 final_report.md 里的所有性能结论一律视为"未对账数据"，不得对外引用。**

## 执行须知（先读这段）

1. 本文的"原代码"块摘自 2026-07-05 的仓库现状。改文件时**用原代码块做字符串搜索定位**，不要依赖行号（行号可能漂移）。若搜索不到原代码块，说明文件被改过——停下来，在 `docs/exp/BLOCKERS.md` 记录差异后再继续。
2. "新代码"块可以整块照抄。除非明确写了"骨架，需补全"，否则不要自己发挥修改。
3. 每完成一个 B 任务：跑对应验收命令 → 写对应 EXP 文档 → 单独 commit（格式 `B<N>: <一句话>`）。
4. 红线：不修改 `{{TD_UPSTREAM}}`、`{{TD_FORK}}`、`{{OLD_BENCH}}`（只读）；不编造数字；修正旧文档时不删原文（文首加"⚠ 本文结论已被 EXP-xxx 修正"，原文标 superseded 保留）。

## 0. 路径与环境

| 占位符 | 值 |
|---|---|
| `{{NEW_ROOT}}` | `/data/cinnzhang_vllm_td_test/new_start` |
| `{{TD_UPSTREAM}}` | `/data/cinnzhang_vllm_td_test/triton_distributed-TD+Flux/python/triton_dist`（clean upstream @ `1b9dc71a`） |
| `{{TD_FORK}}` | `/data/cinnzhang_vllm_td_test/triton_distributed-TD+Flux/python/triton_dist_fp8`（fork HEAD `60bcb68`，只读参照） |
| `{{OLD_BENCH}}` | **待确认**：老 benchmark 仓库（moe_bench_0607，含 run.sh/benchmark.py）在服务器上的位置；没有就从本机同步。全程只读 |
| GPU | `CUDA_VISIBLE_DEVICES=9,11,13,15`，所有对比实验固定这四张卡 |
| Python | `source /data/cinnzhang_vllm_td_test/venvs/vllm-td/bin/activate` |

## 1. 背景：为什么需要 REWORK-B（审计结论摘要）

1. **校验空洞**：`worker.py::_verify` 用 `atol=0.2, rtol=0` 纯绝对容差；而 hidden（`bundle.py::_build_hidden`）和全部权重（`checkpoint.py::_randn`）都乘了 `0.01`，端到端输出量级只有 1e-4~1e-5 —— **全零输出也能 PASS**。
2. **c 组正确性未被证明**：EXP-001 里 c1–c5 的 cos_sim≈0.33。若逐元素差只有 6e-5 而 cos 只有 0.33，数学上要求信号本身就是 1e-5 量级——即 c 组输出与 golden 基本不相关。EXP-001 的"EP 输出接近零位置多所以 cos 低"解释不成立。
3. **EXP-020 归因错误**：fork 对共享 `group_gemm.py` 零改动（`git diff --stat 1b9dc71a..HEAD -- '*group_gemm*'` 只显示新增 `fp8_allgather_group_gemm.py`）；EXP-020 引用的 commit `3653b8c`/`69ab14b` 改的是已迁移的 `function/nvidia/ep_moe_fused.py`。c 组 3–6 倍差距根因未定位。
4. **a1 未对账**：老 bench a1=7.63ms、b3=5.65ms（b3 快 26%）；新 bench a1=5.63ms、b3=5.84ms（b3 慢 4%）。差异集中在 a1 基线，S2 要求的新旧对账没做。

依赖顺序：**B1 → B2 →（B3 ∥ B4）→ B5 → B6 → B7**。B1/B2 未绿之前所有重测数字仍不可信，不要提前做 B3–B6 的采数。

---

## B1 校验可信化

涉及 4 个文件：`moe_bench/data/checkpoint.py`、`moe_bench/data/bundle.py`、`moe_bench/verify.py`、`moe_bench/worker.py`，外加 1 个新测试文件。

### B1-1 `checkpoint.py`：权重初始化按收缩维归一

**原代码**（文件末尾附近）：

```python
def _randn(torch: Any, shape: tuple[int, ...], generator: Any, device: Any) -> Any:
    return torch.randn(shape, generator=generator, device=device, dtype=torch.float32) * 0.01
```

**新代码**（加 `std` 参数，删除固定 0.01）：

```python
def _randn(torch: Any, shape: tuple[int, ...], generator: Any, device: Any, std: float) -> Any:
    return torch.randn(shape, generator=generator, device=device, dtype=torch.float32) * std
```

**原代码**（`build_checkpoint` 函数体）：

```python
    w1_src = _randn(torch, (shape.E, shape.n_gateup, shape.K), generator, device)
    w2_src = _randn(torch, (shape.E, shape.K, shape.n_down), generator, device)
    w1 = _quantized(w1_src)
    w2 = _quantized(w2_src)
    shared_w1 = None
    shared_w2 = None
    if shape.shared_experts:
        shared_w1 = _quantized(_randn(torch, (1, 2 * shape.shared_intermediate, shape.K), generator, device))
        shared_w2 = _quantized(_randn(torch, (1, shape.K, shape.shared_intermediate), generator, device))
    gate_weight = _randn(torch, (shape.E, shape.K), generator, device).to(torch.bfloat16)
```

**新代码**（std = 1/sqrt(该矩阵在 matmul 中的收缩维)。注意 w2 的布局是 `(E, K, n_down)`、在 golden 里以 `intermediate @ w2[e].T` 使用，收缩维是 `n_down`；shared_w2 同理收缩维是 `shared_intermediate`；gate 保持小 std，让 softmax 后的路由分布合理）：

```python
    w1_src = _randn(torch, (shape.E, shape.n_gateup, shape.K), generator, device, std=shape.K ** -0.5)
    w2_src = _randn(torch, (shape.E, shape.K, shape.n_down), generator, device, std=shape.n_down ** -0.5)
    w1 = _quantized(w1_src)
    w2 = _quantized(w2_src)
    shared_w1 = None
    shared_w2 = None
    if shape.shared_experts:
        shared_w1 = _quantized(_randn(torch, (1, 2 * shape.shared_intermediate, shape.K), generator, device, std=shape.K ** -0.5))
        shared_w2 = _quantized(_randn(torch, (1, shape.K, shape.shared_intermediate), generator, device, std=shape.shared_intermediate ** -0.5))
    gate_weight = _randn(torch, (shape.E, shape.K), generator, device, std=0.01).to(torch.bfloat16)
```

### B1-2 `bundle.py`：hidden 去掉 0.01

**原代码**（`_build_hidden`）：

```python
def _build_hidden(torch: Any, cfg: RunCfg, world_size: int, device: Any) -> Any:
    generator = torch.Generator(device=device).manual_seed(cfg.seed + 17)
    return torch.randn(
        (cfg.shape.M_aligned(world_size), cfg.shape.K),
        generator=generator,
        device=device,
        dtype=torch.float32,
    ) * 0.01
```

**新代码**（std=1）：

```python
def _build_hidden(torch: Any, cfg: RunCfg, world_size: int, device: Any) -> Any:
    generator = torch.Generator(device=device).manual_seed(cfg.seed + 17)
    return torch.randn(
        (cfg.shape.M_aligned(world_size), cfg.shape.K),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
```

改完后 v1 shape 的输出量级预期 O(0.1~10)。B1-6 的验收步骤会实测确认。

### B1-3 `verify.py`：`compare_outputs` 增加掩码相对误差与双门判定

**原代码**（整个函数替换）：

```python
def compare_outputs(actual: Any, expected: Any, atol: float, rtol: float) -> dict[str, Any]:
    torch = _torch()
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    denom = expected_f.abs().clamp(min=1e-12)
    max_abs = float(diff.max()) if diff.numel() else 0.0
    max_rel = float((diff / denom).max()) if diff.numel() else 0.0
    cos_sim = float(torch.nn.functional.cosine_similarity(actual_f.flatten(), expected_f.flatten(), dim=0)) if diff.numel() else 1.0
    passed = bool(torch.all(diff <= (atol + rtol * expected_f.abs())))
    return {"max_abs": max_abs, "max_rel": max_rel, "cos_sim": cos_sim, "pass": passed}
```

**新代码**（保持旧调用兼容：不传新参数时行为不变；`rel_p99` 用 kthvalue 而不是 `torch.quantile`，因为 quantile 对超大张量会报错）：

```python
def compare_outputs(
    actual: Any,
    expected: Any,
    atol: float,
    rtol: float,
    cos_threshold: float | None = None,
    rel_p99_threshold: float | None = None,
) -> dict[str, Any]:
    torch = _torch()
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    denom = expected_f.abs().clamp(min=1e-12)
    max_abs = float(diff.max()) if diff.numel() else 0.0
    max_rel = float((diff / denom).max()) if diff.numel() else 0.0
    cos_sim = float(torch.nn.functional.cosine_similarity(actual_f.flatten(), expected_f.flatten(), dim=0)) if diff.numel() else 1.0

    # 掩码相对误差：只在 |expected| 显著非零的位置上算，避免近零元素把 rel 撑爆
    rel_p99 = 0.0
    rel_masked_max = 0.0
    if diff.numel():
        abs_expected = expected_f.abs()
        mask = abs_expected > 1e-3 * float(abs_expected.max())
        if bool(mask.any()):
            rel = (diff[mask] / abs_expected[mask]).flatten()
            k = max(1, int(0.99 * rel.numel()))
            rel_p99 = float(rel.kthvalue(k).values)
            rel_masked_max = float(rel.max())

    if cos_threshold is None and rel_p99_threshold is None:
        passed = bool(torch.all(diff <= (atol + rtol * expected_f.abs())))
    else:
        passed = True
        if cos_threshold is not None:
            passed = passed and (cos_sim >= cos_threshold)
        if rel_p99_threshold is not None:
            passed = passed and (rel_p99 <= rel_p99_threshold)

    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "cos_sim": cos_sim,
        "rel_p99": rel_p99,
        "rel_masked_max": rel_masked_max,
        "pass": passed,
    }
```

### B1-4 `worker.py`：`_verify` 接线双门阈值（并留出 B2 要用的 dump 钩子）

配置系统里 `VerifyCfg.tolerances`（`config.py`）本来就存在但一直没被使用，正好用它做 YAML 覆盖入口。

**原代码**（`_verify` 尾部）：

```python
    m_local = output.shape[0]
    golden_local = golden_full[ctx.rank * m_local : (ctx.rank + 1) * m_local]

    atol = 0.2
    rtol = 0.0
    result = compare_outputs(output, golden_local, atol=atol, rtol=rtol)
    result["vs"] = vs
    return result
```

**新代码**（默认阈值表 + YAML 覆盖 + dump 钩子）：

```python
    m_local = output.shape[0]
    golden_local = golden_full[ctx.rank * m_local : (ctx.rank + 1) * m_local]

    if os.environ.get("MOE_BENCH_DUMP_VERIFY") == "1":
        dump_dir = Path(os.environ.get("MOE_BENCH_DUMP_DIR", "results/verify_dump"))
        dump_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "scheme": spec.code,
                "rank": ctx.rank,
                "output": output.detach().float().cpu(),
                "golden_local": golden_local.detach().float().cpu(),
                "golden_full": golden_full.detach().float().cpu(),
            },
            dump_dir / f"{spec.code}_rank{ctx.rank}.pt",
        )

    gates = dict(_DEFAULT_GATES.get(vs, {"cos_threshold": 0.99, "rel_p99_threshold": 0.10}))
    gates.update(cfg.verify.tolerances.get(vs, {}))
    result = compare_outputs(output, golden_local, atol=0.2, rtol=0.0, **gates)
    result["vs"] = vs
    return result
```

同一文件中，在 `def _verify(...)` **定义之前**加模块级常量（阈值是临时值，B1-6 校准后回填）：

```python
# 双门阈值：先用保守临时值，EXP-021 校准后回填（校准法见 REWORK-B B1-6）
_DEFAULT_GATES: dict[str, dict[str, float]] = {
    "golden_bf16": {"cos_threshold": 0.999, "rel_p99_threshold": 0.05},
    "golden_fp8sim_group128": {"cos_threshold": 0.999, "rel_p99_threshold": 0.05},
    "golden_fp8sim_rowwise": {"cos_threshold": 0.99, "rel_p99_threshold": 0.10},
}
```

再把日志行加上 rel_p99。**原代码**：

```python
                    f"{pass_str}  max_abs={verify_result['max_abs']:.6f}  cos_sim={verify_result['cos_sim']:.8f}"
```

**新代码**：

```python
                    f"{pass_str}  max_abs={verify_result['max_abs']:.6f}  rel_p99={verify_result.get('rel_p99', 0.0):.6f}  cos_sim={verify_result['cos_sim']:.8f}"
```

注意：`worker.py` 已经 `import os`、`from pathlib import Path`、`import torch`，不需要新增 import。

### B1-5 新增哨兵单测 `tests/test_verify_gate.py`（全新文件，整体照抄）

```python
"""B1 sentinel: the verification gate must reject all-zero outputs."""

import torch

from moe_bench.verify import compare_outputs


def test_zero_output_fails_gate():
    torch.manual_seed(0)
    golden = torch.randn(256, 64)
    result = compare_outputs(
        torch.zeros_like(golden), golden, atol=0.2, rtol=0.0,
        cos_threshold=0.99, rel_p99_threshold=0.05,
    )
    assert result["pass"] is False


def test_near_identical_passes_gate():
    torch.manual_seed(0)
    golden = torch.randn(256, 64)
    noisy = golden * 1.0001
    result = compare_outputs(
        noisy, golden, atol=0.2, rtol=0.0,
        cos_threshold=0.999, rel_p99_threshold=0.05,
    )
    assert result["pass"] is True


def test_legacy_call_without_thresholds_keeps_old_behavior():
    golden = torch.ones(8, 8)
    result = compare_outputs(golden * 1.01, golden, atol=0.2, rtol=0.0)
    assert result["pass"] is True
    assert "rel_p99" in result
```

### B1-6 验收流程（按序执行）

```bash
cd {{NEW_ROOT}}
# 1. CPU 单测（改了初始化后 test_golden/test_quantize/test_bundle 若有量级断言会挂，
#    挂了就看断言内容：只允许更新"期望量级"类断言，不允许放宽误差类断言）
python -m pytest tests/test_verify_gate.py tests/test_golden.py tests/test_quantize.py tests/test_bundle.py -q

# 2. GPU 上跑 a1 smoke，记录输出量级与实测误差（这是阈值校准数据）
python -m moe_bench.cli configs/smoke.yaml --set "schemes.enabled=[a1]"
# 从 results/<run>/results.jsonl 读 verify 字段：记录 cos_sim / rel_p99 / max_abs

# 3. 阈值校准：阈值 = a1 实测误差 × 3，向上取整到一位有效数字。
#    例：a1 实测 rel_p99=0.011 → 阈值 0.04（0.033 取整）。
#    把校准值回填到 worker.py 的 _DEFAULT_GATES，并写入 EXP-021。

# 4. a/b 组全跑一遍确认仍 PASS
python -m moe_bench.cli configs/smoke.yaml --set "schemes.enabled=[a1,a2,b1,b2,b3]"
```

**验收标准**：哨兵单测绿；a1 输出 `abs().max()` 在 O(0.1~10)（在步骤 2 临时打印或从 dump 看）；a/b 组过新门；c 组此时**预期 FAIL——这是正确行为**，进入 B2。产出 `EXP-021-verify-recalibration.md`（新旧判定对照、阈值表、量级记录）。

---

## B2 c 组正确性追查

三个候选假设：
- **H1 排列错位**：c 组 `fuse_scatter` 输出的 token 顺序 ≠ `_verify` 假设的"第 i 行 = 全局第 `rank*m_local+i` 个 token"。
- **H2 部分和**：combine 少加了某部分贡献（漏 topk 项 / 漏 shared expert——注意 c 组 spec 若不含 shared，golden 却含 shared，也会造成系统性偏差，先核对 `td_*.py` 的 SPEC 与 golden 是否一致）。
- **H3 权重应用错误**：topk_weights 没乘 / 乘两次 / renormalize 口径不一致。

### B2-1 微型调试配置 `configs/debug_tiny.yaml`（全新文件，整体照抄）

```yaml
run:
  tag: debug_tiny
  seed: 42
  warmup: 2
  repeat: 2
  output_root: results
shape:
  M: 32
  K: 256
  E: 8
  top_k: 2
  n_gateup: 512
  n_down: 256
  shared_experts: 0
routing:
  imbalance:
    kind: none
dist:
  nproc: 2
  master_port: 29511
  cuda_visible_devices: "9,11"
env:
  NVSHMEM_SYMMETRIC_SIZE: "4294967296"
  NVSHMEM_REMOTE_TRANSPORT: none
  NVSHMEM_DISABLE_CUDA_VMM: "1"
  CUDA_DEVICE_MAX_CONNECTIONS: "1"
  C_INCLUDE_PATH: /usr/local/cuda/include
  TRITON_PTXAS_PATH: /usr/local/cuda/bin/ptxas
schemes:
  enabled: [c1]
  c1:
    tunables: {}
verify:
  enabled: true
profile:
  torch_profiler: false
  intra_kernel: false
  nvtx: false
log:
  level: info
  diagnostics: true
sweep_axes: []
```

### B2-2 抓取 dump

```bash
cd {{NEW_ROOT}}
MOE_BENCH_DUMP_VERIFY=1 MOE_BENCH_DUMP_DIR=results/verify_dump_c1 \
  python -m moe_bench.cli configs/debug_tiny.yaml
ls results/verify_dump_c1/   # 应有 c1_rank0.pt / c1_rank1.pt
```

### B2-3 行匹配分析脚本 `tools/debug_row_match.py`（全新文件，整体照抄；`tools/` 目录不存在就创建）

这个脚本一步区分 H1（排列问题）和 H2/H3（数值问题）：

```python
"""Row-match analysis for c-group verify dumps.

Usage:
  python tools/debug_row_match.py results/verify_dump_c1/c1_rank0.pt

Reads {output, golden_local, golden_full} and reports:
  1. direct per-row cosine (output[i] vs golden_local[i])
  2. best-match row in golden_full for each output row
If best-match cos is high but direct cos is low -> permutation problem (H1).
If best-match cos is also low -> numeric problem (H2/H3).
"""

import sys

import torch


def row_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_n = a / a.norm(dim=1, keepdim=True).clamp(min=1e-12)
    b_n = b / b.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return (a_n * b_n).sum(dim=1)


def main() -> int:
    blob = torch.load(sys.argv[1], map_location="cpu")
    out = blob["output"]
    gl = blob["golden_local"]
    gf = blob["golden_full"]
    print(f"scheme={blob['scheme']} rank={blob['rank']} output={tuple(out.shape)} golden_full={tuple(gf.shape)}")
    print(f"output |max|={out.abs().max():.6f}  golden_local |max|={gl.abs().max():.6f}")

    direct = row_cos(out, gl)
    print(f"[direct]     per-row cos: mean={direct.mean():.4f}  min={direct.min():.4f}  frac(cos>0.99)={(direct > 0.99).float().mean():.4f}")

    out_n = out / out.norm(dim=1, keepdim=True).clamp(min=1e-12)
    gf_n = gf / gf.norm(dim=1, keepdim=True).clamp(min=1e-12)
    sim = out_n @ gf_n.T                       # (m_local, M_full)
    best_cos, best_idx = sim.max(dim=1)
    print(f"[best-match] per-row cos: mean={best_cos.mean():.4f}  min={best_cos.min():.4f}  frac(cos>0.99)={(best_cos > 0.99).float().mean():.4f}")
    print(f"[best-match] matched indices unique={best_idx.unique().numel()}/{out.shape[0]}")
    print(f"[best-match] first 16 matched idx: {best_idx[:16].tolist()}")
    expected_start = blob["rank"] * out.shape[0]
    identity = (best_idx == torch.arange(expected_start, expected_start + out.shape[0])).float().mean()
    print(f"[best-match] frac(idx == rank-slice identity)={identity:.4f}")

    # 数值假设的快速检验：整体缩放因子（H3 常表现为固定 scale 偏差）
    scale = (out.flatten() @ gl.flatten()) / (gl.flatten() @ gl.flatten()).clamp(min=1e-12)
    print(f"[H3 check]   least-squares scale(output vs golden_local)={scale:.4f} (1.0 = no scaling issue)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

**读数规则**：
- `direct` 低 + `best-match` 高且 idx 几乎双射 → **H1 排列**：记录 `first 16 matched idx` 的模式（比如按 expert 分段递增 → 输出是"按专家排序"的顺序，没散回 token 原序）。
- `best-match` 也低 → **H2/H3 数值**：看 `[H3 check]` 的 scale——若接近某常数（如 0.5、topk 权重和），是权重应用问题；否则按 B2-4 逐算子对分。
- `output |max|` 远小于 `golden |max|` → H2 部分和。

### B2-4 修复方向

- **H1 确认后**：修在 scheme 侧（计时语义要求方案返回"本 rank token 原序的最终结果"，散回原序是方案应付的成本）。对照 fork `{{TD_FORK}}/function/nvidia/ep_moe_fused.py` 中 forward 返回前对 `reversed_token_scatter_idx` 的使用，确认迁移版 `moe_bench/tdx/function/ep_moe_fused.py` 是否漏了最后的 scatter/gather；若 fork 本身就返回专家序，则在 `moe_bench/schemes/td_common.py` 的 `_run_ep_bf16` / `_run_ep_fp8` 返回前补一次 `output.index_copy_`/`gather`（用 dump 里确认的索引方向，先在 debug_tiny 上验证再上 v1）。
- **H2/H3 确认后**：单卡跑（debug_tiny 改 `nproc: 1`、`cuda_visible_devices: "9"`，无通信路径），在 `moe_bench/tdx/function/ep_moe_fused.py` 的 forward 里按 GEMM1 后 → SiLU 后 → GEMM2 后 → combine 后四个断点分别 `torch.save`，与用 golden 同法手算的中间量对比，锁定第一个偏离点再修。
- 每改一步都先跑 `configs/debug_tiny.yaml` 确认，再上 v1。

### B2-5 验收

```bash
python -m moe_bench.cli configs/smoke.yaml --isolate-schemes --set "schemes.enabled=[c1,c2,c3,c4,c5]"
```

c1/c2 对 `golden_bf16`、c3 对 rowwise、c4/c5 对 group128 全部过 B1 新门。产出 `EXP-022-cgroup-correctness.md`（假设→证据→修复位置→修复后校验表），并在 EXP-001 文首加修正声明。

---

## B3 a1 新旧对账（可与 B4 并行）

### B3-1 先对配置表（很可能差异就在这，先查再跑）

在 EXP-023 里填这张表，逐项从两边实际生效处抄（老 bench 从 run.sh 的 env 默认值 + 启动打印抄；新 bench 从 `results/<run>/config.resolved.yaml` 抄）：

| 项 | 老 bench 生效值 | 新 bench 生效值 | 一致? |
|---|---|---|---|
| M / K / E / top_k | | 5120 / 4096 / 64 / 8 | |
| N_GATEUP（完整维） | | 6144 | |
| N_DOWN | | 3072 | |
| shared expert 开关 / SHARED_INTERMEDIATE | | 1 / 3072 | |
| GPU 子集 | ⚠ 老报告 COMET 的 a1=7.63 可能测于 0-3 或 4-7 | 9,11,13,15 | |
| warmup / repeat | | 50 / 50 | |
| vLLM 版本 | | 0.1.0+cu128 | |
| CUDA_DEVICE_MAX_CONNECTIONS 等 env | | 见 config.resolved.yaml 的 env 段 | |

### B3-2 同卡同配置对跑

```bash
# 老 bench（只读运行，不改任何文件）
cd {{OLD_BENCH}}
CUDA_VISIBLE_DEVICES=9,11,13,15 GROUP_MODE=a1,b3 M=5120 N_GATEUP=6144 SHARED_INTERMEDIATE=3072 \
CUDA_DEVICE_MAX_CONNECTIONS=1 WARMUP=50 REPEAT=50 PRINT_DIAGNOSTICS=0 bash run.sh

# 新 bench（紧接着跑，避免机器负载漂移）
cd {{NEW_ROOT}}
python -m moe_bench.cli configs/run_v1_production.yaml --set "schemes.enabled=[a1,b3]"
```

### B3-3 若配置对齐后差距仍 >5%：抓 profile 对比

新 bench 侧启用 torch profiler：`config.py` 里 `ProfileCfg.torch_profiler` 存在但 worker 未实现，按下面接线。

`worker.py` 中 **原代码**：

```python
            summary, samples, last_output = bench_cuda(
                instance.run, warmup=cfg.warmup, repeat=cfg.repeat
            )
```

**新代码**（计时照旧，计时结束后额外抓 3 步 trace，不污染计时数据）：

```python
            summary, samples, last_output = bench_cuda(
                instance.run, warmup=cfg.warmup, repeat=cfg.repeat
            )

            if cfg.profile.torch_profiler:
                trace_dir = run_dir / "profiles" / "torch"
                trace_dir.mkdir(parents=True, exist_ok=True)
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                ) as prof:
                    for _ in range(3):
                        instance.run()
                    torch.cuda.synchronize()
                prof.export_chrome_trace(str(trace_dir / f"{code}_rank{ctx.rank}.json"))
                if ctx.rank == 0:
                    (trace_dir / f"{code}_top_kernels.txt").write_text(
                        prof.key_averages().table(sort_by="cuda_time_total", row_limit=20),
                        encoding="utf-8",
                    )
```

启用方式：`--set profile.torch_profiler=true`。老 bench 侧用 nsys 或它自带的 profile 脚本抓一份，两边各取 top-10 CUDA kernel 时长表并排，找"新侧少了/快了什么"（重点：NCCL AG/RS kernel 时长、fused_moe kernel 时长两段）。

### B3-4 验收

产出 `EXP-023-a1-reconciliation.md`：配置对表 + 同卡四数（老/新 × a1/b3）+ 差异解释 + **明确写出报告采用的基线口径**。完成前在 `docs/report/final_report.md` §4 表格上方加一行"⚠ 基线待 EXP-023 对账"。

---

## B4 c 组性能重查（可与 B3 并行）

### B4-0 先修文档

在 `EXP-020-tuning-analysis.md` 文首加：

```markdown
> ⚠ 本文的根因分析已被 REWORK-B 推翻：fork 对 `kernels/nvidia/group_gemm.py` 零改动
> （仅新增 `fp8_allgather_group_gemm.py`）；所引 commit `3653b8c`/`69ab14b` 修改的是
> 已迁移进 tdx 的 `function/nvidia/ep_moe_fused.py`。真实根因见 EXP-024。原文保留如下（superseded）。
```

### B4-1 环境变量对表（对表，不要假设）

注意：新框架的 YAML `env:` 段（见 `configs/run_v1_production.yaml`）**已经**设置了 `CUDA_DEVICE_MAX_CONNECTIONS=1`、`NVSHMEM_DISABLE_CUDA_VMM=1`、`NVSHMEM_SYMMETRIC_SIZE=8589934592`、`NVSHMEM_REMOTE_TRANSPORT=none`，并经 `cli.py::env_for_run` 注入子进程。所以不要照搬"环境变量没设"的结论，做法是：

1. 打开 EXP-020 那次运行的 `results/<run>/manifest.json`（含 effective_env）逐项和老 run.sh 的 export 清单对表。
2. 已知两处差异必须验证：老 run.sh 有 `NVSHMEM_BOOTSTRAP=UID`（新配置缺）；新配置多了 `NVSHMEM_REMOTE_TRANSPORT=none`（老侧是否设置待查）。把这两项加/去掉各跑一次 c1 smoke，记录延迟变化：

```bash
python -m moe_bench.cli configs/smoke.yaml --isolate-schemes \
  --set "schemes.enabled=[c1]" --set env.NVSHMEM_BOOTSTRAP=UID
```

### B4-2 同卡老 bench c1 参照数

```bash
cd {{OLD_BENCH}}
CUDA_VISIBLE_DEVICES=9,11,13,15 GROUP_MODE=c1 M=5120 N_GATEUP=6144 SHARED_INTERMEDIATE=3072 \
CUDA_DEVICE_MAX_CONNECTIONS=1 NVSHMEM_DISABLE_CUDA_VMM=1 NVSHMEM_BOOTSTRAP=UID \
NVSHMEM_SYMMETRIC_SIZE=8589934592 WARMUP=50 REPEAT=50 PRINT_DIAGNOSTICS=0 bash run.sh
```

老报告 c1=9.5ms 的卡号出处不明，**必须**重测同卡数字作对账目标；老 bench 用的是 fork 安装态，这个数就是"fork 运行态"口径。

### B4-3 模块身份审计（排除"意外用了 upstream 版本"）

在 `moe_bench/schemes/td_common.py` 的 `build_td_instance` 内 `run()` 函数里，**原代码**：

```python
    def run() -> Any:
        from moe_bench.tdx.compat import apply

        apply()
```

**新代码**（环境变量触发的一次性审计打印）：

```python
    def run() -> Any:
        from moe_bench.tdx.compat import apply

        apply()
        if os.environ.get("MOE_BENCH_TDX_AUDIT") == "1":
            import sys as _sys
            for name in sorted(n for n in _sys.modules if n.startswith(("triton_dist", "moe_bench.tdx"))):
                print(f"TDX-AUDIT {name} -> {getattr(_sys.modules[name], '__file__', None)}", flush=True)
```

文件顶部补 `import os`（该文件当前只有 `import importlib`）。

跑法与判定：

```bash
MOE_BENCH_TDX_AUDIT=1 python -m moe_bench.cli configs/debug_tiny.yaml 2>&1 | grep TDX-AUDIT | sort -u
```

对照 `docs/tdx_migration_manifest.json` 里的迁移清单：**每个被 fork 修改过的模块名，其生效 `__file__` 必须落在 `moe_bench/tdx/` 下**；若同名逻辑同时从 `triton_dist.*` 加载并被 layer 引用（例如 `ep_all2all_fused` 出现 upstream 路径且被使用），就是迁移漏接，直接修对应 import。

再验证 GEMM 常量真实生效：`grep -n "BLOCK_SIZE_N\|BLOCK_SIZE_K\|num_stages\|NUM_SM\|num_sm" moe_bench/tdx/function/ep_moe_fused.py moe_bench/tdx/layers/ep_a2a_fused_layer.py | head -30`，找到实际控制 GEMM/通信 SM 的常量或参数定义处，在其使用点加一条一次性 `print`（同样用 `MOE_BENCH_TDX_AUDIT` 门控），确认运行时值为 fork 默认（BF16 EP：BLOCK_N=256 / BLOCK_K=64 / stages=3 / num_sm=110）。

### B4-4 分段计时

两条路，选一条：
- **优先**：`moe_bench/tdx/layers/ep_a2a_fused_layer.py:50` 已 import fork 的 `ProfilerBuffer, export_to_perfetto_trace`。在 fork 源码里 `grep -rn "ProfilerBuffer" {{TD_FORK}}/layers/nvidia/ep_a2a_fused_layer.py` 找它的启用开关（构造参数或环境变量），照 fork 的用法开起来，导出 perfetto trace。
- **兜底**：用 B3-3 加的 torch profiler（`--set profile.torch_profiler=true`），从 chrome trace 里按 kernel 名把 c1 单步拆成 preprocess / dispatch / GEMM1 / SiLU / GEMM2 / combine 六段。

老（B4-2 fork 态，用老 bench 自带 profile 或 nsys）新（tdx 态）各一份六段表。3 倍差距必然集中在某一两段；锁定后 `diff {{NEW_ROOT}}/moe_bench/tdx/kernels/ep_all2all_fused.py {{TD_FORK}}/kernels/nvidia/ep_all2all_fused.py`（以及对应段的其他文件）——**tdx 副本与 fork 同名文件的 diff 除 import 行外应为空**；若不为空就是迁移错漏，直接修。

### B4-5 验收

`EXP-024-cgroup-perf-rootcause.md`：每步数字 + 根因证据链。结果二选一并明确写出：(a) c1/c3 关闭到同卡 fork 态数字的 ±20% 内；(b) 无法关闭，但有分段计时证据支撑根因，报告改为"upstream 运行态 vs fork 运行态"双口径分开呈现，不得混排。

---

## B5 补齐 microbench

现状：`mb_comm_prims.py`、`mb_quant_kernels.py` 是真实现；其余 5 个是 `run_microbench_plan(...)` 桩。下面给出其中 4 个的完整实现（整文件替换）和 1 个的骨架。公共约定：输出 jsonl 一行一个 case，字段与 `mb_comm_prims.py` 同风格；计时一律 CUDA event，warmup 后 `torch.cuda.synchronize()`。

### B5-1 `microbench/mb_moe_align.py`（EXP-010，整文件替换）

```python
"""Dispatch pre-compute cost: argsort x2, bincount, cumsum vs (M, E, top_k)."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch


def bench_gpu(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return statistics.median(starts[i].elapsed_time(ends[i]) for i in range(repeat))


def main() -> int:
    parser = argparse.ArgumentParser(description="MoE dispatch pre-compute cost")
    parser.add_argument("--output", default="results/microbench_moe_align.jsonl")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for M in [1024, 5120]:
        for E in [64, 128]:
            for top_k in [8, 16]:
                gen = torch.Generator(device=device).manual_seed(42)
                topk_ids = torch.randint(0, E, (M, top_k), generator=gen, device=device, dtype=torch.int32)
                flat = topk_ids.flatten()

                t_sort1 = bench_gpu(lambda: torch.argsort(flat, stable=True), args.warmup, args.repeat)
                order = torch.argsort(flat, stable=True)
                t_sort2 = bench_gpu(lambda: torch.argsort(order, stable=True), args.warmup, args.repeat)
                t_bincount = bench_gpu(lambda: torch.bincount(flat, minlength=E), args.warmup, args.repeat)
                counts = torch.bincount(flat, minlength=E)
                t_cumsum = bench_gpu(lambda: torch.cumsum(counts, dim=0), args.warmup, args.repeat)

                row = {
                    "kind": "microbench", "script": "mb_moe_align.py", "exp_id": "EXP-010",
                    "M": M, "E": E, "top_k": top_k,
                    "argsort1_ms": t_sort1, "argsort2_ms": t_sort2,
                    "bincount_ms": t_bincount, "cumsum_ms": t_cumsum,
                    "total_ms": t_sort1 + t_sort2 + t_bincount + t_cumsum,
                }
                rows.append(row)
                print(json.dumps(row))

    with output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Results written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

单卡即可：`python microbench/mb_moe_align.py`（先 `export CUDA_VISIBLE_DEVICES=9`）。

### B5-2 `microbench/mb_dispatch_prep.py`（EXP-011，整文件替换）

量化 preprocess 链中"host 同步点"的代价——这是 tile-level 路线的固定票价之一：

```python
"""Dispatch preprocess cost: token permutation gather + pinned-host readback sync."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description="dispatch prep and host-sync cost")
    parser.add_argument("--output", default="results/microbench_dispatch_prep.jsonl")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for M, K, top_k in [(1024, 4096, 8), (5120, 4096, 8), (4096, 3200, 16)]:
        gen = torch.Generator(device=device).manual_seed(42)
        hidden = torch.randn(M, K, generator=gen, device=device, dtype=torch.bfloat16)
        perm = torch.randperm(M * top_k, generator=gen, device=device)

        # (a) token permutation gather：把 token 体按调度序重排的纯带宽成本
        def gather():
            return hidden[perm % M]

        for _ in range(args.warmup):
            gather()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeat)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeat)]
        for i in range(args.repeat):
            starts[i].record()
            gather()
            ends[i].record()
        torch.cuda.synchronize()
        gather_ms = statistics.median(starts[i].elapsed_time(ends[i]) for i in range(args.repeat))

        # (b) pinned-host 回读 + 同步：模拟 num_recv_tokens 的 CPU 读点（含整条流 flush 的代价）
        small = torch.randint(0, M, (8,), device=device, dtype=torch.int32)
        pinned = torch.empty(8, dtype=torch.int32, pin_memory=True)
        # 用一个中等 kernel 垫在前面，模拟"流上还有活时被迫等待"
        filler_a = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
        filler_b = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)

        sync_samples = []
        for _ in range(args.warmup):
            torch.mm(filler_a, filler_b)
            pinned.copy_(small, non_blocking=True)
            torch.cuda.synchronize()
        for _ in range(args.repeat):
            torch.mm(filler_a, filler_b)
            t0 = time.perf_counter()
            pinned.copy_(small, non_blocking=True)
            torch.cuda.synchronize()
            sync_samples.append((time.perf_counter() - t0) * 1000.0)
        host_sync_ms = statistics.median(sync_samples)

        row = {
            "kind": "microbench", "script": "mb_dispatch_prep.py", "exp_id": "EXP-011",
            "M": M, "K": K, "top_k": top_k,
            "token_gather_ms": gather_ms,
            "host_readback_sync_ms": host_sync_ms,
        }
        rows.append(row)
        print(json.dumps(row))

    with output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Results written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

说明写进 EXP-011：`host_readback_sync_ms` 含被垫入 GEMM 的排空时间，反映的是"host 同步点必须等流排空"的真实代价上界；纯 D2H 拷贝本身只有几 µs。（可选扩展：4 卡 NVSHMEM 环境下直接对 `kernel_get_ag_splits_and_recv_offset` 计时，需要初始化 nvshmem，做不出来不算阻塞。）

### B5-3 `microbench/mb_group_gemm.py`（EXP-014，整文件替换）

用 vLLM `fused_experts` 单卡测"token 分布不均 → GroupGEMM 效率"：

```python
"""FP8 GroupGEMM efficiency vs token-to-expert distribution (uniform/zipf/hotspot)."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch


def make_topk(M: int, E: int, top_k: int, dist_kind: str, device) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(42)
    if dist_kind == "uniform":
        probs = torch.ones(E, device=device)
    elif dist_kind == "zipf":
        ranks = torch.arange(1, E + 1, device=device, dtype=torch.float32)
        probs = ranks.pow(-1.2)
    elif dist_kind == "hotspot":
        probs = torch.ones(E, device=device)
        # 4 个热点专家合计 60% 流量：hot/(hot+cold)=0.6 → 每个热点权重 = 1.5*(E-4)/4
        probs[:4] = 1.5 * (E - 4) / 4
    else:
        raise ValueError(dist_kind)
    probs = probs / probs.sum()
    topk_ids = torch.multinomial(probs.expand(M, E).contiguous(), top_k, replacement=False, generator=gen).to(torch.int32)
    topk_weights = torch.softmax(torch.randn(M, top_k, generator=gen, device=device), dim=-1)
    return topk_ids, topk_weights


def main() -> int:
    parser = argparse.ArgumentParser(description="GroupGEMM vs token distribution")
    parser.add_argument("--output", default="results/microbench_group_gemm.jsonl")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()

    from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    from moe_bench.config import ShapeCfg
    from moe_bench.data.checkpoint import build_checkpoint

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    shape = ShapeCfg(M=5120, K=4096, E=64, top_k=8, n_gateup=1536, n_down=768, shared_experts=0)
    ckpt = build_checkpoint(shape, seed=42, device=device)
    quant_config = fp8_w8a8_moe_quant_config(
        w1_scale=ckpt.w1.scale, w2_scale=ckpt.w2.scale, block_shape=[128, 128],
    )

    for M in [1024, 5120]:
        gen = torch.Generator(device=device).manual_seed(7)
        hidden = torch.randn(M, shape.K, generator=gen, device=device, dtype=torch.bfloat16)
        for dist_kind in ["uniform", "zipf", "hotspot"]:
            topk_ids, topk_weights = make_topk(M, shape.E, shape.top_k, dist_kind, device)

            def step():
                return fused_experts(
                    hidden_states=hidden, w1=ckpt.w1.fp8, w2=ckpt.w2.fp8,
                    topk_weights=topk_weights, topk_ids=topk_ids,
                    global_num_experts=shape.E, quant_config=quant_config,
                )

            for _ in range(args.warmup):
                step()
            torch.cuda.synchronize()
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeat)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeat)]
            for i in range(args.repeat):
                starts[i].record()
                step()
                ends[i].record()
            torch.cuda.synchronize()
            med = statistics.median(starts[i].elapsed_time(ends[i]) for i in range(args.repeat))

            counts = torch.bincount(topk_ids.flatten().long(), minlength=shape.E).float()
            row = {
                "kind": "microbench", "script": "mb_group_gemm.py", "exp_id": "EXP-014",
                "M": M, "distribution": dist_kind,
                "latency_ms": med,
                "imbalance_max_over_mean": float(counts.max() / counts.mean()),
                "cv": float(counts.std() / counts.mean()),
            }
            rows.append(row)
            print(json.dumps(row))

    with output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Results written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

注意 shape 用了 `n_gateup=1536`（=E 独立小专家的典型规格），使单卡显存放得下 64 专家全量权重；EXP-014 里注明这一点。

### B5-4 `microbench/mb_chunk_overlap.py`（EXP-016，整文件替换）

做法：不重写 overlap 逻辑，直接驱动现有 bench 扫 chunk 数，然后结合 EXP-013 的通信带宽算 η：

```python
"""Chunk-count sweep for b-group overlap schemes; computes overlap efficiency eta.

eta = (T_a1 - T_b) / T_comm_est
  T_a1:      serial baseline latency at the same M (measured in the same sweep)
  T_b:       b-scheme latency at given n_chunks
  T_comm_est: AG+RS wall time estimated from mb_comm_prims data at the same byte sizes
exposed_comm_ms = T_b - (T_a1 - T_comm_est)   # how much comm is still on the critical path
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def latest_results(root: Path) -> Path:
    runs = sorted(p for p in root.iterdir() if p.is_dir())
    return runs[-1] / "results.jsonl"


def read_med(results: Path, scheme: str) -> float:
    for line in results.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["scheme"] == scheme:
            return row["lat_ms"]["med"]
    raise RuntimeError(f"{scheme} not in {results}")


def comm_ms(comm_jsonl: Path, total_bytes: int, field: str) -> float:
    rows = [json.loads(line) for line in comm_jsonl.read_text(encoding="utf-8").splitlines()]
    best = min(rows, key=lambda r: abs(r["total_bytes"] - total_bytes))
    # 按最近点的带宽折算到目标字节数
    bw = best[field]  # GB/s
    world = best["world_size"]
    return (total_bytes * (world - 1) / world) / (bw * 1e9) * 1000.0


def main() -> int:
    parser = argparse.ArgumentParser(description="b-group chunk sweep with eta")
    parser.add_argument("--config", default="configs/run_v1.yaml")
    parser.add_argument("--comm-jsonl", default="results/microbench_comm_prims.jsonl")
    parser.add_argument("--output", default="results/microbench_chunk_overlap.jsonl")
    parser.add_argument("--out-root", default="results/mb_chunk_overlap")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for M in [1024, 5120]:
        K = 4096
        # 基线 a1
        subprocess.run([sys.executable, "-m", "moe_bench.cli", args.config,
                        "--set", "schemes.enabled=[a1]",
                        "--set", f"shape.M={M}",
                        "--set", f"run.output_root={out_root}",
                        "--set", "run.tag=a1_base"], check=True)
        t_a1 = read_med(latest_results(out_root), "a1")

        bytes_bf16 = M * K * 2
        bytes_fp8 = M * K * 1 + (M * K // 128) * 4      # fp8 体 + fp32 scale
        for scheme in ["b1", "b2", "b3"]:
            ag_bytes = bytes_bf16 if scheme == "b1" else bytes_fp8
            rs_bytes = bytes_fp8 if scheme == "b3" else bytes_bf16
            t_comm = (comm_ms(Path(args.comm_jsonl), ag_bytes, "ag_bandwidth_gbps")
                      + comm_ms(Path(args.comm_jsonl), rs_bytes, "rs_bandwidth_gbps"))
            for n_chunks in [1, 2, 4, 8, 16]:
                subprocess.run([sys.executable, "-m", "moe_bench.cli", args.config,
                                "--set", f"schemes.enabled=[{scheme}]",
                                "--set", f"shape.M={M}",
                                "--set", f"schemes.{scheme}.tunables.n_chunks_gateup={n_chunks}",
                                "--set", f"schemes.{scheme}.tunables.n_chunks_down={n_chunks}",
                                "--set", f"run.output_root={out_root}",
                                "--set", f"run.tag={scheme}_c{n_chunks}"], check=True)
                t_b = read_med(latest_results(out_root), scheme)
                eta = (t_a1 - t_b) / t_comm if t_comm > 0 else 0.0
                row = {
                    "kind": "microbench", "script": "mb_chunk_overlap.py", "exp_id": "EXP-016",
                    "M": M, "scheme": scheme, "n_chunks": n_chunks,
                    "latency_ms": t_b, "a1_latency_ms": t_a1,
                    "comm_est_ms": t_comm,
                    "overlap_efficiency_eta": eta,
                    "exposed_comm_ms": t_b - (t_a1 - t_comm),
                }
                rows.append(row)
                print(json.dumps(row))

    with Path(args.output).open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Results written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

前置：`configs/run_v1.yaml` 存在且 EXP-013 的 `results/microbench_comm_prims.jsonl` 在位。η 为负说明该配置下 overlap 净亏（保留原始负值写入 jsonl，分析时按 0 截断另算一列）。

### B5-5 `microbench/mb_tile_order_groupgemm.py`（EXP-012，骨架，需补全）

这是唯一不给完整代码的脚本，因为它要调 upstream 的 grouped GEMM 内核，签名需要现场核对：

1. 方法论沿用老仓库根目录《Tile 排序对 fused_moe GEMM 性能影响 — β' 重排代价定位与优化（半 phase 分组）》：同一批 (token, expert) 数据，只改 tile 处理顺序，量 GroupGEMM 时长差。
2. 实现路径：`{{TD_UPSTREAM}}/python/triton_dist/kernels/nvidia/group_gemm.py` —— 用 `build_block_row_idx_info_kernel`（:40 附近）生成 tile→expert 元数据后，构造三种 tile 顺序数组：①专家有序（默认）②随机打乱（模拟到达序）③半 phase 分组（老文档的优化排法），分别喂给 `moe_grouped_gemm` 计时。先 `grep -n "def moe_grouped_gemm" group_gemm.py` 核对入参再写。
3. shape 用 UND（E=128, K=2048, n_gateup=1536）与 GEN（E=128, K=3200, top_k=16）两组；warmup≥20、repeat≥50。
4. sanity 锚点：老仓库实测 UND 三种排序 322.8 / 401.0 / 329.7 µs——新测的**相对关系**（乱序明显最慢、半 phase 接近有序）必须复现，绝对值允许不同（卡/驱动不同）。相对关系不复现就停下写 BLOCKERS。

### B5-6 验收

5 个脚本各产出 jsonl + `EXP-010/011/012/014/016` 文档，INDEX 更新。**每份 EXP 文档必须有"这组数字解释了什么问题"一段**（对应经验分享 §6 的三问法）。

---

## B6 经验分享数据采集（B1–B4 绿后）

```bash
cd {{NEW_ROOT}}
# M sweep：一次跑完 7 个 M 点（sweep_axes 会展开成 7 个 point）
python -m moe_bench.cli configs/run_v1_production.yaml --isolate-schemes \
  --set "schemes.enabled=[a1,a2,b1,b2,b3,c1,c3]" \
  --set "run.tag=sweep_v1_share" \
  --set 'sweep_axes=[{"path":"shape.M","values":[128,512,1024,2048,3072,4096,5120]}]'
```

（`--set` 的值经 `yaml.safe_load` 解析，上面的 JSON 列表写法可被解析为 sweep_axes 结构；若解析失败，就直接复制 `run_v1_production.yaml` 为 `configs/sweep_v1_share.yaml` 手工加 `sweep_axes:` 段。）

之后为每个 M 计算"通信占比"列：AG bytes = M×K×2，RS 同；用 EXP-013 带宽折算 T_comm，占比 = T_comm / T_a1(M)。汇总表进 `docs/report/`。c4/c5 若 B2/B4 后可用则加入 enabled 列表重跑。

---

## B7 文档与工程收尾

### B7-1 补 EXP-013 / EXP-015 文档

数据已在 `results/microbench_comm_prims.jsonl`、`results/microbench_quant_kernels.jsonl`，按 EXP-TEMPLATE 各写一份（EXP-013 重点：32 GB/s 平台数、半带宽点约在 1–4MB；EXP-015 重点：小 M 时固定 ~19–20µs 启动地板、大 M 逼近带宽上限）。INDEX 里这两行目前是坏链。

### B7-2 修 `worker.py` shard_level 硬编码

`tp_weight_views`（`runtime.py`）其实已经算出 `shard_level`（L0/L2）并放进返回 dict，但 worker 拿不到。最小改法：从配置推导（与 `tp_weight_views` 的判定规则一致）。

`worker.py` 中 **原代码**：

```python
        "shard_level": "L1",
```

**新代码**：

```python
        "shard_level": _shard_level(cfg, scheme_cfg.code),
```

并在文件底部（`_diagnostics_line` 之前或之后均可）加：

```python
def _shard_level(cfg: RunCfg, scheme_code: str) -> str:
    spec = REGISTRY[scheme_code][0]
    if getattr(spec, "parallel", "TP") == "EP":
        return "L0"          # EP 按专家维切分，永不破坏 128 块
    per_rank_intermediate = cfg.shape.n_down // cfg.dist.nproc
    return "L0" if per_rank_intermediate % 128 == 0 else "L2"
```

### B7-3 修过时测试 `tests/test_a5_handoff.py`

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

**新代码**（实现完一个脚本就把它挪进 IMPLEMENTED；B5 做完后 PENDING 应为空列表）：

```python
IMPLEMENTED_MICROBENCH = [
    "mb_comm_prims.py",
    "mb_quant_kernels.py",
    # B5 完成后加入：mb_moe_align.py / mb_dispatch_prep.py / mb_group_gemm.py
    # / mb_chunk_overlap.py / mb_tile_order_groupgemm.py
]
PENDING_MICROBENCH = [
    "mb_moe_align.py",
    "mb_dispatch_prep.py",
    "mb_tile_order_groupgemm.py",
    "mb_group_gemm.py",
    "mb_chunk_overlap.py",
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

（B5 每完成一个脚本：删掉脚本 docstring 里的 `SERVER-VERIFY` 字样，并把文件名从 PENDING 挪到 IMPLEMENTED。）

### B7-4 其余收尾

1. `HANDOFF.md`：S0–S7 清单标注完成状态与对应 EXP 编号；SERVER-VERIFY Summary 表删掉已验证行。
2. `BLOCKERS.md`：c 组 warmup 一条改为"已由 EXP-020(superseded)/EXP-024 处理"；und/gen 一条按最新进展改写。
3. gen shape 补全：`python -m moe_bench.cli configs/sweep_gen.yaml --isolate-schemes`（目前 gen 只有 b1 单点）。
4. `docs/report/final_report.md`：数据章节只允许引用 B1 之后产生的 results 目录；老 bench 对照数标注出处（EXP-023/024）；删除"⚠ 待对账"标注。
5. `docs/exp/INDEX.md` 全量刷新。

---

## 完成判据总表

| # | 判据 | 证明物 |
|---|---|---|
| B1 | 全零输出 FAIL 单测绿；a/b 组过校准后的新门；输出量级 O(0.1~10) | EXP-021 + tests/test_verify_gate.py |
| B2 | c1–c5 v1 全过新门 | EXP-022 + results |
| B3 | 同卡同配置四数对表 + 差异解释 + 基线口径拍板 | EXP-023 |
| B4 | c 组差距根因有分段计时证据；关闭或双口径 | EXP-024 |
| B5 | 5 脚本真实现 + jsonl + EXP 文档 + test_a5 状态表更新 | EXP-010/011/012/014/016 |
| B6 | M sweep + 通信占比表落盘 | results/<ts>_sweep_v1_share |
| B7 | 坏链清零、shard_level 修正、HANDOFF/INDEX/BLOCKERS/report 同步 | git log + 文档 |

全部绿后，通知负责人启动经验分享文档（`{{BENCH_ROOT}}/TILE_OVERLAP_SHARE_OUTLINE.md`）与最终报告的数字填充。
