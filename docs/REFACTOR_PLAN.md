# MoE Bench v2 重构方案（架构设计 + 执行计划）

> 文档角色：本文档是重构的**唯一权威方案**，由总架构师撰写，交给执行工程师（下称"执行者"）实施。
> 执行分**两个阶段**：**阶段一（本地，无 GPU）**在 `{{NEW_ROOT}}` 下完成全部代码重构与交接材料；**阶段二（服务器，有 GPU）**完成测试、验证、微基准、调优与报告。阶段划分与任务清单见 §8。
> 执行者必须按本方案的模块划分、接口定义和 Phase 顺序实施，**不得在旧代码基础上修改**；旧代码只作为行为参考（见 §13 旧代码索引）。
> 遇到方案没有覆盖的决策点：先在 `docs/exp/BLOCKERS.md` 记录问题与候选方案，再向用户请示，不要自行拍板绕过。

---

## 0. 路径定义（执行前核对）

本方案用占位符指代路径，当前取值如下（按本机现有路径填写）。若执行环境路径不同，先更新此表再全文替换：

| 占位符 | 含义 | 当前值 |
|---|---|---|
| `{{BENCH_ROOT}}` | 旧 moe_bench 仓库根目录（只读参考） | `C:\Users\stmoonar\Desktop\files\moe_bench_0607` |
| `{{NEW_ROOT}}` | **重构产出根目录（阶段一全部代码写在这里）** | `C:\Users\stmoonar\Desktop\files\moe_bench_0607\new_start` |
| `{{TD_FORK}}` | 我们修改过的 Triton-distributed fork | `C:\Users\stmoonar\Desktop\files\forks\Triton-distributed` |
| `{{TD_UPSTREAM}}` | 干净的上游 Triton-distributed（checkout 到 `1b9dc71a`）| `C:\Users\stmoonar\Desktop\files\forks\Triton-distributed-upstream`（本地可 clone 供 tdx 搬运比对；服务器上 S0 再 clone 一份运行用） |
| `{{PYTHON}}` | 运行环境 python | `python3` |
| `{{CUDA_HOME}}` | CUDA 安装目录（服务器） | `/usr/local/cuda` |

阶段二开始前，`{{NEW_ROOT}}` 整体同步到服务器；服务器上的真实路径写进 `{{NEW_ROOT}}/HANDOFF.md` 的路径表（见 §8 阶段一交接包），全文占位符以那张表为准更新。

TD fork 的基线 commit 固定为 `1b9dc71a0a58585ac99766d739bf08ec60de4ae7`，我们的全部改动 = `git diff 1b9dc71a..HEAD` + 未提交改动。

---

## 1. 目标与非目标

### 1.1 目标

1. **公平对比** 10 种 MoE 单层方案（串行 / chunk 级 overlap / tile 级通算融合 × TP/EP × BF16/FP8）在 4×RTX Pro 5000 (sm120, PCIe) 上的性能。
2. **输入完全统一**：同一份 hidden、同一个 gate 算出的路由、同一份 128×128 block 量化的 FP8 权重 checkpoint；所有方案输出与纯 PyTorch golden 逐元素可验证。
3. **分散度可配置**：token→expert 分布从均匀到极端热点可调，作为 sweep 维度。
4. **triton_dist 解耦**：fork 里我们写的 FP8 EP/TP 代码与调优配置全部搬进本仓库（`moe_bench/tdx/`），运行只依赖干净的上游 `{{TD_UPSTREAM}}`。
5. **可分析**：stage 级计时、微基准、统一的 profile 产物目录，支撑瓶颈定位（dispatch 前重排、tile 排序、FP8 通信盈亏点等）。
6. **可回溯**：每个实验留下标准化 EXP 文档；最终产出一份可直接做 PPT 的报告。
7. **工程质量**：单一 YAML 配置（告别一长串环境变量）、人类可读日志（可关、不影响计时）、去掉 /tmp 复制 hack、可插拔新增方案。

### 1.2 非目标

- 不做多层模型/端到端推理；只测单个 MoE 层 forward。
- 不做训练/反向。
- 不重写 vLLM / 上游 triton_dist 内核本身（tdx 搬入的除外）。
- 不追求跨机（单机 4/8 卡）。

### 1.3 三个标准形状（内置 preset）

| preset 名 | M | K | E | TOP_K | SHARED | N_GATEUP | N_DOWN(=intermediate) |
|---|---|---|---|---|---|---|---|
| `v1` | 5120 | 4096 | 64 | 8 | 1 | 6144 | 3072 |
| `und` | 6647 | 2048 | 128 | 8 | 1 | 1536 | 768 |
| `gen` | 4096 | 3200 | 128 | 16 | 1 | 1536 | 768 |

**语义统一（重要，与旧代码不同）**：
- `N_GATEUP` = 完整 fused gate+up 输出维度；`intermediate = N_GATEUP / 2`。
- `N_DOWN` = down 矩阵的**输入**维度 = intermediate；配置里允许写但必须等于 `N_GATEUP/2`，否则直接报错（gate/up 输出必须接得上 down 输入）。down 矩阵输出维度恒等于 K（residual 一致性）。旧代码里 `N_down` 指 down 输出维度（默认 =K），语义不同，迁移时注意。
- `M` 不整除 world_size 时向上对齐（6647→6648），日志里必须打印对齐动作。
- shared expert 的 intermediate 默认 = `N_DOWN`（"共享专家权重维度和路由专家一样"）。
- `und`/`gen` 在 TP=4 下 per-rank intermediate = 192，**不是 128 的倍数**，这是旧代码跑不起来的 scale 切分问题，解法见 §5.1，这是 S4 的核心任务。

---

## 2. 现状问题清单（重构动机，执行者需逐条对照避免复刻）

| # | 问题 | 新方案对策 |
|---|---|---|
| 1 | 权重路径不统一：`materialize_tp_weights` 先切 BF16 再重新量化，TP/EP 的 FP8 表示互不一致，也没有全局 checkpoint 概念 | §4.2 WeightCheckpoint：全局量化一次，切分策略分级（§5.1） |
| 2 | `prepare_moe_weights` 是 E×nN×nK 三重 Python 循环，E=128 时极慢 | 向量化 block 量化（§4.2 代码骨架） |
| 3 | 正确性基准是 a1（自身也是 FP8 实现），误差混入基准 | 纯 torch golden 双档（§5.3） |
| 4 | 路由分散度不可控 | gate bias 注入（§5.2） |
| 5 | 配置三套并存（run.sh 环境变量 + CLI + YAML），NVSHMEM/TRITON_DIST_* 等 env 散落 | 单一 YAML + `--set` 覆盖，launcher 统一注入 env（§5.5） |
| 6 | FP8 方案代码和调优参数写死在 triton_dist fork 里 | 全部搬入 `moe_bench/tdx/`，tile/SM/warp 成为 YAML tunables（§5.4） |
| 7 | run.sh 把代码复制到 /tmp 规避 import 冲突 | 规范包布局 + `python -m` 运行；若仓库根残留 `vllm/` 源码目录，移出仓库（§5.5） |
| 8 | 结果 JSONL 是"一行一个 M、列名拼 label"的宽表，难解析 | 一行 = (shape 点 × scheme) 的窄表（§5.7） |
| 9 | profile 产物散落（prof/、results/*/profiles、TD 写死 prof/mega/） | 统一 run 目录布局 + manifest（§5.7） |
| 10 | 诊断打印和计时耦合在主流程里 | logging 分级 + 诊断一律在计时区外（§5.6） |
| 11 | c2 文档标注 sm120 broken——实际 bug 已修复 | c2 正常纳入，迁移时更新文档 |

---

## 3. 总体架构与目录结构

全部新代码位于 `{{NEW_ROOT}}`（即旧仓库下的 `new_start/` 子目录，自成一个可整体拷贝到服务器的独立单元；旧代码留在 `{{BENCH_ROOT}}` 根部只读）：

```
{{NEW_ROOT}}/
├── HANDOFF.md                    # 阶段一→阶段二交接主文档（§8）
├── moe_bench/                    # 新 Python 包（本次重构的全部产出）
│   ├── __init__.py
│   ├── cli.py                    # 入口：解析 YAML+--set → 展开 sweep → 注入 env → 逐点 spawn torchrun
│   ├── worker.py                 # torchrun 进程主流程（原 benchmark.py 职责）
│   ├── config.py                 # 配置 schema（dataclass）+ 加载/合并/校验/快照
│   ├── context.py                # DistContext：init/finalize（NCCL 或 NVSHMEM）
│   ├── logging_util.py           # rank 感知日志
│   ├── timing.py                 # bench_cuda 计时循环、StageTimer、nvtx helpers
│   ├── verify.py                 # golden_bf16 / golden_fp8sim + 比较报告
│   ├── results.py                # JSONL sink、manifest、summary.md 生成
│   ├── data/
│   │   ├── checkpoint.py         # WeightCheckpoint：向量化 128×128 量化 + BF16 视图
│   │   ├── shard.py              # TP/EP 切分器（对齐三级策略）
│   │   ├── routing.py            # gate 路由 + imbalance 控制 + 分布统计
│   │   └── bundle.py             # DataBundle 组装（每个 rank 一份）
│   ├── schemes/
│   │   ├── __init__.py           # REGISTRY: dict[code, SchemeSpec]
│   │   ├── base.py               # SchemeSpec / SchemeInstance / 公共工具
│   │   ├── shared_expert.py      # StaticSharedExpert（重写自旧 data.py）
│   │   ├── vllm_tp.py            # a1
│   │   ├── vllm_ep.py            # a2
│   │   ├── overlap_bf16.py       # b1
│   │   ├── overlap_fp8ag.py      # b2
│   │   ├── overlap_fp8ag_fp8rs.py# b3
│   │   ├── td_ep_bf16.py         # c1
│   │   ├── td_tp_bf16.py         # c2
│   │   ├── td_ep_fp8.py          # c3
│   │   ├── td_tp_fp8.py          # c4
│   │   └── td_tp_fp8_rs.py       # c5
│   ├── overlap/                  # b 组内核（迁移自 moe_overlap/，见 §5.4.5）
│   └── tdx/                      # triton_dist fork 改动搬入（见 §5.4）
│       ├── compat.py             # 运行时兼容 monkeypatch（不改安装的库）
│       ├── kernels/              # ep_all2all_fused.py、fp8_*.py、swiglu_quantize_fp8.py …
│       ├── function/             # ep_moe_fused.py、common.py
│       └── layers/               # fp8_ep_moe.py、fp8_tp_moe.py、ep_moe.py、ep_a2a_fused_layer.py
├── configs/
│   ├── default.yaml              # 全量注释的模板
│   ├── smoke.yaml                # 小形状快速回归（M=256, warmup=3, repeat=5）
│   ├── sweep_v1.yaml / sweep_und.yaml / sweep_gen.yaml
│   └── tune_c3.yaml …            # 调优实验配置
├── tests/                        # pytest 正确性单测（§7.1）
├── microbench/                   # 独立 torchrun 分析脚本（§7.2）
├── docs/
│   ├── exp/                      # 实验留痕：INDEX.md + EXP-NNN-*.md（§10）
│   ├── methodology.md            # §9 落地成文（从本方案抽取后持续补充）
│   └── report/                   # 最终报告 + PPT 素材（§11）
└── results/                      # 运行产物（服务器阶段生成；gitignore，只留 summary 与 EXP 引用）
```

（`REFACTOR_PLAN.md` 本文档复制一份进 `{{NEW_ROOT}}/docs/`，与代码一起同步到服务器。）

分层依赖（只允许上层依赖下层）：

```
cli → worker → schemes → {data, tdx, overlap, vllm(外部), triton_dist(外部)}
                 ↓
        {config, context, timing, verify, results, logging_util}
```

旧的 `benchmark.py`、`benchmarks/`、`moe_overlap/`、`run.sh`、`bench_config.yaml` 在整个迁移期间**只读保留**（对照跑分用），S7 验收后删除。

---

## 4. 核心数据结构与抽象（接口定义）

以下代码骨架是接口契约，字段名、形状约定必须遵守；函数体实现细节执行者可发挥。

**两条为"本地无 GPU 开发"服务的硬性规则**：
1. **数据层与 golden 必须 device 无关**：`data/` 与 `verify.py` 中所有张量操作不写死 `cuda`，device 一律由参数传入。这样量化/切分/路由/golden 的单测在本地 CPU 就能跑（阶段一即可验证），服务器上同一套测试再以 CUDA 跑一遍。
2. **外部重依赖延迟导入**：`vllm`、`triton_dist`、`triton` 只允许在 scheme builder / tdx 函数体内部 import（旧代码已是此模式，保持）。验收标准：在没有安装这三者的本地环境，`import moe_bench` 及 config/data/verify/results 模块全部可导入、CPU 单测可运行。

### 4.1 配置模型（`config.py`）

```python
@dataclass(frozen=True)
class ShapeCfg:
    M: int; K: int; E: int; top_k: int
    n_gateup: int                    # 完整 fused gate+up 维度
    n_down: int                      # = intermediate = n_gateup // 2（构造时校验）
    shared_experts: int              # 0 或 1
    shared_intermediate: int         # 默认 = n_down
    # 派生属性: intermediate, gateup_per_tp(ws), intermediate_per_tp(ws),
    #           local_E(ws), M_aligned(ws)

@dataclass(frozen=True)
class RoutingCfg:
    gate_seed: int = 42
    imbalance_kind: str = "none"     # none | zipf | hotspot
    zipf_s: float = 1.2
    hot_experts: int = 4
    hot_share: float = 0.6
    strength: float = 1.0            # bias 温度 τ

@dataclass(frozen=True)
class SchemeCfg:
    code: str                        # a1..c5
    tunables: dict                   # 本方案组的全部可调参数（§6 各组表）

@dataclass(frozen=True)
class RunCfg:                        # 顶层，1:1 映射 YAML
    tag: str; seed: int; warmup: int; repeat: int
    output_root: str
    shape: ShapeCfg
    routing: RoutingCfg
    schemes: list[SchemeCfg]         # 按 YAML enabled 顺序
    dist: DistCfg                    # nproc, master_port, cuda_visible_devices
    env: dict[str, str]              # 需要注入的进程级环境变量
    verify: VerifyCfg                # 开关、容差表（§5.3）
    profile: ProfileCfg              # torch_profiler / intra_kernel / nvtx 开关
    log: LogCfg                      # level, diagnostics
    sweep_axes: list[SweepAxis]      # [{path: "shape.M", values: [...]}, ...]
```

要求：
- `load_config(yaml_path, set_overrides: list[str]) -> RunCfg`；`--set a.b.c=value` 用点路径覆盖，value 按 YAML 解析（支持列表）。
- 构造时做全部合法性校验（整除、n_down==n_gateup//2、E%ws==0 等），报错信息必须写明"哪个字段、当前值、约束是什么"。
- `resolved_dict()` 输出完全展开的配置（含默认值），用于快照落盘。

### 4.2 权重 checkpoint（`data/checkpoint.py`）

**唯一事实源**：全局 FP8 权重 + 128×128 scale。BF16 视图 = dequant(checkpoint)，供 BF16 方案与 golden 使用——保证"除精度外权重一致"。

```python
FP8_MAX = 448.0   # float8_e4m3fn

@dataclass
class QuantizedTensor:
    fp8: torch.Tensor        # [E, N, K] float8_e4m3fn
    scale: torch.Tensor      # [E, ceil(N/128), ceil(K/128)] float32
    bf16: torch.Tensor       # dequant 视图 [E, N, K] bf16（懒生成，可释放）

@dataclass
class WeightCheckpoint:
    w1: QuantizedTensor          # routed gate_up  [E, n_gateup, K]
    w2: QuantizedTensor          # routed down     [E, K, n_down]
    shared_w1: QuantizedTensor | None   # [1, 2*shared_intermediate, K]
    shared_w2: QuantizedTensor | None   # [1, K, shared_intermediate]
    gate_weight: torch.Tensor    # [E, K] bf16（路由 gate，所有 rank 相同）

def build_checkpoint(shape: ShapeCfg, seed: int, device) -> WeightCheckpoint: ...
```

生成规则（保证 rank 间一致）：所有随机源用 `torch.Generator(device).manual_seed(固定偏移)`，生成后 rank0 `dist.broadcast` 一次校验哈希（debug 模式下），正常路径直接靠同种子。源分布沿用旧代码 `randn * 0.01`。

**向量化 block 量化**（替换旧三重循环，必须实现成这样）：

```python
def quantize_block128(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """w: [E, N, K] float32 → (fp8 [E,N,K], scale [E,nN,nK])，N/K 可非 128 倍数。"""
    E, N, K = w.shape
    Np, Kp = -(-N // 128) * 128, -(-K // 128) * 128
    wp = F.pad(w, (0, Kp - K, 0, Np - N))
    blocks = wp.view(E, Np // 128, 128, Kp // 128, 128)
    amax = blocks.abs().amax(dim=(2, 4))                       # [E, nN, nK]
    scale = (amax / FP8_MAX).clamp(min=1e-12)
    q = (blocks / scale[:, :, None, :, None]).clamp(-FP8_MAX, FP8_MAX)
    fp8 = q.view(E, Np, Kp)[:, :N, :K].to(torch.float8_e4m3fn).contiguous()
    return fp8, scale
```

单测要求：与旧 `prepare_moe_weights` 循环版在小形状上逐位一致（fp8 位模式和 scale 都一致）。

### 4.3 路由（`data/routing.py`）

```python
@dataclass
class RoutingBundle:
    topk_ids_local: torch.Tensor      # [M_local, top_k] int32（本 rank token）
    topk_weights_local: torch.Tensor  # [M_local, top_k] fp32，归一化
    topk_ids_full: torch.Tensor       # [M, top_k]（all-gather 结果，预计算）
    topk_weights_full: torch.Tensor
    stats: RoutingStats               # per-expert counts、imbalance 指标（§5.2）

def build_routing(hidden_local, gate_weight, cfg: RoutingCfg, shape, ctx) -> RoutingBundle: ...
```

约束：路由必须来自 gate 矩阵乘 + softmax + topk（不允许直接采样 ids），分散度通过 logits bias 调（§5.2）。`topk_ids/weights` 的 dtype 与旧代码一致（int32 / fp32，weights 除以 sum 归一化）。

### 4.4 数据束（`data/bundle.py`）

```python
@dataclass
class DataBundle:
    hidden_local: torch.Tensor        # [M_local, K] bf16
    ckpt: WeightCheckpoint
    routing: RoutingBundle
    golden: GoldenOutputs             # §5.3，构建期一次算好
```

每次 torchrun 进程构建一次 DataBundle，所有 scheme 共享；scheme 私有的 shard/量化副本在各自 `build()` 内 materialize、`close()` 时释放（沿用旧代码 per-case 隔离 + empty_cache 的正确做法）。

### 4.5 方案抽象（`schemes/base.py`）

```python
@dataclass(frozen=True)
class SchemeSpec:
    code: str                  # "a1"
    name: str                  # "vLLM-TP-Serial"
    family: str                # baseline | chunk_overlap | td_fused
    parallel: str              # TP | EP
    comm: str                  # 描述串（进报告表格用）
    weight_dtype: str          # fp8 | bf16
    act_quant: str             # none | group128 | rowwise   ← 决定 golden 档位（§5.3）
    requires_nvshmem: bool
    tunables_schema: dict      # 名称→(类型, 默认值, 说明)；build 时校验 YAML tunables

class SchemeInstance:
    run: Callable[[], torch.Tensor]        # 一次完整 forward → 本 rank 输出 [M_local, K] bf16
    run_staged: Callable[[], StageResult] | None   # 可选：带 stage 打点的一次 forward（§7.3）
    diagnostics: dict                      # 参与打印的 shape / 路由信息（计时区外用）
    close: Callable[[], None]              # 释放 stream/ctx/对称内存

BuilderFn = Callable[[RunCfg, DistContext, DataBundle, SchemeCfg], SchemeInstance]
REGISTRY: dict[str, tuple[SchemeSpec, BuilderFn]]   # schemes/__init__.py 中注册
```

**输出契约（公平性的关键）**：每个 scheme 的 `run()` 从 `hidden_local` 出发，返回本 rank 的 `[M_local, K]` bf16 输出分片（含 shared expert 与 topk 加权 combine）。EP 方案内部需要的路由 all-gather 等通信算在自己的计时内（这是方案本身的开销）；但**输入数据的构造、量化好的权重 shard 不算**（预先 materialize）。activation 的动态量化必须在 `run()` 内（真实推理每步都要做）。

新增一个方案 = 新建一个 `schemes/xxx.py` + 在 REGISTRY 注册 + YAML 里 enable，不改其他任何文件——这是"可复用/易扩展"的验收口径。

### 4.6 计时（`timing.py`）

沿用旧 `bench_cuda` 的正确骨架（barrier → warmup → sync+barrier → CUDA events 逐次计时 → sync），增强：
- 输出增加 `p95_ms`、`std_ms`；samples 全量进 JSONL。
- `StageTimer`：在 `run_staged()` 内用 `cuda.Event` 对（stage 名 → 事件对列表），一次 forward 产出各 stage 时长；只在明确请求 stage 分解时使用（事件本身有 ~µs 级扰动，端到端数与 stage 数分开跑）。
- 所有 scheme 的 `run()` 内保留 nvtx range（外层一个 + 内部关键段），`profile.nvtx=false` 时跳过（封装 `maybe_nvtx()`）。

### 4.7 分布式上下文（`context.py`）

```python
@dataclass(frozen=True)
class DistContext:
    rank: int; world_size: int; local_rank: int
    device: torch.device; group: object
    nvshmem_initialized: bool
```

初始化逻辑与旧 `init_distributed` 相同：任一 enabled scheme `requires_nvshmem` → 走 `triton_dist.utils.initialize_distributed`（此前先 `tdx.compat.apply()`，见 §5.4.4），否则纯 NCCL。NCCL warmup 保留。

---

## 5. 关键设计决策详解

### 5.1 权重统一切分与 TP scale 对齐问题（S4 核心）

**问题**：`und`/`gen` 形状 intermediate=768，TP=4 → per-rank intermediate=192，192 % 128 ≠ 0，全局 128×128 scale 无法沿 TP 切分边界干净切开；旧代码在这两个形状直接跑不起来。

**三级策略**（`data/shard.py` 实现，按优先级自动选择并在日志/结果里记录用了哪级）：

- **L0 — 对齐直切（零误差）**：切分边界是 128 的倍数时（EP 切 expert 维恒满足；`v1` 形状 TP 也满足：3072/4=768=6×128），fp8 与 scale 直接切片，与 checkpoint 逐位一致。
- **L1 — BF16 重量化（微小二次误差）**：边界不对齐时，从 BF16 视图切 shard，再对 shard 做 `quantize_block128`（shard 内从 0 列重新对齐，末块允许 <128 的 ragged 块）。误差性质：shard 内 block 是全局 block 的子集，amax 只会变小或不变，重量化格点误差极小且有界；S4 需实测 `max|dequant(L1)-bf16视图|` 并记录在 EXP 文档。
- **L2 — pad 到 128（保底）**：若某内核不支持 ragged 尾块（S4 用最小复现确认，见下），把 per-rank intermediate pad 到 128 倍数（192→256），pad 列权重置 0，silu-and-mul 与 down GEMM 按 pad 后维度算，结果数学不变；**计算量增幅（256/192=+33%）必须写进结果记录和报告**，这是该方案在此形状下的真实代价（或者说明为什么该实现需要这个约束）。

S4 第一步不是改 bench，而是先跑 A2 写好的**最小复现**：`tests/test_ragged_block_quant_gemm.py` —— 单卡、单个 blockwise FP8 GEMM，K 维=192（1 个整块 + 1 个 64 ragged 块），分别过 vLLM `fused_moe` 内核路径和 tdx FP8 GEMM 路径，确定各内核到底支不支持 ragged、坏在哪一步（权重量化 / activation 分组量化 / kernel 索引）。用结论决定每条路径落在 L1 还是 L2，并记录 EXP。

activation 侧同理：K（2048/3200/4096）都被 128 整除，AG 方向 group128 量化无问题；down GEMM 的输入是 per-rank intermediate（192），group128 分组同样 ragged——与权重同级处理。

### 5.2 统一 gate 路由 + 分散度控制

实现（`data/routing.py`）：

```python
logits = hidden_local.float() @ gate_weight.float().T        # [M_local, E]
if cfg.imbalance_kind != "none":
    p = target_dist(cfg, E)          # zipf: p_e ∝ (e+1)^-s（按固定置换打乱专家序）
                                     # hotspot: hot_experts 个专家均分 hot_share，其余均分剩余
    logits += cfg.strength * torch.log(E * p).to(logits)     # bias 注入
topk_w, topk_ids = torch.topk(softmax(logits), top_k)
```

要点：
- gate_weight 全 rank 同种子生成（checkpoint 的一部分），路由天然全局一致；`topk_*_full` 用 all-gather 预生成（在计时区外，作为"输入"）。
- bias 法保持"路由来自 gate 计算"的语义，实际分布**近似**目标分布。`RoutingStats` 必须记录实测值：per-expert counts、`imbalance_factor = max/mean`、变异系数 CV、per-rank(EP owner) token 数；写入 JSONL 与诊断日志。
- 校准（S4 验收）：`none` 时 CV < 0.15；`hotspot(hot_share=0.6, strength=1)` 时热点专家实收份额 ≥ 0.45（达不到就加大 strength，把校准曲线记进 EXP 文档）。
- 三档标准 sweep 值（进 configs）：`none` / `zipf_s=1.2` / `hotspot(4, 0.6)`。

### 5.3 正确性验证（纯 PyTorch golden，双档）

`verify.py` 提供两档参考，全部 FP32 累加、按 expert 分组循环实现（E≤128、M≤7k，构建期一次算完，秒级）：

- **`golden_bf16`**：BF16 视图权重（= checkpoint dequant）、activation 不量化。度量"该方案离理想 BF16 计算差多远"。
- **`golden_fp8sim`(granularity)**：权重同上（本来就是 fp8 格点值）；activation 在两处 GEMM 前做**模拟量化**（quantize→dequant 回 fp32 再乘）：`granularity ∈ {group128, rowwise}`，与各 scheme 的真实 act 量化粒度一致（SchemeSpec.act_quant 声明，执行者迁移每个 scheme 时从代码确认它的量化调用点并填对）。度量"扣掉量化损失后，实现本身是否正确"。

```python
def golden_moe(hidden_full, routing, w1_bf16, w2_bf16, shared_w…,
               act_quant: str) -> torch.Tensor:   # [M, K] fp32
    out = zeros(M, K, fp32)
    for e in range(E):
        rows, slot = (topk_ids_full == e).nonzero(...)
        x = maybe_quant_dequant(hidden_full[rows].float(), act_quant)
        g, u = split(x @ w1_bf16[e].float().T)
        h = maybe_quant_dequant(silu(g) * u, act_quant)
        out.index_add_(0, rows, (h @ w2_bf16[e].float().T) * topk_w[rows, slot, None])
    out += shared 路径（同样规则）
    return out
```

**验证矩阵**：每个 scheme 输出（本 rank 分片）与 `golden[rank*M_local:(rank+1)*M_local]` 比：

| scheme 类型 | 主判据（必须过） | 副判据（记录用） |
|---|---|---|
| BF16 权重方案（c1, c2）| vs golden_bf16，紧容差 | — |
| FP8 方案（其余）| vs golden_fp8sim(对应粒度)，紧容差 | vs golden_bf16，松容差 + cos_sim |

**容差不许拍脑袋**：校准程序（A2 编写、S1 运行） `tests/calibrate_tolerance.py` 对每个形状实测 `|golden_fp8sim − golden_bf16|` 的 max/p99，紧容差 = 对应档 golden 之间残差包络 × 1.5，写进 configs 的 verify 段并在 EXP-001 记录推导。所有比较输出 `max_abs / max_rel / cos_sim / PASS`（沿用旧 verify 输出格式）。**任何"放宽容差让测试通过"的动作必须在 EXP 文档留痕并说明原因。**

### 5.4 triton_dist fork 改动搬入 `moe_bench/tdx/`

原则：搬完之后，`PYTHONPATH` 指向干净的 `{{TD_UPSTREAM}}`（1b9dc71a）即可跑全部 c 组；fork 不再需要安装。

#### 5.4.1 权威清单的获取

执行者在服务器上跑：
```bash
cd {{TD_FORK}} && git diff --stat 1b9dc71a..HEAD && git status --short
```
以输出为准（下表基于当前快照，若有未提交改动一并处理）。

#### 5.4.2 文件处置表

| fork 文件（python/triton_dist/…） | 改动量 | 性质 | 处置 |
|---|---|---|---|
| `kernels/nvidia/fp8_allgather_group_gemm.py` | 新 +551 | c4/c5 内核 | 整文件搬 `tdx/kernels/` |
| `kernels/nvidia/fp8_moe_reduce_rs.py` | 新 +949 | c5/c3-RS 内核 | 整文件搬 |
| `kernels/nvidia/swiglu_quantize_fp8.py` | 新 +179 | 融合 SwiGLU+量化 | 整文件搬 |
| `kernels/nvidia/ep_all2all_fused.py` | 改 +569 | c1/c3 a2a mega kernel | **取 fork HEAD 整文件**搬 `tdx/kernels/`（shadow 上游同名模块） |
| `kernels/nvidia/memory_ops.py` | 改 +72 | 辅助 op | 先 diff：若纯新增函数→提取到 `tdx/kernels/memory_ops_ext.py`；若改了已有函数→整文件搬 |
| `kernels/nvidia/moe_utils.py` | 改 2 行 | num_warps 32→16（调优） | 不整搬；tdx 调用处改为可传 `num_warps` 的本地包装（进 tunables） |
| `function/nvidia/ep_moe_fused.py` | 改 | c1/c3 autograd Function + GEMM 配置 | 整文件搬 `tdx/function/`，写死的 tile/warp/stage 外提为参数（§5.4.3） |
| `function/nvidia/common.py` | 改 +82 | intra-kernel profiler 开关等 | 整文件搬 `tdx/function/common.py` |
| `layers/nvidia/fp8_ep_moe.py` | 改 +62 | c3 层 | 整文件搬 `tdx/layers/`；`TRITON_DIST_FP8_RS` env 改为构造参数 |
| `layers/nvidia/fp8_tp_moe.py` | 新 +568 | c4/c5 层 | 整文件搬；写死的 GEMM config（128/128/128, GROUP 8, warps 8）外提 |
| `layers/nvidia/ep_a2a_fused_layer.py` | 改 +190 | a2a ctx | 整文件搬 |
| `layers/nvidia/ep_moe.py` | 改 2 行 | num_sm 64→110（调优） | 整文件搬（c2 依赖的 `tp_moe.py` 上游未改、不搬），`num_sm` 外提 |
| `jit.py` / `nv_utils.py` / `kernels/common_ops.py` | 小 | **环境兼容修复**（get_ptxas 查找、`__fence`→`fence` 改名） | **不搬文件**；做成 `tdx/compat.py` 运行时 monkeypatch（§5.4.4） |
| `kernels/nvidia/benchmark_dot_scaled.py`、`benchmark_fp8_vs_fp16_gemm.py`、`tune_gemm_config.sh` | 新 | 独立微基准/调优脚本 | 择要迁到 `microbench/`（改造成统一输出格式），不进 tdx |

#### 5.4.3 import 改写与调优参数外提规则

- 搬入文件里 `from triton_dist.X import Y`：若 X 也被搬入 tdx → 改为 `from moe_bench.tdx....X import Y`；否则保留上游 import。搬完跑 `grep -rn "from triton_dist\|import triton_dist" moe_bench/tdx/` 逐条 review，在 `docs/exp/EXP-00x-tdx-migration.md` 里贴上"保留的上游 import 清单"。
- 所有写死的调优常量外提为函数/构造参数，默认值 = fork 当前值（保证行为不变），并接到各 scheme 的 `tunables_schema`。**外提清单**（迁移时逐一核对源码补全）：
  - c1/c3：`FWD_GEMM_BLOCK_SIZE_N/K`、`GEMM_NUM_STAGES`、`GROUP_SIZE_M`、`num_dispatch_warps`、`num_combine_warps`、a2a `num_sm`、`capacity`、gather/scatter index kernel 的 `num_warps`；
  - c3 FP8-RS：`fp8_rs_enabled`（原 env `TRITON_DIST_FP8_RS`）、`num_combine_sms`（原 `TRITON_DIST_FP8RS_NUM_COMBINE_SMS`=8）、`num_reduce_sms_in_combine`（原 `..._REDUCE_SMS...`=100）；
  - c4/c5：GEMM `BLOCK_SIZE_M/N/K`、`GROUP_SIZE_M`、`num_warps`、`num_stages`、`n_chunks_rs`、`block_k_quant/block_n_quant`。
- `TRITON_DIST_*` 环境变量在新代码中**全部消失**（grep 验证），一律走 YAML tunables。

#### 5.4.4 `tdx/compat.py`（不改安装库的兼容层）

```python
_applied = False
def apply():
    """必须在 import 任何 triton_dist.kernels/layers 之前调用（worker 启动最早处）。
    每个 patch 先探测上游是否已经是新行为，是则跳过 —— 同一份代码兼容新旧上游。"""
    # 1) nv_utils.get_ptxas 缺失 → 注入（shutil.which + CUDA_PATH 回退）
    # 2) jit.nvidia_stages_inspection_hook 中 get_ptxas 的取法差异 → wrap 替换
    # 3) common_ops 的 fence 导入名（__fence vs fence）→ try/except 双路径,
    #    若上游模块尚未 import, 用 sys.modules 预注入不可行时改为 import 后 setattr(fence_nv)
```
实现时对照 fork 中这三个文件的 diff（见 §5.4.1 命令），每个 patch 附单测：`tests/test_tdx_compat.py` 在干净上游环境 import c1/c3 的内核模块不报错。

#### 5.4.5 `moe_overlap/` → `moe_bench/overlap/`

b 组内核平移（它本来就在本仓库），趁迁移做三件事：包名规范化（`from moe_bench.overlap import ...`）、去掉 `graph_forward` 之类死代码、`SPLIT_K` 等 config 处理保持现状。行为不许变，用 S2 的 perf parity 验收。

#### 5.4.6 搬运保真验收（S3 硬性 DoD）

同一形状（三个 preset 各一）、同一配置下：
1. 旧 bench（fork 安装）与新 bench（`{{TD_UPSTREAM}}` + tdx）c1/c3/c4/c5 全部 verify PASS；
2. 每组 avg_ms 差异 < 3%（噪声内）；对照表写进 `EXP-00x-tdx-migration.md`。

### 5.5 配置系统与运行入口

**唯一入口**：

```bash
{{PYTHON}} -m moe_bench.cli configs/sweep_und.yaml --set run.tag=my_test --set "schemes.enabled=[a1,b3,c3]"
```

`cli.py` 职责：加载配置 → 展开 `sweep_axes` 笛卡尔积 → 创建 run 目录、写 manifest 与 resolved config → 对每个 sweep 点：注入 `env:` 段的环境变量（NVSHMEM_*、CUDA_VISIBLE_DEVICES、C_INCLUDE_PATH、TRITON_PTXAS_PATH、CUDA_DEVICE_MAX_CONNECTIONS 等，全部有注释的默认值在 default.yaml 里）→ spawn 一次 `torchrun --nproc_per_node=... moe_bench/worker.py --run-dir ... --point-json ...` → 收集退出码。每点独立进程 = 天然隔离 NVSHMEM 堆与显存泄漏（沿用现有 sweep 的正确做法）。最后聚合生成 `summary.md`。

YAML 模板（`configs/default.yaml`，全字段带注释；此处示意）：

```yaml
run: {tag: default, seed: 42, warmup: 20, repeat: 50, output_root: results}
shape: {preset: und}            # 或展开写 M/K/E/top_k/n_gateup/n_down/shared_experts
routing: {imbalance: {kind: none}}
dist: {nproc: 4, master_port: 29500, cuda_visible_devices: "0,1,2,3"}
env:
  NVSHMEM_SYMMETRIC_SIZE: "4294967296"
  NVSHMEM_REMOTE_TRANSPORT: none
  NVSHMEM_DISABLE_CUDA_VMM: "1"
  CUDA_DEVICE_MAX_CONNECTIONS: "1"
schemes:
  enabled: [a1, a2, b1, b2, b3, c1, c2, c3, c4, c5]
  b3: {tunables: {n_chunks_gateup: 2, n_chunks_down: 4}}
  c3: {tunables: {fp8_rs_enabled: true, num_combine_sms: 8, num_reduce_sms_in_combine: 100}}
  c4: {tunables: {n_chunks_rs: 32}}
verify: {enabled: true}         # 容差表由校准程序生成后填入
profile: {torch_profiler: false, intra_kernel: false, nvtx: true}
log: {level: info, diagnostics: false}
sweep_axes: []                  # 例: [{path: shape.M, values: [128,512,1024,2048,3072,4096,5120]}]
```

**/tmp hack 的正式解法**：新包用绝对 import + `python -m` 运行即可。若服务器仓库根存在 `vllm/` 源码目录导致 `import vllm` 冲突，把它移出仓库（或改名 `vllm_src_ref/`）；worker 顶部那段清 sys.path 的补丁不再保留。S1 验收含"不复制任何文件到 /tmp 能直接跑"。

### 5.6 日志

- `logging_util.get_logger()`：Python logging，格式 `[HH:MM:SS.mmm][rank0][INFO] msg`；rank0 → 控制台 + `logs/rank0.log`，其他 rank 只写各自文件（崩溃排查用）。
- 级别：`error/warn/info/debug`；`diagnostics`（per-rank shape/路由表，即旧 `print_rank_debug`）是独立开关，实现为函数，**只在计时循环之外调用**；诊断里的 GPU→CPU 拷贝（counts.tolist 等）在关掉时一行都不许执行（先判断开关再构造字符串/拷贝）。
- 计时循环内部零日志、零字符串格式化。
- 每个 run 目录固化一份完整日志，EXP 文档引用路径而非粘贴长 log。

### 5.7 结果与 profile 产物规范

```
results/<YYYYMMDD_HHMMSS>_<tag>/
├── manifest.json          # git sha(本仓库/TD_UPSTREAM/vllm 版本)、GPU 名与数量、driver/cuda、
│                          #   完整 env 快照、命令行、sweep 点列表
├── config.resolved.yaml
├── results.jsonl          # 窄表：一行 = 一个 (sweep点, scheme)
├── summary.md             # 自动生成：M×scheme 时延表 + vs a1 加速比 + verify 汇总
├── logs/rank*.log
└── profiles/
    ├── torch/<point>_<scheme>_rank<r>.json        # chrome trace
    ├── intra/<point>_<scheme>_*.perfetto-trace    # TD intra-kernel（tdx 的输出目录改为可传参，
    │                                              #   不再写死 prof/mega/）
    └── ncu/<point>_<scheme>_*.ncu-rep
```

`results.jsonl` 单行 schema：

```json
{"run_id": "...", "ts": "...", "scheme": "c3", "shape": {"M":6648,"K":2048,"E":128,"top_k":8,"n_gateup":1536,"n_down":768,"shared":1},
 "routing": {"kind":"zipf","zipf_s":1.2,"imbalance_factor":3.4,"cv":0.9},
 "tunables": {...}, "shard_level": "L1",
 "lat_ms": {"avg":1.23,"min":...,"med":...,"p95":...,"std":...}, "samples_ms": [...],
 "stage_ms": {"quant":..., "ag":..., "gemm1":...},
 "verify": {"vs":"fp8sim_rowwise","max_abs":...,"cos_sim":...,"pass":true},
 "speedup_vs_a1": 1.31}
```

分析脚本 `moe_bench/results.py` 提供 `load_runs(glob) -> pandas.DataFrame`，summary/报告图表都从 JSONL 生成，禁止手抄数字。

---

## 6. 方案组定义（10 组）

统一编码保留现状。每组迁移时在文件头 docstring 写清数据流（下表"流水"列即模板）。

| code | 名称 | family | 并行 | 权重 | act 量化 | 流水（计时区内） | 主要 tunables |
|---|---|---|---|---|---|---|---|
| a1 | vLLM-TP-Serial | baseline | TP | FP8 | group128 | AG(bf16) → vLLM fused_experts → [+shared] → RS(bf16) | — |
| a2 | vLLM-EP-Naive | baseline | EP | FP8 | group128 | AG(hidden+routing) → fused_experts(expert_map) → RS → +shared | — |
| b1 | Overlap-BF16 | chunk_overlap | TP | FP8 | group128 | chunk AG(bf16) ∥ GEMM → chunk RS(bf16) | n_chunks_gateup/down |
| b2 | Overlap-FP8AG | chunk_overlap | TP | FP8 | group128 | 量化→chunk AG(fp8+scale) ∥ GEMM → RS(bf16) | 同上 |
| b3 | Overlap-FP8AG-FP8RS | chunk_overlap | TP | FP8 | group128 | 同 b2 + RS 改 fp8+scale | 同上 |
| c1 | TD-EP-BF16 | td_fused | EP | BF16 | none | NVSHMEM a2a dispatch+GroupGEMM+combine mega kernel | a2a num_sm、GEMM tile/stages、dispatch/combine warps |
| c2 | TD-TP-BF16 | td_fused | TP | BF16 | none | NVSHMEM tile 级 AG-GEMM + RS | n_chunks_rs、GEMM tile |
| c3 | TD-EP-FP8 | td_fused | EP | FP8 | rowwise | rowwise 量化→ll-a2a fp8 dispatch+FP8 GroupGEMM→SwiGLU+量化→GroupGEMM+combine（可选 FP8 RS） | FP8 GEMM tile、fp8_rs_enabled、combine/reduce SMs |
| c4 | TD-TP-FP8 | td_fused | TP | FP8 | group128 | FP8 AG + FP8 GroupGEMM + RS(bf16) | GEMM tile、n_chunks_rs、block_*_quant |
| c5 | TD-TP-FP8-RS | td_fused | TP | FP8 | group128 | 同 c4 + FP8 RS | 同 c4 |

共性约定：
- shared expert 一律走独立 stream 并发（沿用现有 event 同步模式，抽到 `schemes/base.py` 复用）；`shared_experts: 0` 时整段跳过。
- a1 是**性能对照锚点**（报告里的 vs a1 列），不再兼任正确性基准（golden 已接管）。
- c2 的 sm120 bug 已修复：迁移后在三形状全跑通并 verify，顺手把 CLAUDE.md/文档里的过时标注清掉。

---

## 7. 单测与微基准

### 7.1 正确性单测（`tests/`，pytest；分布式 case 用 `torchrun -m pytest` 或脚本包装）

| 测试 | 卡数 | 验证内容 |
|---|---|---|
| `test_quantize.py` | 1 | 向量化量化 vs 旧循环版逐位一致；ragged 维度 pad 正确 |
| `test_checkpoint.py` | 4 | 各 rank checkpoint 哈希一致；BF16 视图=dequant |
| `test_shard.py` | 1 | L0 切分逐位=checkpoint 切片；L1 重量化误差 < 校准界；L2 pad 后 dequant 数学等价 |
| `test_routing.py` | 4 | 路由确定性（两次构建一致）、rank 间 full 视图一致、imbalance 命中校准目标 |
| `test_golden.py` | 1 | golden 自洽：E=1/topk=1 退化成单 FFN 与直接矩阵乘一致；fp8sim 与 bf16 档残差量级合理 |
| `test_ragged_block_quant_gemm.py` | 1 | §5.1 的最小复现（vLLM 路径 + tdx FP8 GEMM 路径） |
| `test_tdx_compat.py` | 1 | 干净上游 + compat.apply() 后 tdx 模块可 import |
| `test_schemes_smoke.py` | 4 | 每个 scheme 在 M=256/`v1` 缩小形状上 run 一次 + verify PASS（CI 级快速回归） |

### 7.2 微基准（`microbench/`，每个独立 torchrun 脚本，输出统一走 results.jsonl 格式，`kind: microbench`）

这些是瓶颈分析的弹药，脚本在 A5 写好、S5 逐个实施，每个至少产出一篇 EXP 文档：

| 脚本 | 目的 | 关键设计 |
|---|---|---|
| `mb_moe_align.py` | dispatch 前重排（sort/align/pad）代价定位 | 复刻《Tile 排序对 fused_moe GEMM 性能影响》方法：同一 GEMM kernel、只换 (sorted_ids, expert_ids) 元数据、GEMM 独占整卡；对比原版 align / TD 的 gather-scatter index 排序；扫 M/E/topk |
| `mb_dispatch_prep.py` | c1/c3 的 `calc_gather_scatter_index` + token permute 的绝对代价与 scaling | 单独调 kernel 计时，M∈{1k..7k}×E∈{64,128}×topk∈{8,16}，画代价占端到端百分比 |
| `mb_tile_order_groupgemm.py` | tile 排序/expert 局部性对 FP8 GroupGEMM 的影响（L2 工作集分析） | 同上方法论移植到 tdx FP8 GroupGEMM；对照 expert 主序 vs phase 主序 vs 半 phase 分组；配合 NCU L2 hit rate |
| `mb_comm_prims.py` | NCCL AG/RS vs NVSHMEM a2a/ll 的消息大小-带宽曲线；FP8(payload+scale) 的**有效**带宽 | 消息 1KB→256MB sweep；FP8 曲线要算上 scale 传输与量化/反量化 kernel 时间，找 BF16→FP8 的盈亏平衡点 |
| `mb_group_gemm.py` | FP8 vs BF16 GroupGEMM 效率随 per-expert token 分布的变化 | 均匀/zipf/hotspot 三种 counts 下同 FLOPs 对比；tile 配置 sweep（接调优） |
| `mb_quant_kernels.py` | rowwise/group128 量化、SwiGLU+量化融合核的带宽利用率 | 单核计时 + 理论带宽对比 |
| `mb_chunk_overlap.py` | b 组 chunk 数 → overlap 效率曲线 | n_chunks ∈ {1,2,4,8,16}；报告 overlap 效率 η（§9.4 定义） |

### 7.3 stage 分解（内建于 schemes）

非融合方案（a/b 组、c4/c5 的 RS 段）实现 `run_staged()`：quant / AG / gemm1 / silu / gemm2 / RS / shared 各 stage 用事件对计时。融合 mega kernel（c1/c3）无法用事件切，stage 数据来自 TD intra-kernel profiler（perfetto trace 里 dispatch/GEMM/combine 的 SM 时间线），tdx 迁移时把 profiler 输出目录参数化接到 `profiles/intra/`。

---

## 8. 执行计划（阶段一：本地 A1–A5；阶段二：服务器 S0–S7）

每个 Phase 的产出包括代码 + 对应文档 + git commit（规范：`A<N>/S<N> <模块>: <做了什么>`；一个逻辑改动一个 commit）。**Phase 未过 DoD 不得进入下一个**。

**阶段一在本地（无 GPU）完成**：产出全部代码 + 交接材料。本地做不了的验证（CUDA 单测、parity、性能）一律不做假设——凡是"写了但没在 GPU 上验证过"的行为差异点，在代码处标注 `# SERVER-VERIFY: <要验证什么、怎么验证>`，并由 A5 汇总进 HANDOFF.md。
**阶段二在服务器完成**：按 HANDOFF.md 的任务清单执行测试、验证、分析、调优与报告。

---

### 阶段一（本地重构，产出 = `{{NEW_ROOT}}` 全部内容）

#### A1 — 框架骨架与公共设施
1. `config.py / cli.py / worker.py / context.py / logging_util.py / timing.py / results.py` 全部实现。
2. 本地可验证部分：config 加载/合并/`--set` 覆盖/校验/快照、sweep 展开、results.jsonl 读写与 summary 生成（喂假数据测）、日志分级——全部配 CPU 单测。
3. cli 的 torchrun spawn 逻辑加 `--dry-run`：只打印将要执行的命令与注入的 env，不真正启动（本地即可验证命令构造正确性）。
- **DoD**：CPU 单测绿；`python -m moe_bench.cli configs/smoke.yaml --dry-run` 输出正确的 torchrun 命令与 env。

#### A2 — 数据层 + golden + 校准程序
1. `data/`：checkpoint（向量化量化）、routing（含 imbalance）、shard（L0/L1/L2 三级策略）、golden 双档、`calibrate_tolerance.py`——全部 device 无关（§4 规则 1）。
2. CPU 单测：`test_quantize.py`（vs 旧循环版逐位一致，旧实现拷贝进 tests 作 oracle）、`test_shard.py`、`test_routing.py`（单进程模拟多 rank：world_size 作参数、all-gather 用本地拼接 stub）、`test_golden.py`。
3. `test_ragged_block_quant_gemm.py` 与容差校准程序写好（依赖 GPU 的部分标记 skip-if-no-cuda）。
- **DoD**：CPU 单测绿；imbalance 目标分布的数学行为（bias→分布偏移方向与单调性）在 CPU 上验证。

#### A3 — a/b 组 schemes + overlap 迁移
1. `overlap/` 平移清理（§5.4.5）；a1、a2、b1、b2、b3 按新抽象重写（行为参考旧 case builder）；`run_staged()` 实现。
2. 本地验证限于：模块可导入（vllm 缺席时延迟导入不炸）、builder 的参数拼装逻辑用 mock 测（tunables 校验、shard 选级路径）。
- **DoD**：`python -m compileall` / 静态检查干净；scheme 注册表完整；SERVER-VERIFY 标记就位。

#### A4 — tdx 搬运 + c 组 schemes（风险最高）
1. 按 §5.4 处置表从 `{{TD_FORK}}` 搬文件、改 import、外提 tunables、写 `compat.py`；本地 clone `{{TD_UPSTREAM}}` 用于逐文件 diff 比对搬运保真（文本级：搬入文件 vs fork HEAD 一致、import 改写清单可审计）。
2. c1、c2、c3、c4、c5 按新抽象重写。
3. 本地验证：`grep -rn "from triton_dist\|import triton_dist" moe_bench/tdx/` 逐条 review 并记录；`grep -rn TRITON_DIST_ moe_bench/` 无 env 依赖残留；tunables 默认值与 fork 当前值逐项核对表。
- **DoD**：搬运审计表成文（进 `docs/exp/EXP-A4-tdx-migration.md` 草稿，服务器验证后补 parity 数据）；compat.py 的每个 patch 附"探测-跳过"逻辑。

#### A5 — 配置、微基准脚本、文档骨架、交接包
1. `configs/` 全套（default/smoke/三形状 sweep/调优模板）；`microbench/` 七个脚本（§7.2）写完，标注 SERVER-VERIFY。
2. `docs/` 骨架：`methodology.md`（§9 落地）、`exp/INDEX.md`、EXP 模板、`report/` 大纲占位。
3. **`HANDOFF.md`（交接主文档，阶段二执行者的入口）**，必须包含：
   - 服务器路径替换表（占位符 → 待填真实路径）；
   - 环境准备步骤（clone `{{TD_UPSTREAM}}`@1b9dc71a、PYTHONPATH 设置、依赖版本要求、验证环境就绪的命令）；
   - S0–S7 任务清单（每项 = 目的 + 具体命令 + 预期输出 + 失败时的排查入口）；
   - 全部 `SERVER-VERIFY` 标记汇总表（文件:行号、要验证什么、验证方法）；
   - 已知风险与坑（§13 已知坑 + 搬运中新发现的）。
- **DoD**：`{{NEW_ROOT}}` 自包含可整体同步；无 GPU 环境下 `import moe_bench` + CPU 单测全绿 + `--dry-run` 可用；HANDOFF.md 能让不了解上下文的执行者直接开工。

---

### 阶段二（GPU 服务器，入口 = HANDOFF.md）

#### S0 — 环境准备 + 旧 bench 基线快照
1. 同步 `{{NEW_ROOT}}`、填好路径表；clone `{{TD_UPSTREAM}}`@1b9dc71a；确认 vllm / fork 版本与 HANDOFF 记录一致。
2. 用**旧 bench**（`{{BENCH_ROOT}}` 原样代码 + fork）在三形状跑 `GROUP_MODE=all`（M 取 preset 值，warmup=20 repeat=50），保存完整 results。
3. `EXP-000-baseline-snapshot.md`：环境信息 + 三形状 × 10 组时延表（und/gen 跑不起来的组如实记 FAIL+报错——这正是重构要修的）。
- **DoD**：EXP-000 成文；后续所有 parity 对照以它为准。

#### S1 — 单测全绿 + 容差校准 + smoke
1. 全部单测以 CUDA 跑（含 CPU 已绿的那些）；`calibrate_tolerance.py` 产出三形状容差表 → 写入 configs。
2. `configs/smoke.yaml`（缩小形状）全 10 组跑通 + verify PASS。
3. 清掉 A1/A2 相关的 SERVER-VERIFY 标记（逐条验证后删除标记并在 HANDOFF 勾选）。
- **DoD**：单测绿；`EXP-001-golden-and-tolerance.md`（容差推导 + L1 二次量化误差实测）。

#### S2 — a/b 组 parity
1. `v1` 形状：a1/a2/b1/b2/b3 verify PASS；与 EXP-000 对比 avg_ms 差 < 3%。
- **DoD**：`EXP-002-ab-migration-parity.md`（新旧对照表）；对应 SERVER-VERIFY 清零。

#### S3 — tdx 保真 + c 组 parity
1. PYTHONPATH 指向 `{{TD_UPSTREAM}}`（干净上游），c1–c5 跑通；§5.4.6 保真验收（verify PASS + avg_ms 差 < 3%）。
- **DoD**：`EXP-003-tdx-migration.md` 定稿（A4 草稿 + parity 数据）；对应 SERVER-VERIFY 清零。

#### S4 — 三形状全通 + 分散度校准
1. 跑 `test_ragged_block_quant_gemm.py` 定位 und/gen 断点 → 按 §5.1 敲定各路径 L1/L2 → 三形状 × 10 组全部 verify PASS。
2. imbalance 三档实测校准（§5.2），必要时调 strength 并回写 configs。
3. sweep 配置（三形状 × M 列表 × 分布三档）跑通，summary.md 自动生成。
- **DoD**：`EXP-004-scale-shard-fix.md`（断点分析 + 选型理由 + L2 pad 代价定量）；`EXP-005-imbalance-calibration.md`。

#### S5 — 微基准与瓶颈分析
1. §7.2 七个微基准逐个实施，每个一篇 EXP（编号 010+）。
2. stage 分解 + intra-kernel profile 打通，产出各方案在三形状的 stage 占比图数据。
3. 核心问题的第一轮答案：**tile 级融合 vs chunk 级 overlap vs 串行**的差距来自哪里（exposed comm？重排？GEMM 效率？），每个结论都要有"对照实验 + profile 证据"双支撑。
- **DoD**：EXP-010~016；`docs/methodology.md` 随做随补案例。

#### S6 — 调优
1. 对 c3/c4/c5 的 tunables 做网格/坐标下降 sweep（每形状），b 组扫 chunk 数；调优脚本进 `microbench/tune_*.py`，结果进 JSONL。
2. 每形状每方案落一份 `configs/tuned/<shape>_<scheme>.yaml`。
- **DoD**：`EXP-020-tuning-<scheme>.md` 系列（调优前后对照 + 最优配置表 + 为什么这个配置好——结合 §9 方法论解释，不许只贴数字）。

#### S7 — 总报告 + 清理
1. 按 §11 大纲产出 `docs/report/final_report.md`（图表数据全部来自 results.jsonl，附生成脚本）。
2. 确认 SERVER-VERIFY 全部清零；删除旧代码（benchmark.py、benchmarks/、moe_overlap/、run.sh、bench_config.yaml），更新 CLAUDE.md/README 为新用法。
- **DoD**：新人只看 README 能复现 sweep；报告可直接转 PPT。

---

## 9. 性能分析方法论（执行者必读，落地为 docs/methodology.md）

### 9.1 计时纪律
- warmup ≥ 20（首轮含 triton 编译/autotune/L2 预热，绝不能计入）；repeat ≥ 50；**报 median**，avg 仅参考；同配置连跑 3 次看抖动，抖动 > 效应量的实验无结论，必须加大 repeat 或找抖动源（时钟、邻居进程、温度）。
- 多卡计时：各 rank 各自计时，报告取**最慢 rank**（木桶效应才是真实时延）；barrier 只放在计时区外。
- 改动 A/B 对照必须同进程或背靠背跑，隔天的数不可比。

### 9.2 拆解顺序（漏斗）
端到端 → stage（事件/intra-kernel）→ kernel 排行（torch profiler top_ops）→ 单 kernel 深挖（NCU）。每层只带着上一层的"最大头"下钻，不要平铺全量。NCU 重点章节：Memory Workload（L2 hit rate、DRAM 吞吐）、Scheduler（occupancy、eligible warps）、指令 mix。

### 9.3 先算上限，再看实测
- 通信下限：`t ≥ bytes / BW`。PCIe 5.0 x16 单向 ~63 GB/s（实测以 `mb_comm_prims` 为准，方法论以实测带宽为分母）。AG 字节数 = `(ws-1)/ws × M × K × dtype_bytes`；FP8 减半但要加 scale 与量化 kernel。
- 计算下限：GEMM FLOPs / 峰值；MoE GroupGEMM 的有效峰值要按 per-expert m_e 的 tile 利用率折减（pad 浪费 = padded_tiles/需求 tiles）。
- 实测 / 上限 = 效率。效率 > 80% 的段不值得继续优化，去看别的段——报告里每个"瓶颈"结论都要有这个比值。

### 9.4 overlap 专用指标
- **overlap 效率** `η = (T_serial − T_overlap) / min(T_comm, T_compute)`（T_serial 用同方案强制串行或 a1 分段和），η∈[0,1]；η 低说明重叠没做出来（调度/资源抢占），η 高但总时延仍差说明代价在别处（重排、量化、kernel 变慢）。
- **exposed comm** = 端到端 − 纯计算时间（同数据、通信换成预填充跑一次即得），这是 overlap 方案间对比的核心量。
- chunk 数权衡：chunk 多 → 隐藏窗口多但每 chunk 启动/同步开销 × n；扫出 U 型曲线并解释两端。
- 融合 kernel 的"重叠"看 intra-kernel timeline：通信 SM 与计算 SM 的时间线是否真的并行、比例是否合理（c3 的 combine/reduce SM 划分实验就是这个方法）。

### 9.5 对照实验设计（参照《Tile 排序对 fused_moe GEMM 性能影响》一文，作为范文精读）
- **单一变量**：如该文只换 (sti, eid, npp) 元数据、kernel 完全相同；本 bench 里对应"只换 tunable / 只换分布 / 只换精度"。
- **公平对齐**：pad 数、FLOPs、输入数据必须对齐（该文方案 3 严格保持 pad 与方案 2 一致）；不能对齐的差异要定量声明（如 L2 pad 的 +33%）。
- **隔离环境**：测 kernel 自身影响时独占整卡（无并发通信流）；测端到端再放回真实并发。
- **提出机制假设 → 设计实验证伪**：如该文"wave 内跨 expert 数 → B weight L2 工作集"假设，用 wave 内 expert 数与收益比例的相关性验证。每篇 EXP 必须有明确的假设句和"数据如何支持/推翻它"。

### 9.6 常见陷阱清单
autotune 首次调用污染计时｜L2/TLB 冷热不一致（对照组交替跑而非分批跑）｜`CUDA_DEVICE_MAX_CONNECTIONS` 影响 stream 并发（固定在 env 段并写进 manifest）｜NCCL channel 数受 env 影响｜GPU 时钟漂移（长 sweep 中途插 a1 锚点复测）｜torch profiler 本身序列化 kernel（profile 数据只用于占比，不用于绝对时延）｜barrier 把最慢 rank 的时间摊给所有人（stage 计时要 per-rank 看）。

### 9.7 profile 工具决策表

| 问题 | 工具 |
|---|---|
| 哪个 kernel 占大头 | torch.profiler（内建 `--profile`） |
| stream 并发/overlap 是否真发生 | nsys（`nsys profile torchrun ...`，产物进 profiles/） |
| 融合 kernel 内部各段 | TD intra-kernel profiler（perfetto） |
| 单 kernel 为什么慢 | ncu（参考旧 `ncu_profile.sh` 的用法，脚本化进 microbench） |

---

## 10. 文档留痕规范

- 目录 `docs/exp/`，命名 `EXP-NNN-<slug>.md`，NNN 全局递增；`INDEX.md` 一行一条（编号、日期、一句话结论、状态）。
- 模板（每篇必含，空节写"无"）：

```markdown
# EXP-NNN: <标题>
> 日期 / 机器 / 本仓库 sha / TD sha / 配置文件路径 / 结果目录路径
## TL;DR（≤3 句：结论 + 数字）
## 背景与假设（要验证/证伪什么）
## 实验设计（控制变量表：固定了什么、只变了什么、怎么保证公平）
## 结果（表格，含抖动；原始数据指向 results.jsonl 路径，不手抄）
## 分析（数据如何支持假设；效率比值；反例讨论）
## 结论与后续（进入哪个 Phase / 产生哪个新 EXP）
```

- 纪律：实验先写"背景与假设"再跑；失败/无结论的实验**同样成文**（防止重复踩坑）；EXP 引用 run 目录必须是 manifest 完整的目录。
- `docs/exp/BLOCKERS.md`：执行中所有待用户决策的问题，格式"问题 / 影响 / 候选方案 / 建议"。

---

## 11. 最终报告（PPT 底稿）大纲 — `docs/report/final_report.md`

1. **背景与目标**：单层 MoE 通算融合方案对比；硬件（4×Pro5000 sm120 PCIe）与三个形状的业务来源。
2. **方案矩阵**：§6 的表 → 一页方案地图（串行/chunk/tile × TP/EP × 精度）。
3. **公平性方法**：统一 checkpoint、gate 路由、golden 验证体系（一页讲清为什么数字可信）。
4. **端到端结果**：三形状 × M sweep × 分布三档的时延/加速比曲线（vs a1）；每形状一页 + 一页总表。
5. **瓶颈拆解**：每方案 stage 占比堆叠图；exposed comm 对比；tile 融合 vs chunk overlap 的差距归因。
6. **深入分析案例**（每个 = 一篇 EXP 的浓缩，1–2 页）：dispatch 前重排代价、tile 排序与 L2 工作集、FP8 通信盈亏点、FP8-RS SM 划分、chunk 数 U 型曲线、imbalance 对 EP/TP 的不对称影响。
7. **我们做了什么 & 每步收益**：时间轴表——重构基线 → 各项优化/调优的 Δms 与 Δ%（数据来自 EXP-020 系列）。
8. **结论与后续**：哪种方案在哪个 (形状, M, 分布) 区间胜出（给出选型决策表）；已知限制（L2 pad 代价、sm120 特有问题）；下一步方向。

每张图有生成脚本（`docs/report/plots/*.py`，输入 results.jsonl），报告可随新数据一键重出。

---

## 12. 执行者工作纪律

1. 顺序执行 Phase，DoD 不过不前进；每完成一个实验/任务即更新 `docs/exp/INDEX.md` 状态。
2. **不修改**已安装的 vllm / 上游 triton_dist（tdx + compat monkeypatch 是唯一通道）。
3. 不为通过测试放宽容差/跳过验证；确需调整走 BLOCKERS.md 留痕。
4. 性能结论必须有"对照实验 + 上限比值"双证据（§9.3/9.5）。
5. commit 规范 `A<N>/S<N> <模块>: <改动>`；旧代码目录在 S7 前不许动。
6. 同一问题多次尝试（约 3 种思路）仍无法解决时：停止继续试错，写 BLOCKERS 条目，附最小复现，请示用户。
7. 本地阶段（A1–A5）专属：不得声称任何未经 GPU 验证的结论（"应该能跑"要写成 SERVER-VERIFY 标记，不是写进文档当事实）；每个 SERVER-VERIFY 必须同时进代码注释和 HANDOFF 汇总表，两处缺一不可。

## 13. 旧代码参考索引（只读，可参考不可照抄迁移）

| 新模块 | 参考旧代码 | 参考什么 |
|---|---|---|
| timing.py | `benchmarks/common.py: bench_cuda, profile_forward` | 计时循环骨架、comm/compute 分类关键词 |
| verify.py 输出格式 | `benchmarks/common.py: verify_against_reference` | max_abs/max_rel/cos_sim 打印格式 |
| data/checkpoint.py | `benchmarks/data.py: prepare_moe_weights` | 量化语义（新实现须与它逐位一致） |
| data/routing.py | `benchmarks/data.py: build_benchmark_data` 路由段 | gate 计算、all-gather、归一化细节 |
| schemes/shared_expert.py | `benchmarks/data.py: StaticSharedExpert` | vllm kernel 调用参数 |
| schemes/vllm_*.py | `benchmarks/vllm_cases.py` | fused_experts / quant_config / expert_map 用法 |
| schemes/overlap_*.py | `benchmarks/overlap_cases.py` + `moe_overlap/` | overlap state 构建与调用 |
| schemes/td_*.py | `benchmarks/td_cases.py`, `benchmarks/fp8_cases.py` | 层初始化次序（import 顺序坑）、ctx/finalize、triton.set_allocator |
| tdx/ | `{{TD_FORK}}` diff | §5.4 |
| microbench/mb_moe_align.py 等 | 《Tile 排序对 fused_moe GEMM 性能影响》.md、`test/ncu_*.py`、`ncu_profile.sh` | 实验方法与 NCU 用法 |
| cli.py sweep | `run.sh` sweep 段 | 逐点独立进程 + 汇总的模式 |

已知坑（旧代码注释里散落的，集中列出）：c3 的 import 顺序必须先 `fp8_ep_moe` 后 `ep_moe_fused`；c4/c5 需要 `triton.set_allocator`；NVSHMEM symmetric heap 大小进程启动即固定（大 M 需调 env 段）；`triton.language.target_info` 的 import error 无害。
