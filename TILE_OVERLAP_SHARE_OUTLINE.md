# 《MoE 通信-计算 Overlap 实战经验》分享文档大纲

> 定位：一篇写给做推理/训练性能优化同事的经验分享，回答三个问题——**为什么要做 overlap、怎么做、做了能有多少提升（以及什么时候没用）**。
> 素材状态标记：✅ 数据已有可直接用｜🕐 待 REWORK-B（标注依赖项）｜📐 原理性内容，现在就能写。
> 数据红线：只引用能追溯到 results/ 目录或 EXP 文档的数字；新旧 bench 数字未对账（EXP-023/024）前不混排；不出现代码量/commit 数这类工作量统计。
> 环境口径全文统一注明一次：4×RTX PRO 5000 72GB Blackwell（sm120），PCIe 5.0，无 NVLink；这是所有结论的边界条件。

---

## §1 背景：MoE 单层为什么被通信卡住 📐✅

**目标**：让读者在两分钟内认同"这台机器上通信是一等公民问题"。

要点：
- MoE 单层数据流五步：gate 路由 → dispatch 通信（TP AG / EP A2A）→ GroupGEMM(gate+up) → SiLU → GroupGEMM(down) → combine 通信（RS / combine），可选 shared expert 并行流。
- 关键硬件事实：NCCL AG/RS 在本机 PCIe 上**饱和于 ~32 GB/s**（✅ EXP-013 实测，带宽-消息大小曲线，256MB 处 32.0/32.1 GB/s）。
- 算一笔账：v1 shape（M=5120, K=4096）BF16 hidden AG ≈ 30MB ≈ 1.2–1.5ms，对照端到端 ~5.6ms，**串行方案 20%+ 的时间 GPU 计算在空转**。
- 一句话立论：通信藏不掉就是纯损失，而它原则上可以和计算并行。

图表：EXP-013 带宽曲线（消息大小 1KB→256MB，AG/RS 两条线，标出饱和平台）；串行时间轴示意图（通信段涂橙色标"空转"）。

## §2 收益模型：overlap 的天花板与成本 📐

**目标**：给读者一个能带走的心算公式，后面所有实验数据都往这个公式上挂。

要点：
- **净收益 = min(通信时间, 可重叠计算时间) − overlap 引入的开销**。
- 天花板：通信占比 × 端到端时间。通信占比 20% 的负载，overlap 做得再完美也只有 20% 提升空间——先测占比再动手。
- 成本清单（每一项后文都有对应实测）：
  - chunk 化使 GEMM 变小变碎，效率下降（🕐 EXP-016 chunk sweep）；
  - 额外量化/反量化 kernel，每次 ~20–29µs（✅ EXP-015）；
  - 多流同步、event 等待、kernel 启动排队；
  - tile-level 路线：preprocess 前置链 + host 同步点（🕐 EXP-011）。
- 推论：存在负收益区间（M 小、通信占比低），§5 专门讲。

图表：一张"收益-成本天平"示意图；η（重叠效率）定义框：η = (T_comm + T_comp − T_total) / T_comm。

## §3 路线一：chunk-level overlap（b 组，双流 + FP8 压缩通信）📐 + 🕐

**目标**：讲清最容易落地的做法，以及它的两个调优旋钮（chunk 数、通信精度）。

要点：
- 原理：把 M 维切成 n 个 chunk，通信流搬 chunk i+1 的同时计算流算 chunk i；三行时序图（通信流/计算流/串行对照）。
- 递进三档：b1 仅 chunk 重叠（BF16 通信）→ b2 AG 改 FP8（通信字节减半）→ b3 AG+RS 都 FP8。FP8 通信的字节账：BF16 30MB → FP8+scale ~15.5MB。
- 实现要点（踩坑向，📐 现在就能写）：
  - `CUDA_DEVICE_MAX_CONNECTIONS=1` 保证 kernel 入队顺序可控——**不设这个 overlap 时序直接乱**；
  - 双流 + event 依赖的正确挂法；chunk 间依赖只用 event，不用 stream sync；
  - FP8 RS 的 dequant-累加顺序与 scale 处理（老仓库 FP8_RS_OVERLAP_OPT.md 的迭代过程可浓缩为"三次尝试到超越 BF16"的小故事：5.365→5.235→5.120ms，✅ 老 bench 数据，注明口径）。
- 提升数据（🕐 全部待 EXP-023 对账 + B6 sweep）：
  - a1 vs b1/b2/b3 @ M sweep 曲线；
  - chunk 数 sweep：1/2/4/8/16 的 η 曲线与端到端曲线（🕐 EXP-016）——预期形态：chunk 太少藏不住通信、太多 GEMM 碎片化，中间有最优点；
  - **老/新 bench 当前矛盾必须先解决才能写数字**：老口径 b3 比 a1 快 26%，新口径慢 4%，差异集中在 a1 基线（EXP-023 出结论后按对账口径写）。

图表：三行时序图（复用 PPT_PLAN.md S3 图 B）；chunk 数-延迟 U 型曲线；M sweep 折线（a1 与 b3 交叉点即"overlap 开始值回票价"的 M）。

## §4 路线二：tile-level overlap（c 组，Triton-Distributed 融合 kernel）📐

**目标**：讲清"比 chunk 更细"的重叠为什么可能更好，机制细节直接复用 PPT_PLAN_EP_DEEPDIVE.md（D1–D8 已写完，图纸现成）。

要点：
- 核心思想：重叠粒度从 chunk（~百 µs）细到 GEMM tile（~µs）——数据到了一个 tile 的量就放行一个 tile 的计算，发送侧算完一个 tile 就立刻发。
- dispatch 的三种搬运方式（📐 D5 已有完整图纸）：one-stage 直达（重复发送换低延迟）vs two-stage 专家粒度放行 vs two-stage tile 粒度放行；选择逻辑 `num_tail_sms>0 → two-stage`、`USE_BLOCK_WISE_BARRIER → tile 粒度`。
- 两个融合 kernel：dispatch+GEMM①（到齐即算）与 GEMM②+combine（算完即发，fuse_scatter 源侧 topk_reduce）；FP8 变体连 scale 一起传。
- 代价面（诚实讲）：preprocess 前置链（argsort×2、splits 交换、scatter index、**pinned host 回读同步点**）+ NVSHMEM 上下文与 buffer 清零；这些是 tile-level 的"固定票价"，小 M 时占比会放大（🕐 EXP-010/011 出具体 µs）。
- β' tile 排序与 L2：tile 处理顺序影响权重复用的 L2 命中——✅ 老仓库实测 UND 322.8/401.0/329.7µs、GEN 558.4/602.5/578.5µs（三种排序，注明实验条件），🕐 EXP-012 在新框架复测。
- ⚠ 性能数字暂缓：新 bench 的 c 组当前比 fork 运行态慢 3–6 倍且正确性校验存疑（EXP-022/024 整改中），本节初稿只写机制与代价结构，数字等 EXP-024 出双口径结论后填。

图表：复用 PPT_PLAN_EP_DEEPDIVE.md 的 D1（全景四阶段）、D5（三种搬运方式三栏图）、D6/D7（两个融合 kernel 咬合图）；preprocess 开销条形图（🕐 EXP-011）。

## §5 什么时候 overlap 没用：负收益区间 📐 + 🕐

**目标**：这是全文最有辨识度的一节——大多数分享只讲收益，讲清"什么时候别做"更值钱。

要点：
- 由 §2 公式推出三个负收益条件：① 通信占比低（M 小或带宽高）；② chunk 化的 GEMM 效率损失 > 藏住的通信；③ 固定开销（量化 kernel、preprocess、同步）摊不薄。
- 用 M sweep 数据画"收益随 M 变化"曲线，标出盈亏平衡点（🕐 B6）。
- 如果对账后"新 bench b 组 0.96x"仍成立，把它作为正面案例展开：为什么这套配置下 overlap 不赚——而不是藏起来（🕐 EXP-023 之后定稿）。
- 决策清单（可截图带走）：先测通信占比 → 估天花板 → 数固定开销 → 再决定 chunk-level / tile-level / 不做。

## §6 microbench 方法论：让每个数字回答一个问题 📐 + ✅

**目标**：分享"怎么做性能分析"本身，呼应 microbench 应该解释这些问题的要求。

要点：
- 三问法：**为什么慢**（分段/分 kernel 归因，不许猜）→ **改了多少**（单变量对照）→ **为什么是这个数**（对着硬件上限算利用率）。
- 案例 1 ✅：EXP-013 带宽曲线回答"AG 1.2ms 是不是合理"——30MB / 32GB/s ≈ 0.94ms，加上启动开销即实测值，说明慢在链路不在实现。
- 案例 2 ✅：EXP-015 量化 kernel 19–29µs、大 M 时逼近带宽上限——回答"FP8 路线的量化税可以接受吗"。
- 反面案例（匿名化处理）：一次错误归因的教训——把 3 倍差距归因于"上游 kernel 没优化"，但 diff 证明该 kernel 两边完全相同；教训：**先 diff 代码、再分段计时，最后才允许下结论**（素材来自 EXP-020→EXP-024 修正过程）。
- 计时纪律清单：CUDA event 计时、warmup 覆盖 JIT（c 组 warmup=5 时数字虚高 3–6 倍的教训）、固定卡号（16 卡机不同子集 PCIe 拓扑不同）、校验容差必须与输出量级匹配（全零能 PASS 的教训）。

## §7 踩坑清单（速查表）📐

一页表格，每行：坑 → 症状 → 解法。已确定收录：
1. `CUDA_DEVICE_MAX_CONNECTIONS` 未设 → overlap 时序错乱/不生效。
2. NVSHMEM 环境变量（`DISABLE_CUDA_VMM`、`SYMMETRIC_SIZE`、`BOOTSTRAP`）→ 初始化失败或性能异常。
3. Triton JIT 编译混入计时 → warmup 不足时 c 组虚慢数倍。
4. 校验容差为绝对值且远大于输出量级 → 假 PASS。
5. 128×128 block scale 与 TP 切分不对齐（per-rank 192 列问题）→ L0 直切 / L1 重量化 / L2 padding 三级策略。
6. 16 卡机上换 GPU 子集对比 → PCIe 拓扑不同，数字不可比。
7. EP 输出 token 顺序与 golden 假设不一致 → cos_sim 崩到 0.33 的排查过程（🕐 EXP-022 定稿后补充结论）。

## §8 结论与选型建议 📐 + 🕐

要点：
- 一张选型表：负载特征（M 大小、通信占比、拓扑）→ 推荐路线（不做 / chunk-level / tile-level）。
- 本机（PCIe、4 卡）的最终推荐与数据支撑（🕐 待 B6 sweep + EXP-023/024）。
- 展望：NVLink 机器上结论会如何移动（带宽 ×10 → 通信占比骤降 → overlap 收益空间压缩，方法论不变）。

## 附录

- A. 全部数据表（带 results/ 目录回溯路径与机器/卡号/配置）。
- B. 复现命令（新 bench configs + microbench 脚本调用）。
- C. 术语表：AG/RS/A2A、chunk/tile、η、L0/L1/L2、one-stage/two-stage、fuse_scatter。

---

## 写作顺序建议（给执行者）

1. **现在就能写**：§1、§2、§4 机制部分、§6 框架与两个 ✅ 案例、§7 表格 1–6 行 —— 全部 📐/✅ 素材。
2. **REWORK-B 的 B1–B4 完成后**：§3 数字、§5 定稿、§6 反面案例定稿、§7 第 7 行。
3. **B5/B6 完成后**：三条骨干曲线（M sweep、chunk sweep、preprocess 条形图）入图，§8 选型表定稿。

## 依赖数据一览

| 图/数字 | 来源 | 状态 |
|---|---|---|
| 带宽-消息大小曲线 | EXP-013 jsonl | ✅ |
| 量化 kernel 开销 | EXP-015 jsonl | ✅ |
| β' tile 排序（老口径） | 老仓库 β' 报告 | ✅（注明条件） |
| b3 优化迭代小故事 | FP8_RS_OVERLAP_OPT.md | ✅（注明老口径） |
| a1 vs b 组提升幅度 | EXP-023 对账 + B6 sweep | 🕐 |
| chunk 数 U 型曲线 / η | EXP-016 | 🕐 |
| preprocess/dispatch 分段开销 | EXP-010/011 | 🕐 |
| c 组性能（双口径） | EXP-024 | 🕐 |
| β' 复测（新框架） | EXP-012 | 🕐 |
