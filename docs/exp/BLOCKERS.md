# Blockers

## 活跃
| Issue | Impact | 处理 |
|---|---|---|
| mb_tile_order_groupgemm.py 未实现 | EXP-012 缺失 | 需要对 upstream group_gemm 的 tile 元数据接口做现场核对后实现 |

## 已解决（留档）
| Issue | 解决方式 |
|---|---|
| c 组 warmup=5 时延迟虚高 3-6 倍 | 根因不是 JIT 而是计时污染（setup 在 run() 闭包内），已修，见 EXP-024。 |
| c 组校验 cos~0.33 | 漏加 shared expert，已修，见 EXP-022。 |
| und/gen 的 TP 组 intermediate_per_tp=192 不对齐 128 | runtime.py 的 tp_weight_views 自动走 L2 padding（192->256），结果行 shard_level=L2 标注。 |
| 校验门空洞（atol=0.2 绝对容差） | 三道门（cos/rel_p99/scale）+ 哨兵测试，见 EXP-021。 |
| a1 新旧口径差异 | a1 差 1.6% 属噪声，采用新口径（EXP-023）；b3 已修复见 EXP-026。 |
| b3 迁移退化 0.6ms | 根因=StaticSharedExpert 每调用重做 config/量化。修复后恢复到 1.05x vs a1（EXP-026）。 |
