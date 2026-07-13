# Experiment Index

| ID | Date | Summary | Status |
|---|---|---|---|
| EXP-A4 | 2026-07-04 | 本地 tdx 迁移审计草稿。 | draft |
| EXP-001 | 2026-07-04 | 初次 10 组 smoke。c 组解释与延迟数字已被 EXP-022/024 修正，见文首声明。 | superseded-partial |
| EXP-010 | 2026-07-05 | dispatch 前置链固定票价 0.19-0.22ms，对 M/E/top_k 不敏感。 | done |
| EXP-013 | 2026-07-05 | NCCL AG/RS 在 PCIe 上饱和于 32 GB/s，半带宽点 1-4MB。 | done |
| EXP-015 | 2026-07-05 | group128 量化 19-29us：小 M launch 地板，大 M 逼近带宽。 | done |
| EXP-020 | 2026-07-05 | c 组性能差归因（已被推翻，见文首声明与 EXP-024）。 | superseded |
| EXP-021 | 2026-07-05 | 校验门重校准：三道门（cos/rel_p99/scale）+ 阈值表 + rel 门局限说明。 | done |
| EXP-022 | 2026-07-05 | c 组 cos=0.33 根因=漏加 shared expert；修复后全过新门。 | done |
| EXP-023 | 2026-07-05 | a1 对账完成（Δ1.6%，采用新口径）；b3 差距归因证伪，转 EXP-025。 | done |
| EXP-024 | 2026-07-05 | c 组 3-6x 根因=计时污染（setup 在 run() 闭包内）；修复后 c3 全场最快。 | done |
| EXP-025 | 2026-07-05 | b3 退化初步排查（chunk配置）。根因已被 EXP-026 修正为 shared expert 实现退化。 | superseded |
| EXP-026 | 2026-07-05 | shared expert 退化修复：FP8直通+预构建runtime，b3恢复1.05x，b2恢复1.05x。 | done |
