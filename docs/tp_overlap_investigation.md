# TD-TP-FP8 vs Chunk Overlap Investigation

This branch isolates two schemes with matching communication precision:

- `b2`: chunk-level FP8 all-gather + FP8 GroupGEMM + BF16 reduce-scatter.
- `c4`: Triton Distributed tile-level FP8 all-gather + FP8 GroupGEMM + BF16 reduce-scatter.

`b1`, `b3`, `c3`, and `c5` are intentionally excluded. In particular, the
current `c5` implementation falls back to the same BF16 NVSHMEM ring used by
`c4`, so it is not evidence for an FP8 reduce-scatter comparison.

## One-command remote run

```bash
bash scripts/run_tp_overlap_investigation.sh
```

The script activates
`/data/cinnzhang_vllm_td_test/venvs/vllm-td/bin/activate`, selects an idle
four-GPU group, leaves `CUDA_DEVICE_MAX_CONNECTIONS` unset, and packages all
results and profiles into `results/tp_overlap_investigation_<timestamp>.zip`.

Useful controls:

```bash
# Quick end-to-end and profile smoke without long sweeps.
RUN_SWEEPS=0 WARMUP=10 REPEAT=20 bash scripts/run_tp_overlap_investigation.sh

# Skip Nsight Systems when it is unavailable.
RUN_NSYS=0 bash scripts/run_tp_overlap_investigation.sh

# Explicit GPU override; automatic priority selection remains the default.
GPUS=8,10,12,14 bash scripts/run_tp_overlap_investigation.sh
```

## Experiment matrix

| Case | Purpose |
|---|---|
| paired V1, shared on | Reproduce the full operator gap |
| paired V1, shared off | Remove shared-expert scheduling from the gap |
| M sweep | Separate fixed launch/synchronization overhead from throughput |
| routing sweep | Measure layout, padding, and load-imbalance sensitivity |
| b2 chunk sweep | Find the best coarse AG/RS granularity |
| c4 RS chunk sweep | Detect over-fragmentation from `n_chunks_rs=32` |
| component benchmark | Measure layout, group128 quantization, and exposed scale AG |
| torch profiler | Rank kernel time by scheme |
| Nsight Systems | Inspect actual stream overlap and exposed communication gaps |

The component benchmark uses the same four-rank MAX latency aggregation as the
end-to-end benchmark. It records both vLLM alignment variants, both TD routing
layout passes, b2/c4 group128 quantization, full FP8+scale NCCL all-gather, and
the c4 scale-only all-gather that must complete before its tile producer and
consumer are launched.

## Artifacts

- `analysis/comparison.md`: paired b2/c4 latency ratios and component table.
- `analysis/comparison.csv`: all focused result rows.
- `analysis/component_benchmarks.json`: preprocessing and communication costs.
- `*/profiles/torch/*.json`: Chrome/Perfetto-compatible torch traces.
- `*/profiles/torch/*_top_kernels.txt`: top CUDA kernels by total time.
- `profiles/nsys/*.nsys-rep`: steady-state multi-stream timelines.
- `validation_report.json`: fairness and sample-count checks.

Profile runs are separate from end-to-end timing runs. Profiler latency must not
be used as the headline performance number.
