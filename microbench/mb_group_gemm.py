"""FP8 GroupGEMM efficiency vs token-to-expert distribution (uniform/zipf/hotspot)."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch


def make_topk(M, E, top_k, dist_kind, device):
    gen = torch.Generator(device=device).manual_seed(42)
    if dist_kind == "uniform":
        probs = torch.ones(E, device=device)
    elif dist_kind == "zipf":
        ranks = torch.arange(1, E + 1, device=device, dtype=torch.float32)
        probs = ranks.pow(-1.2)
    elif dist_kind == "hotspot":
        probs = torch.ones(E, device=device)
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
                "M": M, "distribution": dist_kind, "latency_ms": med,
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

from common import run_microbench_plan


def main() -> int:
    return run_microbench_plan(
        __file__,
        "EXP-014",
        "FP8/BF16 GroupGEMM efficiency versus token distribution",
        {"dtype": ["bf16", "fp8"], "distribution": ["uniform", "zipf", "hotspot"], "tile": [64, 128, 256]},
        ["latency_ms", "tflops", "tile_utilization"],
        "implement GroupGEMM distribution sweep on GPU server",
    )


if __name__ == "__main__":
    raise SystemExit(main())
