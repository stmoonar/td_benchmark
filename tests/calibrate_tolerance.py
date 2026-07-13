from __future__ import annotations

import argparse
import json

import pytest

torch = pytest.importorskip("torch")

from moe_bench.config import ShapeCfg
from moe_bench.data.checkpoint import build_checkpoint
from moe_bench.data.routing import RoutingCfg, build_routing
from moe_bench.context import DistContext
from moe_bench.verify import golden_moe


def calibrate(shape: ShapeCfg, seed: int, act_quant: str) -> dict[str, float]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden = torch.randn((shape.M, shape.K), generator=generator, dtype=torch.bfloat16) * 0.01
    ckpt = build_checkpoint(shape, seed=seed, device="cpu")
    routing = build_routing(hidden, ckpt.gate_weight, RoutingCfg(), shape, DistContext())
    bf16 = golden_moe(hidden, routing, ckpt.w1.bf16, ckpt.w2.bf16, ckpt.shared_w1.bf16 if ckpt.shared_w1 else None, ckpt.shared_w2.bf16 if ckpt.shared_w2 else None, "none")
    fp8 = golden_moe(hidden, routing, ckpt.w1.bf16, ckpt.w2.bf16, ckpt.shared_w1.bf16 if ckpt.shared_w1 else None, ckpt.shared_w2.bf16 if ckpt.shared_w2 else None, act_quant)
    err = (fp8 - bf16).abs().flatten()
    return {"max_abs": float(err.max()), "p99_abs": float(torch.quantile(err, 0.99)), "tight_atol": float(err.max() * 1.5)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--act-quant", choices=["group128"], default="group128")
    args = parser.parse_args()
    shape = ShapeCfg(M=256, K=64, E=8, top_k=2, n_gateup=32, n_down=16, shared_experts=1)
    print(json.dumps(calibrate(shape, seed=42, act_quant=args.act_quant), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
