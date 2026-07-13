"""c4 dataflow: FP8 AG + FP8 GroupGEMM + BF16 RS."""

from __future__ import annotations

from typing import Any

from .base import SchemeInstance, SchemeSpec
from .td_common import FP8_TP_TUNABLES, build_td_instance


SPEC = SchemeSpec("c4", "TD-TP-FP8", "td_fused", "TP", "FP8 AG + FP8 GroupGEMM + RS(bf16)", "fp8", "group128", True, FP8_TP_TUNABLES)


def build(cfg: Any, ctx: Any, data: Any, scheme_cfg: Any) -> SchemeInstance:
    return build_td_instance(SPEC, "moe_bench.tdx.layers.fp8_tp_moe", cfg, ctx, data, scheme_cfg)
