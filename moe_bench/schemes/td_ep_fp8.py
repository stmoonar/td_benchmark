"""c3 dataflow: rowwise quant -> ll-a2a FP8 dispatch + FP8 GroupGEMM -> combine."""

from __future__ import annotations

from typing import Any

from .base import SchemeInstance, SchemeSpec
from .td_common import FP8_EP_TUNABLES, build_td_instance


SPEC = SchemeSpec("c3", "TD-EP-FP8", "td_fused", "EP", "rowwise quant -> ll-a2a FP8 dispatch+GroupGEMM -> combine", "fp8", "rowwise", True, FP8_EP_TUNABLES)


def build(cfg: Any, ctx: Any, data: Any, scheme_cfg: Any) -> SchemeInstance:
    return build_td_instance(SPEC, "moe_bench.tdx.layers.fp8_ep_moe", cfg, ctx, data, scheme_cfg)
