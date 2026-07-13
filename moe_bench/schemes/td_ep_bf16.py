"""c1 dataflow: NVSHMEM a2a dispatch + BF16 GroupGEMM + combine mega kernel."""

from __future__ import annotations

from typing import Any

from .base import SchemeInstance, SchemeSpec
from .td_common import BF16_EP_TUNABLES, build_td_instance


SPEC = SchemeSpec("c1", "TD-EP-BF16", "td_fused", "EP", "NVSHMEM a2a dispatch+GroupGEMM+combine mega kernel", "bf16", "none", True, BF16_EP_TUNABLES)


def build(cfg: Any, ctx: Any, data: Any, scheme_cfg: Any) -> SchemeInstance:
    return build_td_instance(SPEC, "moe_bench.tdx.layers.ep_moe", cfg, ctx, data, scheme_cfg)
