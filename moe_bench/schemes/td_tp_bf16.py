"""c2 dataflow: NVSHMEM tile-level AG-GEMM + RS using BF16 weights."""

from __future__ import annotations

from typing import Any

from .base import SchemeInstance, SchemeSpec
from .td_common import BF16_TP_TUNABLES, build_td_instance


SPEC = SchemeSpec("c2", "TD-TP-BF16", "td_fused", "TP", "NVSHMEM tile-level AG-GEMM + RS", "bf16", "none", True, BF16_TP_TUNABLES)


def build(cfg: Any, ctx: Any, data: Any, scheme_cfg: Any) -> SchemeInstance:
    return build_td_instance(SPEC, "triton_dist.layers.nvidia.tp_moe", cfg, ctx, data, scheme_cfg)
