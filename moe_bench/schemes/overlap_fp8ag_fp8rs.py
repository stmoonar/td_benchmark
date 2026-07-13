"""b3 dataflow: quantize -> chunk AG(fp8+scale) || GEMM -> RS(fp8+scale)."""

from __future__ import annotations

from typing import Any

from .base import SchemeInstance, SchemeSpec
from .overlap_common import OVERLAP_TUNABLES, build_overlap_instance


SPEC = SchemeSpec(
    code="b3",
    name="Overlap-FP8AG-FP8RS",
    family="chunk_overlap",
    parallel="TP",
    comm="quantize -> chunk AG(fp8+scale) || GEMM -> RS(fp8+scale)",
    weight_dtype="fp8",
    act_quant="group128",
    requires_nvshmem=False,
    tunables_schema=OVERLAP_TUNABLES,
)


def build(cfg: Any, ctx: Any, data: Any, scheme_cfg: Any) -> SchemeInstance:
    return build_overlap_instance(SPEC, "moe_forward_overlap_fp8ag_fp8rs", cfg, ctx, data, scheme_cfg)
