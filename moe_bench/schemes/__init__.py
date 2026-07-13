from __future__ import annotations

from typing import Any

from . import (
    overlap_bf16,
    overlap_fp8ag,
    overlap_fp8ag_fp8rs,
    td_ep_bf16,
    td_ep_fp8,
    td_tp_bf16,
    td_tp_fp8,
    td_tp_fp8_rs,
    vllm_ep,
    vllm_tp,
)
from .base import BuilderFn, SchemeConfigError, SchemeSpec


REGISTRY: dict[str, tuple[SchemeSpec, BuilderFn]] = {
    vllm_tp.SPEC.code: (vllm_tp.SPEC, vllm_tp.build),
    vllm_ep.SPEC.code: (vllm_ep.SPEC, vllm_ep.build),
    overlap_bf16.SPEC.code: (overlap_bf16.SPEC, overlap_bf16.build),
    overlap_fp8ag.SPEC.code: (overlap_fp8ag.SPEC, overlap_fp8ag.build),
    overlap_fp8ag_fp8rs.SPEC.code: (overlap_fp8ag_fp8rs.SPEC, overlap_fp8ag_fp8rs.build),
    td_ep_bf16.SPEC.code: (td_ep_bf16.SPEC, td_ep_bf16.build),
    td_tp_bf16.SPEC.code: (td_tp_bf16.SPEC, td_tp_bf16.build),
    td_ep_fp8.SPEC.code: (td_ep_fp8.SPEC, td_ep_fp8.build),
    td_tp_fp8.SPEC.code: (td_tp_fp8.SPEC, td_tp_fp8.build),
    td_tp_fp8_rs.SPEC.code: (td_tp_fp8_rs.SPEC, td_tp_fp8_rs.build),
}


def build_scheme(cfg: Any, ctx: Any, data: Any, scheme_cfg: Any):
    try:
        _spec, builder = REGISTRY[scheme_cfg.code]
    except KeyError as exc:
        raise SchemeConfigError(f"scheme {scheme_cfg.code!r} is not registered") from exc
    return builder(cfg, ctx, data, scheme_cfg)


__all__ = ["REGISTRY", "build_scheme", "SchemeConfigError", "SchemeSpec"]
