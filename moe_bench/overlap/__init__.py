"""Migrated overlap runtime entry points.

The heavy vLLM/Triton imports live in ``overlap_forward`` and ``kernels``.
Keep this package import-safe locally; resolve the runtime symbols lazily on
the GPU server.
"""

from __future__ import annotations

_EXPORTS = {
    "MoEOverlapState",
    "moe_forward_overlap",
    "moe_forward_overlap_fp8ag",
    "moe_forward_overlap_fp8ag_fp8rs",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    from . import overlap_forward

    return getattr(overlap_forward, name)
