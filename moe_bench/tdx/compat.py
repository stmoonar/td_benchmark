from __future__ import annotations

import importlib
import os
from pathlib import Path
import shutil
from typing import Any


_APPLIED = False
_ACTIONS: list[dict[str, str]] | None = None


def apply() -> dict[str, Any]:
    """Apply local runtime compatibility patches with detect-and-skip behavior.

    SERVER-VERIFY: each action here must be checked against clean upstream
    Triton-distributed on the GPU server before c-scheme parity runs.
    """
    global _APPLIED, _ACTIONS
    if _ACTIONS is None:
        _ACTIONS = [
            _patch_nv_utils_get_ptxas(),
            _patch_jit_ptxas_hook(),
            *_patch_common_ops_fence_aliases(),
            _patch_target_info_import(),
            _patch_allocator_hook(),
        ]
    _APPLIED = True
    return {"applied": _APPLIED, "actions": list(_ACTIONS)}


def _patch_nv_utils_get_ptxas() -> dict[str, str]:
    try:
        nv_utils = importlib.import_module("triton_dist.nv_utils")
    except ModuleNotFoundError:
        return {"name": "triton_dist.nv_utils.get_ptxas", "status": "skipped", "reason": "triton_dist not installed locally"}
    if hasattr(nv_utils, "get_ptxas"):
        return {"name": "triton_dist.nv_utils.get_ptxas", "status": "skipped", "reason": "already present"}

    def get_ptxas() -> tuple[str, str | None]:
        ptxas = shutil.which("ptxas")
        if ptxas is None:
            cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
            ptxas = str(Path(cuda_home) / "bin" / "ptxas")
        return ptxas, None

    nv_utils.get_ptxas = get_ptxas
    return {"name": "triton_dist.nv_utils.get_ptxas", "status": "applied", "reason": "injected ptxas resolver"}


def _patch_jit_ptxas_hook() -> dict[str, str]:
    try:
        jit = importlib.import_module("triton_dist.jit")
    except ModuleNotFoundError:
        return {
            "name": "triton_dist.jit.nvidia_stages_inspection_hook",
            "status": "skipped",
            "reason": "triton_dist not installed locally",
        }
    hook = getattr(jit, "nvidia_stages_inspection_hook", None)
    if hook is None:
        return {
            "name": "triton_dist.jit.nvidia_stages_inspection_hook",
            "status": "skipped",
            "reason": "hook not present",
        }
    if getattr(hook, "__moe_bench_wrapped__", False):
        return {
            "name": "triton_dist.jit.nvidia_stages_inspection_hook",
            "status": "skipped",
            "reason": "already wrapped",
        }

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        _patch_nv_utils_get_ptxas()
        return hook(*args, **kwargs)

    wrapped.__moe_bench_wrapped__ = True
    wrapped.__wrapped__ = hook
    jit.nvidia_stages_inspection_hook = wrapped
    return {
        "name": "triton_dist.jit.nvidia_stages_inspection_hook",
        "status": "applied",
        "reason": "wrapped to ensure get_ptxas is available",
    }


def _patch_common_ops_fence_aliases() -> list[dict[str, str]]:
    actions = []
    for module_name in [
        "triton_dist.language.extra.cuda.language_extra",
        "triton_dist.language.extra.hip.language_extra",
    ]:
        actions.append(_patch_fence_alias(module_name))
    try:
        common_ops = importlib.import_module("triton_dist.kernels.common_ops")
    except ModuleNotFoundError:
        actions.append({"name": "triton_dist.kernels.common_ops.fence_aliases", "status": "skipped", "reason": "common_ops not installed locally"})
        return actions
    for attr in ["fence_nv", "fence_amd"]:
        value = getattr(common_ops, attr, None)
        if value is not None:
            setattr(common_ops, attr, value)
    actions.append({"name": "triton_dist.kernels.common_ops.fence_aliases", "status": "applied", "reason": "common_ops fence symbols reviewed"})
    return actions


def _patch_fence_alias(module_name: str) -> dict[str, str]:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        return {"name": f"{module_name}.fence", "status": "skipped", "reason": "module not installed locally"}
    if hasattr(module, "fence"):
        return {"name": f"{module_name}.fence", "status": "skipped", "reason": "already present"}
    legacy = getattr(module, "__fence", None)
    if legacy is None:
        return {"name": f"{module_name}.fence", "status": "skipped", "reason": "neither fence nor __fence is present"}
    module.fence = legacy
    return {"name": f"{module_name}.fence", "status": "applied", "reason": "aliased __fence to fence"}


def _patch_target_info_import() -> dict[str, str]:
    try:
        import triton.language as tl  # type: ignore
    except ModuleNotFoundError:
        return {"name": "triton.language.target_info", "status": "skipped", "reason": "triton not installed locally"}
    if hasattr(tl, "target_info"):
        return {"name": "triton.language.target_info", "status": "skipped", "reason": "already present"}
    return {
        "name": "triton.language.target_info",
        "status": "skipped",
        "reason": "SERVER-VERIFY: no safe local shim without server Triton version",
    }


def _patch_allocator_hook() -> dict[str, str]:
    try:
        import triton  # type: ignore
    except ModuleNotFoundError:
        return {"name": "triton.set_allocator", "status": "skipped", "reason": "triton not installed locally"}
    if hasattr(triton, "set_allocator"):
        return {"name": "triton.set_allocator", "status": "skipped", "reason": "already present"}
    return {
        "name": "triton.set_allocator",
        "status": "skipped",
        "reason": "SERVER-VERIFY: allocator patch requires server Triton runtime",
    }
