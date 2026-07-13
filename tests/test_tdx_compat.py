import json
from pathlib import Path
import re
import sys
import types

import moe_bench.tdx.compat as compat
from moe_bench.tdx.compat import apply


def test_tdx_compat_apply_is_idempotent_and_records_detect_skip_actions():
    first = apply()
    second = apply()

    assert first["applied"] is True
    assert second["applied"] is True
    assert first["actions"]
    assert all("status" in action and "name" in action for action in first["actions"])


def test_tdx_compat_imports_without_triton_dist_installed():
    result = apply()

    assert any(action["status"] in {"applied", "skipped"} for action in result["actions"])


def test_tdx_migration_manifest_files_exist():
    manifest = json.loads(Path("docs/tdx_migration_manifest.json").read_text(encoding="utf-8"))

    assert manifest["td_upstream_commit"] == "1b9dc71a0a58585ac99766d739bf08ec60de4ae7"
    missing = [item["target"] for item in manifest["files"] if not Path(item["target"]).is_file()]

    assert missing == []


def test_tdx_migration_has_no_triton_dist_runtime_env_tunables():
    offenders = []
    for path in Path("moe_bench/tdx").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "TRITON_DIST_" in text or re.search(r"os\.(?:getenv|environ\.get)\(\s*f?[\"']TRITON_DIST_", text):
            offenders.append(path.as_posix())

    assert offenders == []


def test_tdx_runtime_tree_excludes_benchmark_scripts():
    assert list(Path("moe_bench/tdx").rglob("benchmark_*.py")) == []


def test_tdx_compat_patches_fake_clean_upstream_modules(monkeypatch):
    monkeypatch.setattr(compat, "_ACTIONS", None)
    monkeypatch.setattr(compat, "_APPLIED", False)

    nv_utils = types.ModuleType("triton_dist.nv_utils")
    jit = types.ModuleType("triton_dist.jit")
    cuda_extra = types.ModuleType("triton_dist.language.extra.cuda.language_extra")
    hip_extra = types.ModuleType("triton_dist.language.extra.hip.language_extra")
    common_ops = types.ModuleType("triton_dist.kernels.common_ops")
    cuda_extra.__fence = lambda: "cuda-fence"
    hip_extra.__fence = lambda: "hip-fence"
    common_ops.fence_nv = cuda_extra.__fence
    common_ops.fence_amd = hip_extra.__fence

    def hook(*args, **kwargs):
        return "hooked"

    jit.nvidia_stages_inspection_hook = hook
    for name in [
        "triton_dist",
        "triton_dist.language",
        "triton_dist.language.extra",
        "triton_dist.language.extra.cuda",
        "triton_dist.language.extra.hip",
        "triton_dist.kernels",
    ]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "triton_dist.nv_utils", nv_utils)
    monkeypatch.setitem(sys.modules, "triton_dist.jit", jit)
    monkeypatch.setitem(sys.modules, "triton_dist.language.extra.cuda.language_extra", cuda_extra)
    monkeypatch.setitem(sys.modules, "triton_dist.language.extra.hip.language_extra", hip_extra)
    monkeypatch.setitem(sys.modules, "triton_dist.kernels.common_ops", common_ops)

    result = apply()

    action_by_name = {action["name"]: action["status"] for action in result["actions"]}
    assert action_by_name["triton_dist.nv_utils.get_ptxas"] == "applied"
    assert action_by_name["triton_dist.jit.nvidia_stages_inspection_hook"] == "applied"
    assert action_by_name["triton_dist.language.extra.cuda.language_extra.fence"] == "applied"
    assert nv_utils.get_ptxas()[0].endswith("ptxas")
    assert jit.nvidia_stages_inspection_hook.__moe_bench_wrapped__ is True
    assert cuda_extra.fence() == "cuda-fence"
