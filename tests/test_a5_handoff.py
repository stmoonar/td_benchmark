from pathlib import Path
import re

from moe_bench.config import load_config


def test_a5_configs_are_present_and_loadable():
    for path in [
        "configs/default.yaml",
        "configs/smoke.yaml",
        "configs/sweep_v1.yaml",
        "configs/sweep_und.yaml",
        "configs/sweep_gen.yaml",
        "configs/tune_c3.yaml",
    ]:
        assert Path(path).is_file()
        load_config(path)


IMPLEMENTED_MICROBENCH = [
    "mb_comm_prims.py",
    "mb_quant_kernels.py",
    "mb_moe_align.py",
    "mb_dispatch_prep.py",
    "mb_group_gemm.py",
    "mb_chunk_overlap.py",
]
PENDING_MICROBENCH = [
    "mb_tile_order_groupgemm.py",
]


def test_microbench_scripts_state():
    for name in IMPLEMENTED_MICROBENCH:
        path = Path("microbench") / name
        assert path.is_file()
        assert "SERVER-VERIFY" not in path.read_text(encoding="utf-8"), f"{name} 已实现却仍带 SERVER-VERIFY 标记"
    for name in PENDING_MICROBENCH:
        path = Path("microbench") / name
        assert path.is_file()
        assert "SERVER-VERIFY" in path.read_text(encoding="utf-8"), f"{name} 已实现请挪入 IMPLEMENTED_MICROBENCH"


def test_docs_and_handoff_include_required_phase_two_sections():
    required = [
        "docs/REFACTOR_PLAN.md",
        "docs/methodology.md",
        "docs/exp/INDEX.md",
        "docs/exp/BLOCKERS.md",
        "docs/exp/EXP-TEMPLATE.md",
        "docs/report/final_report.md",
        "HANDOFF.md",
    ]
    for path in required:
        assert Path(path).is_file()

    handoff = Path("HANDOFF.md").read_text(encoding="utf-8")
    for heading in [
        "Path Replacement Table",
        "Environment Preparation",
        "Dependency Versions",
        "S0-S7 Task Checklist",
        "SERVER-VERIFY Summary",
        "Known Risks",
    ]:
        assert heading in handoff

    for dependency in ["torch", "vllm", "triton", "triton_dist", "NVSHMEM"]:
        assert dependency in handoff


def test_phase_one_gitignore_excludes_generated_artifacts():
    gitignore = Path(".gitignore")

    assert gitignore.is_file()
    patterns = set(gitignore.read_text(encoding="utf-8").splitlines())
    for pattern in [
        "results/",
        "profiles/",
        "prof/",
        "__pycache__/",
        ".pytest_cache/",
        "*.py[cod]",
    ]:
        assert pattern in patterns


def test_handoff_server_verify_line_references_point_to_markers():
    handoff = Path("HANDOFF.md").read_text(encoding="utf-8")
    summary = handoff.split("## SERVER-VERIFY Summary", 1)[1].split("## Known Risks", 1)[0]
    current_file = None
    refs = []
    for ref in re.findall(r"`([^`]+)`", summary):
        for part in [item.strip() for item in ref.split(" and ")]:
            if not part:
                continue
            if part.startswith(":"):
                assert current_file is not None
                refs.append((current_file, int(part[1:])))
                continue
            if ":" not in part:
                continue
            path, line_text = part.rsplit(":", 1)
            if not line_text.isdigit():
                continue
            current_file = path
            refs.append((path, int(line_text)))

    for path, line_no in refs:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        assert "SERVER-VERIFY" in lines[line_no - 1], f"{path}:{line_no} is stale"
