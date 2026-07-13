import importlib
from pathlib import Path


def test_a2_modules_are_importable_without_vllm_triton_or_cuda():
    for module_name in [
        "moe_bench.data",
        "moe_bench.data.checkpoint",
        "moe_bench.data.shard",
        "moe_bench.data.routing",
        "moe_bench.data.bundle",
        "moe_bench.verify",
    ]:
        importlib.import_module(module_name)


def test_calibrate_tolerance_script_exists_for_server_phase():
    assert Path("tests/calibrate_tolerance.py").is_file()
