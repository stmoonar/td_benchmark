import sys
import types

from moe_bench.config import load_config
from moe_bench.context import DistContext
from moe_bench.schemes import REGISTRY, build_scheme
from moe_bench.schemes.base import validate_tunables
from tests.test_schemes_ab import _DummyData


def test_c_registry_contains_all_tdx_schemes():
    expected = {
        "c1": ("TD-EP-BF16", "EP", "bf16", "none"),
        "c2": ("TD-TP-BF16", "TP", "bf16", "none"),
        "c3": ("TD-EP-FP8", "EP", "fp8", "rowwise"),
        "c4": ("TD-TP-FP8", "TP", "fp8", "group128"),
        "c5": ("TD-TP-FP8-RS", "TP", "fp8", "group128"),
    }

    for code, (name, parallel, weight_dtype, act_quant) in expected.items():
        spec, builder = REGISTRY[code]
        assert spec.name == name
        assert spec.parallel == parallel
        assert spec.weight_dtype == weight_dtype
        assert spec.act_quant == act_quant
        assert spec.requires_nvshmem is True
        assert callable(builder)


def test_c3_tunables_have_defaults_and_bool_validation():
    spec, _ = REGISTRY["c3"]

    values = validate_tunables(spec, {})

    assert values["fp8_rs_enabled"] is True
    assert values["num_combine_sms"] == 8
    assert values["num_reduce_sms_in_combine"] == 100
    assert values["fp8_fwd_gemm_block_size_n"] == 128
    assert values["fp8_fwd_gemm_block_size_k"] == 128
    assert values["fp8_fwd_gemm_num_stages"] == 4
    assert values["gemm_group_size_m"] == 1


def test_c1_defaults_match_fork_sm120_tuning():
    spec, _ = REGISTRY["c1"]

    values = validate_tunables(spec, {})

    assert values["num_sm"] == 110
    assert values["bf16_fwd_gemm_block_size_n"] == 256
    assert values["bf16_fwd_gemm_block_size_k"] == 64
    assert values["bf16_fwd_gemm_num_stages"] == 3


def test_build_c_scheme_is_import_safe_before_triton_dist_runtime():
    cfg = load_config("configs/smoke.yaml", ["schemes.enabled=[c3]", "schemes.c3.tunables.fp8_rs_enabled=true"])

    instance = build_scheme(cfg, DistContext(), data=None, scheme_cfg=cfg.schemes[0])

    assert instance.diagnostics["scheme"] == "c3"
    assert instance.diagnostics["requires_nvshmem"] is True


def test_c_scheme_run_calls_tdx_adapter_with_data_and_tunables(monkeypatch):
    calls = []
    module = types.ModuleType("moe_bench.tdx.layers.fp8_ep_moe")

    def run_moe_bench_scheme(**kwargs):
        calls.append(kwargs)
        return "td-result"

    module.run_moe_bench_scheme = run_moe_bench_scheme
    monkeypatch.setitem(sys.modules, module.__name__, module)
    cfg = load_config("configs/smoke.yaml", ["schemes.enabled=[c3]"])
    instance = build_scheme(cfg, DistContext(nvshmem_initialized=True), _DummyData(), cfg.schemes[0])

    assert instance.run() == "td-result"
    assert calls[0]["scheme"] == "c3"
    assert calls[0]["data"] is not None
    assert calls[0]["tunables"]["fp8_rs_enabled"] is True
