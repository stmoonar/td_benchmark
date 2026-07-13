from pathlib import Path

import pytest

from moe_bench.schemes import REGISTRY
from moe_bench.schemes.base import SchemeConfigError, validate_tunables


def test_c_registry_contains_only_fp8_tdx_schemes():
    expected = {
        "c3": ("TD-EP-FP8", "EP", "fp8", "group128"),
        "c4": ("TD-TP-FP8", "TP", "fp8", "group128"),
        "c5": ("TD-TP-FP8-RS", "TP", "fp8", "group128"),
    }

    assert "c1" not in REGISTRY
    assert "c2" not in REGISTRY

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


def test_fp8_quantization_block_sizes_cannot_be_overridden():
    spec, _ = REGISTRY["c4"]

    with pytest.raises(SchemeConfigError):
        validate_tunables(spec, {"block_k_quant": 64})


def test_c4_gate_gemm_tunables_match_the_existing_kernel_defaults():
    spec, _ = REGISTRY["c4"]

    values = validate_tunables(spec, {})

    assert values["gemm_block_m"] == 128
    assert values["gemm_block_n"] == 128
    assert values["gemm_block_k"] == 128
    assert values["gemm_group_size_m"] == 8
    assert values["gemm_num_warps"] == 8
    assert values["gemm_num_stages"] == 4


def test_c3_source_uses_explicit_group128_activation_quantization():
    td_common = Path("moe_bench/schemes/td_common.py").read_text(encoding="utf-8")
    fp8_ep = Path("moe_bench/tdx/layers/fp8_ep_moe.py").read_text(encoding="utf-8")

    assert "_quantize_fp8_rowwise" not in td_common
    assert "_quantize_fp8_rowwise" not in fp8_ep
    assert "_quantize_fp8_blockwise(bundle.hidden_local, block_k=128)" in td_common


def test_investigation_config_uses_best_joint_gate_configuration():
    from moe_bench.config import load_config

    cfg = load_config("configs/investigate_tp_overlap.yaml")
    c4 = next(scheme for scheme in cfg.schemes if scheme.code == "c4")

    assert c4.tunables["gemm_block_m"] == 128
    assert c4.tunables["gemm_group_size_m"] == 1
    assert c4.tunables["gemm_num_warps"] == 8
    assert c4.tunables["gemm_num_stages"] == 2


def test_production_consumer_keeps_wait_enabled():
    source = Path("moe_bench/tdx/kernels/fp8_allgather_group_gemm.py").read_text(
        encoding="utf-8"
    )

    assert "WAIT_FOR_AG=wait_for_ag" in source
    assert "wait_for_ag=True" in source
