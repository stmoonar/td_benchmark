import pytest
import sys
import types

from moe_bench.config import load_config
from moe_bench.context import DistContext
from moe_bench.schemes import REGISTRY, build_scheme
from moe_bench.schemes.base import SchemeConfigError, validate_tunables
import moe_bench.overlap as overlap


def test_ab_registry_contains_specs_without_importing_external_gpu_packages():
    for code in ["a1", "a2", "b1", "b2", "b3"]:
        spec, builder = REGISTRY[code]
        assert spec.code == code
        assert spec.weight_dtype == "fp8"
        assert spec.act_quant == "group128"
        assert spec.requires_nvshmem is False
        assert callable(builder)


def test_overlap_package_is_import_safe_and_exposes_lazy_api_names():
    assert "MoEOverlapState" in overlap.__all__
    assert "moe_forward_overlap_fp8ag_fp8rs" in overlap.__all__


def test_overlap_tunable_defaults_and_unknown_key_validation():
    spec, _ = REGISTRY["b3"]

    values = validate_tunables(spec, {"n_chunks_down": 8})

    assert values["n_chunks_gateup"] == 2
    assert values["n_chunks_down"] == 8
    with pytest.raises(SchemeConfigError):
        validate_tunables(spec, {"not_a_tunable": 1})


def test_a1_and_a2_specs_preserve_baseline_parallel_modes():
    assert REGISTRY["a1"][0].parallel == "TP"
    assert REGISTRY["a2"][0].parallel == "EP"


class _DummyScale:
    def contiguous(self):
        return self


class _DummyTensor:
    shape = (8, 16)

    def contiguous(self):
        return self


class _DummyQuant:
    fp8 = _DummyTensor()
    scale = _DummyScale()
    bf16 = _DummyTensor()


class _DummyCkpt:
    w1 = _DummyQuant()
    w2 = _DummyQuant()
    shared_w1 = None
    shared_w2 = None


class _DummyRouting:
    topk_ids_local = _DummyTensor()
    topk_weights_local = _DummyTensor()
    topk_ids_full = _DummyTensor()
    topk_weights_full = _DummyTensor()


class _DummyData:
    hidden_local = _DummyTensor()
    ckpt = _DummyCkpt()
    routing = _DummyRouting()


def _install_fake_vllm(monkeypatch, calls):
    fused_moe = types.ModuleType("vllm.model_executor.layers.fused_moe.fused_moe")
    config = types.ModuleType("vllm.model_executor.layers.fused_moe.config")

    def fused_experts(**kwargs):
        calls.append(kwargs)
        return "vllm-result"

    def fp8_w8a8_moe_quant_config(**kwargs):
        return {"quant": kwargs}

    fused_moe.fused_experts = fused_experts
    config.fp8_w8a8_moe_quant_config = fp8_w8a8_moe_quant_config
    for name in [
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.fused_moe",
    ]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, fused_moe.__name__, fused_moe)
    monkeypatch.setitem(sys.modules, config.__name__, config)


def test_vllm_schemes_use_precomputed_routing():
    from pathlib import Path

    tp_source = Path("moe_bench/schemes/vllm_tp.py").read_text(encoding="utf-8")
    ep_source = Path("moe_bench/schemes/vllm_ep.py").read_text(encoding="utf-8")

    assert "routing_full(bundle)" in tp_source
    assert "routing_local(bundle)" in ep_source
    assert "gate_weight" not in tp_source
    assert "gate_weight" not in ep_source


def test_overlap_scheme_constructs_state_and_calls_forward(monkeypatch):
    calls = []

    class FakeState:
        def __init__(self, **kwargs):
            calls.append(("state", kwargs))

        def ensure_streams(self, device):
            calls.append(("streams", device))

    def fake_forward(**kwargs):
        calls.append(("forward", kwargs))
        return "overlap-result"

    monkeypatch.setitem(overlap.__dict__, "MoEOverlapState", FakeState)
    monkeypatch.setitem(overlap.__dict__, "moe_forward_overlap_fp8ag_fp8rs", fake_forward)
    cfg = load_config("configs/smoke.yaml", ["schemes.enabled=[b3]", "schemes.b3.tunables.n_chunks_down=7"])
    instance = build_scheme(cfg, DistContext(device="cuda:0"), _DummyData(), cfg.schemes[0])

    assert instance.run() == "overlap-result"
    assert calls[0] == ("state", {"n_chunks_gateup": 2, "n_chunks_down": 7, "tp_group": None})
    assert calls[-1][1]["w1"] is _DummyQuant.fp8
