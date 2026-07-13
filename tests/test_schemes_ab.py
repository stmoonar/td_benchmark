import pytest
import sys
import types

from moe_bench.config import load_config
from moe_bench.context import DistContext
from moe_bench.schemes import REGISTRY, build_scheme
from moe_bench.schemes.base import SchemeConfigError, validate_tunables
from moe_bench.schemes.shared_expert import StaticSharedExpert
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


def test_build_scheme_is_import_safe_and_returns_diagnostics():
    cfg = load_config("configs/smoke.yaml", ["schemes.enabled=[b1]", "schemes.b1.tunables.n_chunks_gateup=4"])
    scheme_cfg = cfg.schemes[0]

    instance = build_scheme(cfg, DistContext(), data=None, scheme_cfg=scheme_cfg)

    assert instance.run_staged is not None
    assert instance.diagnostics["scheme"] == "b1"
    assert instance.diagnostics["tunables"]["n_chunks_gateup"] == 4
    assert instance.diagnostics["parallel"] == "TP"


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


def test_vllm_schemes_call_fused_experts_with_new_data_bundle(monkeypatch):
    calls = []
    _install_fake_vllm(monkeypatch, calls)
    cfg = load_config("configs/smoke.yaml", ["schemes.enabled=[a1,a2]"])

    outputs = []
    for scheme_cfg in cfg.schemes:
        instance = build_scheme(cfg, DistContext(), _DummyData(), scheme_cfg)
        outputs.append(instance.run())

    assert outputs == ["vllm-result", "vllm-result"]
    assert calls[0]["hidden_states"] is _DummyData.hidden_local
    assert calls[0]["topk_ids"] is _DummyRouting.topk_ids_full
    assert calls[1]["expert_map"] == "rank-local"
    assert calls[1]["topk_ids"] is _DummyRouting.topk_ids_full


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


def test_static_shared_expert_calls_vllm_fused_experts(monkeypatch):
    calls = []
    _install_fake_vllm(monkeypatch, calls)
    expert = StaticSharedExpert(
        _DummyQuant.fp8,
        _DummyQuant.scale,
        _DummyQuant.fp8,
        _DummyQuant.scale,
        [128, 128],
        8,
    )

    assert expert(_DummyData.hidden_local) == "vllm-result"
    assert calls[0]["hidden_states"] is _DummyData.hidden_local
    assert calls[0]["w1"] is _DummyQuant.fp8
