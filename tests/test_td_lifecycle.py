from __future__ import annotations

from types import SimpleNamespace

from moe_bench.schemes import td_common
from moe_bench.schemes.base import SchemeSpec


def test_td_instance_wires_layer_cleanup_into_close(monkeypatch) -> None:
    calls: list[str] = []
    spec = SchemeSpec(
        code="test",
        name="test",
        family="td_fused",
        parallel="EP",
        comm="test",
        weight_dtype="fp8",
        act_quant="group128",
        requires_nvshmem=True,
        tunables_schema={},
    )
    ctx = SimpleNamespace(rank=0, world_size=4)

    monkeypatch.setattr("moe_bench.tdx.compat.apply", lambda: None)
    monkeypatch.setattr(td_common.importlib, "import_module", lambda _target: object())
    monkeypatch.setattr(td_common, "require_data", lambda data: data)
    monkeypatch.setattr(td_common, "routing_local", lambda _data: (object(), object()))
    monkeypatch.setattr(td_common, "ep_weight_views", lambda *_args: {})
    monkeypatch.setattr(
        td_common,
        "_build_ep_fp8",
        lambda *_args: (lambda: "output", lambda: calls.append("close")),
    )

    instance = td_common.build_td_instance(
        spec,
        "fake.td.layer",
        cfg=object(),
        ctx=ctx,
        data=object(),
        scheme_cfg=SimpleNamespace(tunables={}),
    )

    assert instance.run() == "output"
    instance.close()
    assert calls == ["close"]
