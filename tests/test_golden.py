import pytest

torch = pytest.importorskip("torch")

from moe_bench.data.routing import RoutingBundle, RoutingStats
from moe_bench.verify import compare_outputs, golden_moe, maybe_quant_dequant


def test_golden_moe_e1_topk1_matches_direct_ffn():
    hidden = torch.randn((5, 3), generator=torch.Generator().manual_seed(3), dtype=torch.bfloat16)
    w1 = torch.randn((1, 4, 3), generator=torch.Generator().manual_seed(4), dtype=torch.bfloat16)
    w2 = torch.randn((1, 3, 2), generator=torch.Generator().manual_seed(5), dtype=torch.bfloat16)
    routing = RoutingBundle(
        topk_ids_local=torch.zeros((5, 1), dtype=torch.int32),
        topk_weights_local=torch.ones((5, 1), dtype=torch.float32),
        topk_ids_full=torch.zeros((5, 1), dtype=torch.int32),
        topk_weights_full=torch.ones((5, 1), dtype=torch.float32),
        stats=RoutingStats(torch.tensor([5], dtype=torch.int32), 1.0, 0.0, torch.tensor([5], dtype=torch.int32), 1.0),
    )

    golden = golden_moe(hidden, routing, w1, w2, None, None, act_quant="none")
    gate, up = (hidden.float() @ w1[0].float().T).chunk(2, dim=-1)
    direct = torch.nn.functional.silu(gate) * up
    direct = direct @ w2[0].float().T

    assert torch.allclose(golden, direct)


def test_group128_quant_dequant_preserves_shape_and_is_close_for_small_values():
    x = torch.randn((3, 129), generator=torch.Generator().manual_seed(6), dtype=torch.float32) * 0.01

    qdq = maybe_quant_dequant(x, "group128")

    assert qdq.shape == x.shape
    assert (qdq - x).abs().max() < 2e-3


def test_compare_outputs_reports_required_metrics_and_pass_flag():
    ref = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    actual = torch.tensor([[1.01, 1.99]], dtype=torch.float32)

    report = compare_outputs(actual, ref, atol=0.02, rtol=0.0)

    assert report["pass"] is True
    assert {"max_abs", "max_rel", "cos_sim", "pass"} <= set(report)
