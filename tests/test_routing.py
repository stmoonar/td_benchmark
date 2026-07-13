import pytest

torch = pytest.importorskip("torch")

from moe_bench.config import RoutingCfg, ShapeCfg
from moe_bench.context import DistContext
from moe_bench.data.routing import build_routing, target_distribution


def test_routing_is_deterministic_and_returns_normalized_fp32_weights():
    shape = ShapeCfg(M=16, K=4, E=4, top_k=2, n_gateup=8, n_down=4, shared_experts=0)
    hidden = torch.randn((16, 4), generator=torch.Generator().manual_seed(1), dtype=torch.bfloat16)
    gate = torch.randn((4, 4), generator=torch.Generator().manual_seed(2), dtype=torch.bfloat16)
    ctx = DistContext(rank=0, world_size=1, local_rank=0, device="cpu")

    first = build_routing(hidden, gate, RoutingCfg(), shape, ctx)
    second = build_routing(hidden, gate, RoutingCfg(), shape, ctx)

    assert torch.equal(first.topk_ids_full, second.topk_ids_full)
    assert torch.equal(first.topk_weights_full, second.topk_weights_full)
    assert first.topk_ids_local.dtype == torch.int32
    assert first.topk_weights_local.dtype == torch.float32
    assert torch.allclose(first.topk_weights_local.sum(dim=1), torch.ones(16))


def test_hotspot_bias_moves_more_tokens_to_hot_experts_than_none():
    shape = ShapeCfg(M=2000, K=4, E=8, top_k=1, n_gateup=8, n_down=4, shared_experts=0)
    hidden = torch.randn((2000, 4), generator=torch.Generator().manual_seed(1), dtype=torch.bfloat16)
    gate = torch.randn((8, 4), generator=torch.Generator().manual_seed(2), dtype=torch.bfloat16) * 0.01
    ctx = DistContext(rank=0, world_size=1, local_rank=0, device="cpu")

    base = build_routing(hidden, gate, RoutingCfg(imbalance_kind="none"), shape, ctx)
    hot = build_routing(
        hidden,
        gate,
        RoutingCfg(imbalance_kind="hotspot", hot_experts=2, hot_share=0.75, strength=1.0),
        shape,
        ctx,
    )

    assert hot.stats.hot_expert_share > base.stats.hot_expert_share
    assert hot.stats.hot_expert_share >= 0.45


def test_target_distribution_is_normalized():
    dist = target_distribution(RoutingCfg(imbalance_kind="zipf", zipf_s=1.2), experts=8, device="cpu")

    assert torch.all(dist > 0)
    assert torch.isclose(dist.sum(), torch.tensor(1.0))


def test_routing_uses_local_all_gather_stub_for_full_outputs_and_stats():
    shape = ShapeCfg(M=8, K=4, E=8, top_k=1, n_gateup=8, n_down=4, shared_experts=0)
    hidden = torch.randn((4, 4), generator=torch.Generator().manual_seed(3), dtype=torch.bfloat16)
    gate = torch.randn((8, 4), generator=torch.Generator().manual_seed(4), dtype=torch.bfloat16)
    ctx = DistContext(rank=1, world_size=2, local_rank=1, device="cpu")

    def gather_local(tensor):
        prefix = torch.zeros_like(tensor)
        return torch.cat([prefix, tensor], dim=0)

    routing = build_routing(hidden, gate, RoutingCfg(), shape, ctx, all_gather=gather_local)

    assert routing.topk_ids_full.shape == (8, 1)
    assert routing.topk_weights_full.shape == (8, 1)
    assert torch.equal(routing.topk_ids_full[:4], torch.zeros((4, 1), dtype=torch.int32))
    assert torch.equal(routing.topk_ids_full[4:], routing.topk_ids_local)
    assert int(routing.stats.per_expert_counts.sum()) == 8
