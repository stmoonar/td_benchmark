import pytest

torch = pytest.importorskip("torch")

from moe_bench.config import ShapeCfg, load_config, replace_cfg
from moe_bench.context import DistContext
from moe_bench.data.bundle import build_data_bundle


def test_build_data_bundle_assembles_rank_local_inputs_and_golden_outputs():
    cfg = load_config("configs/smoke.yaml")
    ctx = DistContext(rank=0, world_size=1, local_rank=0, device="cpu")

    bundle = build_data_bundle(cfg, ctx, device="cpu")

    assert bundle.hidden_local.shape == (cfg.shape.M, cfg.shape.K)
    assert bundle.hidden_local.dtype == torch.bfloat16
    assert bundle.ckpt.w1.fp8.shape == (cfg.shape.E, cfg.shape.n_gateup, cfg.shape.K)
    assert bundle.ckpt.w2.fp8.shape == (cfg.shape.E, cfg.shape.K, cfg.shape.n_down)
    assert bundle.routing.topk_ids_local.shape == (cfg.shape.M, cfg.shape.top_k)
    assert bundle.routing.topk_weights_local.shape == (cfg.shape.M, cfg.shape.top_k)
    assert bundle.golden.bf16.shape == (cfg.shape.M, cfg.shape.K)
    assert bundle.golden.fp8sim_group128.shape == (cfg.shape.M, cfg.shape.K)
    assert bundle.golden.fp8sim_rowwise.shape == (cfg.shape.M, cfg.shape.K)


def test_build_data_bundle_is_deterministic_for_same_config_and_rank():
    cfg = load_config("configs/smoke.yaml")
    ctx = DistContext(rank=0, world_size=1, local_rank=0, device="cpu")

    first = build_data_bundle(cfg, ctx, device="cpu")
    second = build_data_bundle(cfg, ctx, device="cpu")

    assert torch.equal(first.hidden_local, second.hidden_local)
    assert torch.equal(first.routing.topk_ids_local, second.routing.topk_ids_local)
    assert torch.equal(first.golden.bf16, second.golden.bf16)


def test_build_data_bundle_aligns_m_and_slices_by_rank():
    cfg = replace_cfg(
        load_config("configs/smoke.yaml"),
        shape=ShapeCfg(M=5, K=8, E=4, top_k=2, n_gateup=8, n_down=4, shared_experts=0),
    )
    rank0 = DistContext(rank=0, world_size=2, local_rank=0, device="cpu")
    rank1 = DistContext(rank=1, world_size=2, local_rank=1, device="cpu")

    first = build_data_bundle(cfg, rank0, device="cpu")
    second = build_data_bundle(cfg, rank1, device="cpu")

    assert first.hidden_local.shape == (3, 8)
    assert second.hidden_local.shape == (3, 8)
    assert not torch.equal(first.hidden_local, second.hidden_local)
