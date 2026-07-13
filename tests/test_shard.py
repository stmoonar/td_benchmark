import pytest

torch = pytest.importorskip("torch")

from moe_bench.data.checkpoint import dequantize_block128, quantize_block128
from moe_bench.data.shard import ShardPlan, shard_columns, shard_with_padding


def test_l0_shard_uses_direct_fp8_and_scale_slice_on_128_aligned_boundary():
    weight = torch.randn((1, 256, 16), dtype=torch.float32) * 0.01
    fp8, scale = quantize_block128(weight)

    shard = shard_columns(fp8, scale, dequantize_block128(fp8, scale), rank=1, world_size=2)

    assert shard.level == "L0"
    assert torch.equal(shard.fp8.view(torch.uint8), fp8[:, 128:256, :].contiguous().view(torch.uint8))
    assert torch.equal(shard.scale, scale[:, 1:2, :].contiguous())


def test_l1_shard_requantizes_bf16_when_boundary_is_ragged():
    weight = torch.randn((1, 384, 16), dtype=torch.float32) * 0.01
    fp8, scale = quantize_block128(weight)
    bf16 = dequantize_block128(fp8, scale)

    shard = shard_columns(fp8, scale, bf16, rank=1, world_size=2)

    assert shard.level == "L1"
    assert shard.fp8.shape == (1, 192, 16)
    assert shard.scale.shape == (1, 2, 1)


def test_l2_padding_preserves_original_values_and_records_overhead():
    weight = torch.randn((1, 384, 16), dtype=torch.float32) * 0.01
    _, _, bf16 = quantize_block128(weight)[0], None, weight.to(torch.bfloat16)

    shard = shard_with_padding(bf16, rank=1, world_size=2, block=128)

    assert shard.level == "L2"
    assert shard.padded_columns == 256
    assert shard.original_columns == 192
    assert shard.compute_overhead == pytest.approx(256 / 192)
    assert torch.equal(shard.bf16[:, :192, :], bf16[:, 192:384, :])
    assert torch.count_nonzero(shard.bf16[:, 192:, :]) == 0


def test_shard_plan_reports_aligned_and_ragged_levels_without_torch_values():
    assert ShardPlan(columns=3072, world_size=4).preferred_level() == "L0"
    assert ShardPlan(columns=768, world_size=4, require_aligned_kernel=True).preferred_level() == "L2"
    assert ShardPlan(columns=384, world_size=2).preferred_level() == "L1"
