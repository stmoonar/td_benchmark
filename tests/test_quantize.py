import pytest

torch = pytest.importorskip("torch")

from moe_bench.data.checkpoint import FP8_MAX, build_checkpoint, dequantize_block128, quantize_block128
from moe_bench.config import ShapeCfg


def _loop_quantize_block128(w):
    e_count, n_dim, k_dim = w.shape
    n_blocks = (n_dim + 127) // 128
    k_blocks = (k_dim + 127) // 128
    scale = torch.empty((e_count, n_blocks, k_blocks), dtype=torch.float32, device=w.device)
    fp8 = torch.empty_like(w, dtype=torch.float8_e4m3fn)
    for e in range(e_count):
        for ni in range(n_blocks):
            for ki in range(k_blocks):
                n0, n1 = ni * 128, min((ni + 1) * 128, n_dim)
                k0, k1 = ki * 128, min((ki + 1) * 128, k_dim)
                block = w[e, n0:n1, k0:k1]
                sc = (block.abs().amax() / FP8_MAX).clamp(min=1e-12)
                scale[e, ni, ki] = sc
                fp8[e, n0:n1, k0:k1] = (block / sc).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return fp8.contiguous(), scale


def test_quantize_block128_matches_loop_oracle_for_ragged_shape():
    generator = torch.Generator(device="cpu").manual_seed(7)
    weight = torch.randn((2, 130, 65), generator=generator, dtype=torch.float32) * 0.01

    actual_fp8, actual_scale = quantize_block128(weight)
    expected_fp8, expected_scale = _loop_quantize_block128(weight)

    assert torch.equal(actual_fp8.view(torch.uint8), expected_fp8.view(torch.uint8))
    assert torch.equal(actual_scale, expected_scale)
    assert actual_scale.shape == (2, 2, 1)


def test_dequantize_block128_returns_bf16_view_shape():
    weight = torch.randn((1, 130, 65), dtype=torch.float32) * 0.01
    fp8, scale = quantize_block128(weight)

    dequant = dequantize_block128(fp8, scale)

    assert dequant.shape == weight.shape
    assert dequant.dtype == torch.bfloat16


def test_build_checkpoint_uses_expected_global_shapes():
    shape = ShapeCfg(M=8, K=6, E=3, top_k=1, n_gateup=8, n_down=4, shared_experts=1)

    ckpt = build_checkpoint(shape, seed=11, device="cpu")

    assert ckpt.w1.fp8.shape == (3, 8, 6)
    assert ckpt.w2.fp8.shape == (3, 6, 4)
    assert ckpt.shared_w1.fp8.shape == (1, 8, 6)
    assert ckpt.shared_w2.fp8.shape == (1, 6, 4)
    assert ckpt.gate_weight.shape == (3, 6)
