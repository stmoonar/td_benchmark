"""B1 sentinel: the verification gate must reject all-zero outputs."""

import torch

from moe_bench.verify import compare_outputs


def test_zero_output_fails_gate():
    torch.manual_seed(0)
    golden = torch.randn(256, 64)
    result = compare_outputs(
        torch.zeros_like(golden), golden, atol=0.2, rtol=0.0,
        cos_threshold=0.99, rel_p99_threshold=0.05,
    )
    assert result["pass"] is False


def test_near_identical_passes_gate():
    torch.manual_seed(0)
    golden = torch.randn(256, 64)
    noisy = golden * 1.0001
    result = compare_outputs(
        noisy, golden, atol=0.2, rtol=0.0,
        cos_threshold=0.999, rel_p99_threshold=0.05,
    )
    assert result["pass"] is True


def test_legacy_call_without_thresholds_keeps_old_behavior():
    golden = torch.ones(8, 8)
    result = compare_outputs(golden * 1.01, golden, atol=0.2, rtol=0.0)
    assert result["pass"] is True
    assert "rel_p99" in result
    assert "scale" in result


def test_uniformly_scaled_output_fails_gate():
    torch.manual_seed(0)
    golden = torch.randn(256, 64)
    result = compare_outputs(
        golden * 0.5, golden, atol=0.2, rtol=0.0,
        cos_threshold=0.99, rel_p99_threshold=5.0, scale_tolerance=0.05,
    )
    assert result["pass"] is False
    assert abs(result["scale"] - 0.5) < 0.01
