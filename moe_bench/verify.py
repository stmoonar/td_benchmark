from __future__ import annotations

from typing import Any


def maybe_quant_dequant(x: Any, granularity: str) -> Any:
    if granularity == "none":
        return x.float()
    if granularity == "group128":
        return _group_quant_dequant(x, block=128)
    if granularity == "rowwise":
        return _rowwise_quant_dequant(x)
    raise ValueError(f"unknown activation quantization granularity {granularity}")


def golden_moe(
    hidden_full: Any,
    routing: Any,
    w1_bf16: Any,
    w2_bf16: Any,
    shared_w1_bf16: Any | None,
    shared_w2_bf16: Any | None,
    act_quant: str,
) -> Any:
    torch = _torch()
    functional = _torch_functional()
    hidden = hidden_full.float()
    tokens = hidden.shape[0]
    out_dim = w2_bf16.shape[1]
    out = torch.zeros((tokens, out_dim), dtype=torch.float32, device=hidden.device)
    for expert in range(w1_bf16.shape[0]):
        matches = (routing.topk_ids_full == expert).nonzero(as_tuple=False)
        if matches.numel() == 0:
            continue
        rows = matches[:, 0]
        slots = matches[:, 1]
        x = maybe_quant_dequant(hidden[rows], act_quant)
        gate, up = (x @ w1_bf16[expert].float().T).chunk(2, dim=-1)
        intermediate = functional.silu(gate) * up
        intermediate = maybe_quant_dequant(intermediate, act_quant)
        contribution = intermediate @ w2_bf16[expert].float().T
        weights = routing.topk_weights_full[rows, slots].float().unsqueeze(-1)
        out.index_add_(0, rows, contribution * weights)
    if shared_w1_bf16 is not None and shared_w2_bf16 is not None:
        x = maybe_quant_dequant(hidden, act_quant)
        gate, up = (x @ shared_w1_bf16[0].float().T).chunk(2, dim=-1)
        intermediate = maybe_quant_dequant(functional.silu(gate) * up, act_quant)
        out = out + intermediate @ shared_w2_bf16[0].float().T
    return out


def compare_outputs(
    actual: Any,
    expected: Any,
    atol: float,
    rtol: float,
    cos_threshold: float | None = None,
    rel_p99_threshold: float | None = None,
    scale_tolerance: float | None = None,
) -> dict[str, Any]:
    torch = _torch()
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    denom = expected_f.abs().clamp(min=1e-12)
    max_abs = float(diff.max()) if diff.numel() else 0.0
    max_rel = float((diff / denom).max()) if diff.numel() else 0.0
    cos_sim = float(torch.nn.functional.cosine_similarity(actual_f.flatten(), expected_f.flatten(), dim=0)) if diff.numel() else 1.0

    rel_p99 = 0.0
    rel_masked_max = 0.0
    if diff.numel():
        abs_expected = expected_f.abs()
        mask = abs_expected > 1e-3 * float(abs_expected.max())
        if bool(mask.any()):
            rel = (diff[mask] / abs_expected[mask]).flatten()
            k = max(1, int(0.99 * rel.numel()))
            rel_p99 = float(rel.kthvalue(k).values)
            rel_masked_max = float(rel.max())

    # 最小二乘缩放因子：抓"整体乘了个常数"类 bug（cos 对此完全不敏感）
    scale = 1.0
    if diff.numel():
        denom_sq = float((expected_f * expected_f).sum())
        if denom_sq > 0.0:
            scale = float((actual_f * expected_f).sum() / denom_sq)

    if cos_threshold is None and rel_p99_threshold is None and scale_tolerance is None:
        passed = bool(torch.all(diff <= (atol + rtol * expected_f.abs())))
    else:
        passed = True
        if cos_threshold is not None:
            passed = passed and (cos_sim >= cos_threshold)
        if rel_p99_threshold is not None:
            passed = passed and (rel_p99 <= rel_p99_threshold)
        if scale_tolerance is not None:
            passed = passed and (abs(scale - 1.0) <= scale_tolerance)

    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "cos_sim": cos_sim,
        "rel_p99": rel_p99,
        "rel_masked_max": rel_masked_max,
        "scale": scale,
        "pass": passed,
    }


def _group_quant_dequant(x: Any, block: int) -> Any:
    torch = _torch()
    functional = _torch_functional()
    original = x.shape[-1]
    padded_size = ((original + block - 1) // block) * block
    padded = functional.pad(x.float(), (0, padded_size - original))
    blocks = padded.view(*padded.shape[:-1], padded_size // block, block)
    scale = (blocks.abs().amax(dim=-1, keepdim=True) / 448.0).clamp(min=1e-12)
    quant = (blocks / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * scale
    return quant.view(*padded.shape)[..., :original].contiguous()


def _rowwise_quant_dequant(x: Any) -> Any:
    torch = _torch()
    scale = (x.float().abs().amax(dim=-1, keepdim=True) / 448.0).clamp(min=1e-12)
    return (x.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * scale


def _torch() -> Any:
    import torch

    return torch


def _torch_functional() -> Any:
    from torch.nn import functional

    return functional
