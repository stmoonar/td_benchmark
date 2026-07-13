from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from moe_bench.config import ShapeCfg


FP8_MAX = 448.0


@dataclass
class QuantizedTensor:
    fp8: Any
    scale: Any
    bf16: Any


@dataclass
class WeightCheckpoint:
    w1: QuantizedTensor
    w2: QuantizedTensor
    shared_w1: QuantizedTensor | None
    shared_w2: QuantizedTensor | None
    gate_weight: Any


def quantize_block128(w: Any) -> tuple[Any, Any]:
    torch = _torch()
    functional = _torch_functional()
    e_count, n_dim, k_dim = w.shape
    n_padded = _ceil_to(n_dim, 128)
    k_padded = _ceil_to(k_dim, 128)
    padded = functional.pad(w, (0, k_padded - k_dim, 0, n_padded - n_dim))
    blocks = padded.view(e_count, n_padded // 128, 128, k_padded // 128, 128)
    amax = blocks.abs().amax(dim=(2, 4))
    scale = (amax / FP8_MAX).clamp(min=1e-12)
    quant = (blocks / scale[:, :, None, :, None]).clamp(-FP8_MAX, FP8_MAX)
    fp8 = quant.view(e_count, n_padded, k_padded)[:, :n_dim, :k_dim].to(torch.float8_e4m3fn).contiguous()
    return fp8, scale.contiguous()


def dequantize_block128(fp8: Any, scale: Any) -> Any:
    torch = _torch()
    functional = _torch_functional()
    e_count, n_dim, k_dim = fp8.shape
    n_padded = _ceil_to(n_dim, 128)
    k_padded = _ceil_to(k_dim, 128)
    padded = functional.pad(fp8.float(), (0, k_padded - k_dim, 0, n_padded - n_dim))
    blocks = padded.view(e_count, n_padded // 128, 128, k_padded // 128, 128)
    dequant = blocks * scale[:, :, None, :, None].float()
    return dequant.view(e_count, n_padded, k_padded)[:, :n_dim, :k_dim].to(torch.bfloat16).contiguous()


def build_checkpoint(shape: ShapeCfg, seed: int, device: Any) -> WeightCheckpoint:
    torch = _torch()
    generator = torch.Generator(device=device).manual_seed(seed)
    w1_src = _randn(torch, (shape.E, shape.n_gateup, shape.K), generator, device, std=shape.K ** -0.5)
    w2_src = _randn(torch, (shape.E, shape.K, shape.n_down), generator, device, std=shape.n_down ** -0.5)
    w1 = _quantized(w1_src)
    w2 = _quantized(w2_src)
    shared_w1 = None
    shared_w2 = None
    if shape.shared_experts:
        shared_w1 = _quantized(_randn(torch, (1, 2 * shape.shared_intermediate, shape.K), generator, device, std=shape.K ** -0.5))
        shared_w2 = _quantized(_randn(torch, (1, shape.K, shape.shared_intermediate), generator, device, std=shape.shared_intermediate ** -0.5))
    gate_weight = _randn(torch, (shape.E, shape.K), generator, device, std=0.01).to(torch.bfloat16)
    return WeightCheckpoint(w1=w1, w2=w2, shared_w1=shared_w1, shared_w2=shared_w2, gate_weight=gate_weight)


def _quantized(w: Any) -> QuantizedTensor:
    fp8, scale = quantize_block128(w)
    return QuantizedTensor(fp8=fp8, scale=scale, bf16=dequantize_block128(fp8, scale))


def _randn(torch: Any, shape: tuple[int, ...], generator: Any, device: Any, std: float) -> Any:
    return torch.randn(shape, generator=generator, device=device, dtype=torch.float32) * std


def _ceil_to(value: int, block: int) -> int:
    return ((value + block - 1) // block) * block


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("A2 tensor operations require torch; import-only module checks do not.") from exc
    return torch


def _torch_functional() -> Any:
    from torch.nn import functional

    return functional
