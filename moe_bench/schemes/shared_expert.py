"""Shared expert with pre-built static runtime (ported from old bench data.py).

Design contract (must not regress, see overlap_forward.py:511):
- BF16 input  -> quantize once via moe_kernel_quantize_input, then 2 fused GEMMs.
- FP8 input   -> used DIRECTLY with the caller-provided block input_scale;
                 no dequantize/requantize round-trip.
- All buffers, moe_align results and kernel config are built once per instance.
"""

from __future__ import annotations

from typing import Any


class StaticSharedExpert:
    def __init__(self, w1_fp8: Any, w1_scale: Any, w2_fp8: Any, w2_scale: Any,
                 block_shape: list[int], max_tokens: int) -> None:
        self.w1_fp8 = w1_fp8
        self.w1_scale = w1_scale
        self.w2_fp8 = w2_fp8
        self.w2_scale = w2_scale
        self.block_shape = block_shape
        self.max_tokens = max_tokens
        self.hidden_size = w1_fp8.shape[2]
        self.gateup_dim = w1_fp8.shape[1]
        self.intermediate_size = self.gateup_dim // 2
        self.output_size = w2_fp8.shape[1]
        self._runtime = None

    def _ensure_runtime(self):
        if self._runtime is not None:
            return self._runtime

        import torch
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            _get_config_dtype_str,
            dispatch_fused_moe_kernel as invoke_fused_moe_kernel,
            try_get_optimal_moe_config,
        )
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
        from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
        from vllm.triton_utils import tl

        device = self.w1_fp8.device
        topk_ids = torch.zeros(self.max_tokens, 1, device=device, dtype=torch.int32)
        config_dtype = _get_config_dtype_str(
            use_fp8_w8a8=True,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            ocp_mx_scheme=None,
            dtype=torch.bfloat16,
        )
        config = try_get_optimal_moe_config(
            self.w1_fp8.shape,
            self.w2_fp8.shape,
            1,
            config_dtype,
            self.max_tokens,
            block_shape=self.block_shape,
        )
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, config["BLOCK_SIZE_M"], 1, None)
        C_gateup = torch.empty(self.max_tokens, 1, self.gateup_dim, device=device, dtype=torch.bfloat16)
        intermediate = torch.empty(self.max_tokens, self.intermediate_size, device=device, dtype=torch.bfloat16)
        C_down = torch.empty(self.max_tokens, 1, self.output_size, device=device, dtype=torch.bfloat16)

        self._runtime = {
            "invoke": invoke_fused_moe_kernel,
            "quantize": moe_kernel_quantize_input,
            "compute_type": tl.bfloat16,
            "config": config,
            "sorted_token_ids": sorted_token_ids,
            "expert_ids": expert_ids,
            "num_tokens_post_padded": num_tokens_post_padded,
            "C_gateup": C_gateup,
            "intermediate": intermediate,
            "C_down": C_down,
        }
        return self._runtime

    def __call__(self, x: Any, input_scale: Any | None = None) -> Any:
        import torch

        rt = self._ensure_runtime()
        quant_dtype = torch.float8_e4m3fn
        if x.dtype == quant_dtype:
            if input_scale is None:
                raise ValueError("FP8 shared expert input requires input_scale")
            if input_scale.shape[0] != x.shape[0]:
                raise ValueError(
                    f"input_scale rows {input_scale.shape[0]} do not match input rows {x.shape[0]}")
            q_input = x.contiguous()
            a1_scale = input_scale.contiguous()
        else:
            q_input, a1_scale = rt["quantize"](x, None, quant_dtype, False, self.block_shape)

        rows = x.shape[0]
        C_gateup = rt["C_gateup"][:rows]
        intermediate = rt["intermediate"][:rows]
        C_down = rt["C_down"][:rows]

        rt["invoke"](
            q_input,
            self.w1_fp8,
            C_gateup,
            a1_scale,
            self.w1_scale,
            None,
            topk_weights=None,
            sorted_token_ids=rt["sorted_token_ids"],
            expert_ids=rt["expert_ids"],
            num_tokens_post_padded=rt["num_tokens_post_padded"],
            mul_routed_weight=False,
            top_k=1,
            config=rt["config"],
            compute_type=rt["compute_type"],
            use_fp8_w8a8=True,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=self.block_shape,
        )
        torch.ops._C.silu_and_mul(intermediate, C_gateup.view(-1, self.gateup_dim))
        q_intermediate, a2_scale = rt["quantize"](intermediate, None, quant_dtype, False, self.block_shape)
        rt["invoke"](
            q_intermediate,
            self.w2_fp8,
            C_down,
            a2_scale,
            self.w2_scale,
            None,
            topk_weights=None,
            sorted_token_ids=rt["sorted_token_ids"],
            expert_ids=rt["expert_ids"],
            num_tokens_post_padded=rt["num_tokens_post_padded"],
            mul_routed_weight=False,
            top_k=1,
            config=rt["config"],
            compute_type=rt["compute_type"],
            use_fp8_w8a8=True,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=self.block_shape,
        )
        return C_down.view(-1, self.output_size)[:rows]
