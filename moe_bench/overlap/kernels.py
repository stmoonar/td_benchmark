"""
MoE overlap 专用 Triton kernel 和封装函数。

包含:
  1. fused_moe_kernel_accumulate: K-split 累加模式的 triton kernel
     (IS_FIRST_CHUNK=True 覆盖写, IS_FIRST_CHUNK=False 累加)
  2. invoke_moe_kernel_n_split: N 维切分的 kernel 调用（Down GEMM 用）
  3. invoke_moe_kernel_k_split: K 维切分的累加 kernel 调用（GateUp GEMM 用）

从 demo (run_fused_moe_comm_overlap_bench.py) 移植而来，
适配 vllm 的 invoke_fused_moe_kernel API。
"""

from __future__ import annotations

from typing import Any

import torch

from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_moe_kernel,
    dispatch_fused_moe_kernel as invoke_fused_moe_kernel,
    try_get_optimal_moe_config,
    write_zeros_to_output,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.triton_utils import tl, triton


# ============================================================
# 1. K-split 累加 Triton Kernel
# ============================================================

@triton.jit
def fused_moe_kernel_accumulate(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_bias_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # Strides
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bbe,
    stride_bbn,
    # Block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # K-split scale offset
    k_scale_group_offset,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IS_FIRST_CHUNK: tl.constexpr,
):
    """
    与 vllm 的 fused_moe_kernel 基本相同，但支持 K-split 累加模式:
    - IS_FIRST_CHUNK=True: 直接写入 C (覆盖)
    - IS_FIRST_CHUNK=False: 先 load C 的旧值，加上新计算结果再 store (累加)
    - k_scale_group_offset: A_scale/B_scale 在 K group 维的偏移量
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        if IS_FIRST_CHUNK:
            write_zeros_to_output(
                c_ptr, stride_cm, stride_cn, pid_n, N,
                offs_token, token_mask,
                BLOCK_SIZE_M, BLOCK_SIZE_N, compute_type,
            )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    if use_int8_w8a16:
        b_scale_ptrs = (
            b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
        )
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8 or use_int8_w8a8:
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
            )
        elif per_channel_quant:
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
            )
            b_scale = tl.load(b_scale_ptrs)
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)

    if HAS_BIAS:
        bias_ptrs = b_bias_ptr + off_experts * stride_bbe + offs_bn * stride_bbn
        bias = tl.load(bias_ptrs, mask=(offs_bn < N), other=0.0)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_SIZE_K
                offs_ks = k_start // group_k + k_scale_group_offset
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
                accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            else:
                if use_fp8_w8a8:
                    accumulator = tl.dot(a, b, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b)
        else:
            accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if HAS_BIAS:
        accumulator = accumulator + bias[None, :]
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]
    if use_int8_w8a16:
        accumulator = (accumulator * b_scale).to(compute_type)
    elif use_fp8_w8a8 or use_int8_w8a8:
        if group_k > 0 and group_n > 0:
            accumulator = accumulator.to(compute_type)
        else:
            accumulator = (accumulator * a_scale * b_scale).to(compute_type)
    else:
        accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)

    if IS_FIRST_CHUNK:
        tl.store(c_ptrs, accumulator, mask=c_mask)
    else:
        old_c = tl.load(c_ptrs, mask=c_mask, other=0.0).to(compute_type)
        accumulator = old_c + accumulator
        tl.store(c_ptrs, accumulator, mask=c_mask)


# ============================================================
# 2. N 维切分的 kernel 调用封装（Down GEMM 用）
# ============================================================

def invoke_moe_kernel_n_split(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor | None,
    B_scale: torch.Tensor | None,
    B_zp: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    N_offset: int,
    N_chunk: int,
    block_shape: list[int] | None = None,
    B_bias: torch.Tensor | None = None,
) -> None:
    """
    N 维切分的 invoke_fused_moe_kernel 封装。

    在调用 vllm 原生的 invoke_fused_moe_kernel 之前，
    手动对 B、B_scale、C 做 N 维切片。

    Args:
        N_offset: B 矩阵在 N 维的偏移
        N_chunk: 本次计算的 N 维大小
    """
    E, full_N, K = B.shape

    # 切片 B: [E, N_chunk, K]
    B_slice = B[:, N_offset:N_offset + N_chunk, :]

    # 切片 B_scale（如果存在且是 block-wise 3D scale）
    B_scale_slice = B_scale
    if B_scale is not None and B_scale.ndim == 3 and block_shape is not None:
        group_n = block_shape[0]
        if group_n > 0:
            n_group_offset = N_offset // group_n
            n_group_chunk = N_chunk // group_n
            B_scale_slice = B_scale[:, n_group_offset:n_group_offset + n_group_chunk, :]

    # 切片 C: C 的形状是 [M, top_k, full_N] 或 [M*top_k, 1, full_N]
    # 我们需要切出 N 维的对应部分
    C_slice = C[:, :, N_offset:N_offset + N_chunk]

    # B_zp 切片（如果存在）
    B_zp_slice = B_zp
    if B_zp is not None and B_zp.ndim == 3 and block_shape is not None:
        group_n = block_shape[0]
        if group_n > 0:
            n_group_offset = N_offset // group_n
            n_group_chunk = N_chunk // group_n
            B_zp_slice = B_zp[:, n_group_offset:n_group_offset + n_group_chunk, :]

    # B_bias 切片
    B_bias_slice = B_bias
    if B_bias is not None:
        B_bias_slice = B_bias[:, N_offset:N_offset + N_chunk]

    invoke_fused_moe_kernel(
        A,
        B_slice,
        C_slice,
        A_scale,
        B_scale_slice,
        B_zp_slice,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
        top_k,
        config,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        block_shape=block_shape,
        B_bias=B_bias_slice,
    )


# ============================================================
# 3. K 维切分的累加 kernel 调用封装（GateUp GEMM 用）
# ============================================================

def invoke_moe_kernel_k_split(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor | None,
    B_scale: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    per_channel_quant: bool,
    K_offset: int,
    K_chunk: int,
    is_first_chunk: bool,
    block_shape: list[int] | None = None,
    A_K_offset: int = -1,
) -> None:
    """
    K 维切分的 fused_moe_kernel_accumulate 调用封装。

    将 GEMM 的 K 维切分为多个 chunk，每个 chunk 计算部分和并累加到 C：
      - A[:, A_K_offset:A_K_offset+K_chunk] @ B[:, :, K_offset:K_offset+K_chunk].T
      - 第一个 chunk: 直接写入 C（覆盖）
      - 后续 chunk: 累加到 C 的已有值上

    Args:
        K_offset: B 和 scale 在 K 维的偏移（对应 w1 的 K 维位置）
        K_chunk: 本次 K 维大小
        is_first_chunk: 是否为第一个 chunk
        block_shape: [group_n, group_k] for block-wise quantization
        A_K_offset: A 在 K 维的偏移。-1 表示使用 K_offset（A 和 B 用相同偏移）。
                    当 A 已经是 AG 后的 K 子段 [M, K_chunk] 时，应传 0。
    """
    if A_K_offset < 0:
        A_K_offset = K_offset

    M = A.size(0)
    num_tokens = M * top_k
    E, full_N, full_K = B.shape

    group_n = block_shape[0] if block_shape is not None else 0
    group_k = block_shape[1] if block_shape is not None else 0

    # 切分 A 的 K 维
    A_k_chunk = A[:, A_K_offset:A_K_offset + K_chunk]

    # 切分 B 的 K 维
    B_k_chunk = B[:, :, K_offset:K_offset + K_chunk]

    # 切分 scale
    A_scale_chunk = A_scale
    B_scale_chunk = B_scale
    k_scale_group_offset_val = 0

    if use_fp8_w8a8 or use_int8_w8a8:
        if group_k > 0 and A_scale is not None and A_scale.ndim == 2:
            # B_scale 始终用 K_offset 切（B 是完整的 w1）
            k_group_offset = K_offset // group_k
            num_k_groups = (K_chunk + group_k - 1) // group_k
            if B_scale is not None and B_scale.ndim == 3:
                B_scale_chunk = B_scale[:, :, k_group_offset:k_group_offset + num_k_groups]
            # A_scale 根据 A_K_offset 决定是否切片
            if A_K_offset == 0 and K_offset != 0:
                # A 和 A_scale 已经是 AG 后的 K 子段，不再切片
                A_scale_chunk = A_scale
            else:
                a_k_group_offset = A_K_offset // group_k
                A_scale_chunk = A_scale[:, a_k_group_offset:a_k_group_offset + num_k_groups]
            k_scale_group_offset_val = 0  # already sliced

    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        EM = min(EM, A.size(0) * top_k * config["BLOCK_SIZE_M"])

    config_copy = config.copy()
    BLOCK_SIZE_K = config_copy.pop("BLOCK_SIZE_K", 128)
    # fused_moe_kernel_accumulate 自身通过 IS_FIRST_CHUNK 实现 K-split 累加，
    # 不需要原始 fused_moe_kernel 的 SPLIT_K 参数（新版 vllm config 会带上此 key）
    config_copy.pop("SPLIT_K", None)
    if block_shape is not None:
        BLOCK_SIZE_K = min(BLOCK_SIZE_K, min(block_shape[0], block_shape[1]))

    grid = (
        triton.cdiv(EM, config_copy["BLOCK_SIZE_M"])
        * triton.cdiv(full_N, config_copy["BLOCK_SIZE_N"]),
    )

    fused_moe_kernel_accumulate[grid](
        A_k_chunk,
        B_k_chunk,
        C,
        None,  # B_bias
        A_scale_chunk,
        B_scale_chunk,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        full_N,
        K_chunk,
        EM,
        num_tokens,
        A_k_chunk.stride(0),
        A_k_chunk.stride(1),
        B_k_chunk.stride(0),
        B_k_chunk.stride(2),
        B_k_chunk.stride(1),
        C.stride(1),
        C.stride(2),
        A_scale_chunk.stride(0) if A_scale_chunk is not None and A_scale_chunk.ndim == 2 else 0,
        A_scale_chunk.stride(1) if A_scale_chunk is not None and A_scale_chunk.ndim == 2 else 0,
        B_scale_chunk.stride(0) if B_scale_chunk is not None and B_scale_chunk.ndim >= 2 else 0,
        B_scale_chunk.stride(2) if B_scale_chunk is not None and B_scale_chunk.ndim == 3 else 0,
        B_scale_chunk.stride(1) if B_scale_chunk is not None and B_scale_chunk.ndim >= 2 else 0,
        0, 0,  # stride_bbe, stride_bbn
        group_n,
        group_k,
        k_scale_group_offset=k_scale_group_offset_val,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=False,
        per_channel_quant=per_channel_quant,
        HAS_BIAS=False,
        IS_FIRST_CHUNK=is_first_chunk,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        **config_copy,
    )
