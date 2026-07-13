"""
MoE 通信-计算 Overlap Forward 实现。

包含两个版本：
  1. moe_forward_overlap       — BF16 AG 版（原始实现）
  2. moe_forward_overlap_fp8ag — FP8 AG 版（AG 前量化 BF16→FP8，通信量减半）

FP8 AG 版核心改动：
  - AG 前将 hidden_states 从 BF16 量化为 FP8 (block-wise)，通信量减半
  - AG 额外传输一个小的 scale tensor (~32KB/chunk)
  - GateUp GEMM 直接用 FP8 输入 + scale，无需反量化
  - Shared Expert 接收 FP8 AG 结果和对应 block scale
  - 使用分离式排队（减少 stream context switch）
  - C_gateup 输出为 bf16（与 baseline 一致，避免多余的 fp32 store + 转换）
  - NCCL 兼容：FP8 tensor 通过 .view(torch.uint8) 传输，避免 sm90 检查

完整 TP overlap 数据流：
  Attention RS 后: [M, K] partial sum → RS 沿 M scatter → [M/tp, K]
    每个 rank 持有 M/tp 个 token，K 维完整
  Local Gate Routing: [M/tp, K] × W_gate[K, E] → topk_ids[M/tp, top_k]
  Phase 1: AG(K-split) + GateUp GEMM overlap
    AG 沿 M 维 gather: [M/tp, K/n_chunks] → [M, K/n_chunks]
    GateUp GEMM K-split: partial sum 累加
  Phase 2: SiLU
  Phase 3: Down GEMM(N-split) + RS overlap
    Down GEMM 沿 N 维分 chunk
    RS 沿 M 维 scatter: [M, N_chunk] → [M/tp, N_chunk]
  输出: [M/tp, K_out]
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist

from vllm.model_executor.layers.fused_moe.fused_moe import (
    _get_config_dtype_str,
    dispatch_fused_moe_kernel as invoke_fused_moe_kernel,
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from moe_bench.overlap.kernels import (
    invoke_moe_kernel_k_split,
    invoke_moe_kernel_n_split,
)
from vllm.triton_utils import tl

import logging
logger = logging.getLogger(__name__)


@dataclass
class MoEOverlapState:
    """MoE overlap 运行时状态，在模型 __init__ 时创建一次，forward 时复用。"""

    n_chunks_gateup: int = 4
    n_chunks_down: int = 4
    tp_group: Optional[dist.ProcessGroup] = field(default=None, repr=False)

    _compute_stream: Optional[torch.cuda.Stream] = field(
        default=None, repr=False, init=False
    )
    _comm_stream: Optional[torch.cuda.Stream] = field(
        default=None, repr=False, init=False
    )
    _shared_stream: Optional[torch.cuda.Stream] = field(
        default=None, repr=False, init=False
    )

    @property
    def tp_size(self) -> int:
        if self.tp_group is not None:
            return dist.get_world_size(self.tp_group)
        return dist.get_world_size()

    def ensure_streams(self, device: torch.device):
        if self._compute_stream is None:
            low_pri, high_pri = torch.cuda.Stream.priority_range()
            self._compute_stream = torch.cuda.Stream(device, priority=low_pri)
            self._comm_stream = torch.cuda.Stream(device, priority=high_pri)
            self._shared_stream = torch.cuda.Stream(device, priority=low_pri)

    @classmethod
    def from_env(cls, tp_group=None) -> "MoEOverlapState":
        return cls(
            n_chunks_gateup=int(os.getenv("MOE_OVERLAP_CHUNKS_GATEUP", "4")),
            n_chunks_down=int(os.getenv("MOE_OVERLAP_CHUNKS_DOWN", "4")),
            tp_group=tp_group,
        )


def moe_forward_overlap(
    state: MoEOverlapState,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    global_num_experts: int,
    expert_map: Optional[torch.Tensor] = None,
    shared_mlp: Optional[torch.nn.Module] = None,
    M_valid: Optional[int] = None,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
) -> torch.Tensor:
    """BF16 AG 版 MoE overlap forward（分离式排队实现）。"""
    torch.cuda.nvtx.range_push("MOE_OVERLAP_FORWARD")

    device = hidden_states.device
    state.ensure_streams(device)
    compute_stream = state._compute_stream
    comm_stream = state._comm_stream
    shared_stream = state._shared_stream
    tp_size = state.tp_size
    tp_group = state.tp_group

    caller_stream = torch.cuda.current_stream()

    M_local, K = hidden_states.shape
    M = M_local * tp_size
    E, N_gateup, _ = w1.shape
    K_out = w2.shape[1]

    M_moe = M_valid if (M_valid is not None and M_valid < M) else M

    n_chunks_gateup = state.n_chunks_gateup
    n_chunks_down = state.n_chunks_down
    K_chunk = K // n_chunks_gateup
    N_down_chunk = K_out // n_chunks_down

    # ================================================================
    # Step 0: Triton config + routing
    # ================================================================
    config_dtype = _get_config_dtype_str(
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        ocp_mx_scheme=None,
        dtype=hidden_states.dtype,
    )
    config = try_get_optimal_moe_config(
        w1.shape, w2.shape, top_k, config_dtype, M_moe, block_shape=block_shape,
    )
    BLOCK_SIZE_M = config["BLOCK_SIZE_M"]

    topk_ids_moe = topk_ids[:M_moe]
    topk_weights_moe = topk_weights[:M_moe]
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids_moe, BLOCK_SIZE_M, global_num_experts, expert_map
    )

    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    else:
        compute_type = tl.float32

    quant_dtype = None
    if use_fp8_w8a8:
        quant_dtype = torch.float8_e4m3fn
    elif use_int8_w8a8:
        quant_dtype = torch.int8

    # ================================================================
    # Step 1: 分配 buffer
    # ================================================================
    C_gateup = torch.empty(
        M_moe, top_k, N_gateup, device=device, dtype=hidden_states.dtype,
    )
    intermediate_cache = torch.empty(
        M_moe * top_k, N_gateup // 2, device=device, dtype=hidden_states.dtype,
    )
    C_down = torch.empty(
        M_moe * top_k, 1, K_out, device=device, dtype=hidden_states.dtype,
    )

    ag_inputs = [
        hidden_states[:, c * K_chunk:(c + 1) * K_chunk].contiguous()
        for c in range(n_chunks_gateup)
    ]
    ag_outputs = [
        torch.empty(M, K_chunk, device=device, dtype=hidden_states.dtype)
        for _ in range(n_chunks_gateup)
    ]

    rs_inputs = [
        torch.zeros(M, N_down_chunk, device=device, dtype=hidden_states.dtype)
        for _ in range(n_chunks_down)
    ]
    rs_outputs = [
        torch.empty(M_local, N_down_chunk, device=device, dtype=hidden_states.dtype)
        for _ in range(n_chunks_down)
    ]

    # Events
    entry_ev = caller_stream.record_event()
    ag_evs = [torch.cuda.Event(enable_timing=False) for _ in range(n_chunks_gateup)]
    shared_ev = torch.cuda.Event(enable_timing=False)
    comp_evs = [torch.cuda.Event(enable_timing=False) for _ in range(n_chunks_down)]

    comm_stream.wait_event(entry_ev)
    compute_stream.wait_event(entry_ev)
    shared_stream.wait_event(entry_ev)

    shared_output = None

    # ================================================================
    # Phase 1: 一次性排所有 BF16 AG 到 comm_stream（分离式排队）
    # ================================================================
    with torch.cuda.stream(comm_stream):
        for c in range(n_chunks_gateup):
            dist.all_gather_into_tensor(
                ag_outputs[c], ag_inputs[c], group=tp_group,
            )
            ag_evs[c].record(comm_stream)

    # ================================================================
    # Phase 1.5: Shared Expert 排到 shared_stream
    # ================================================================
    if shared_mlp is not None:
        with torch.cuda.stream(shared_stream):
            shared_stream.wait_event(ag_evs[n_chunks_gateup - 1])
            ag_bf16_full = torch.cat(ag_outputs, dim=1)[:M_moe]
            shared_output = shared_mlp(ag_bf16_full)
            shared_ev.record(shared_stream)

    # ================================================================
    # Phase 2: 一次性排所有 GateUp + SiLU + Down 到 compute_stream
    # ================================================================
    with torch.cuda.stream(compute_stream):
        # ---- GateUp GEMM (K-split) ----
        for c in range(n_chunks_gateup):
            compute_stream.wait_event(ag_evs[c])
            K_offset = c * K_chunk
            ag_moe = ag_outputs[c][:M_moe]

            q_chunk, a_chunk_scale = moe_kernel_quantize_input(
                ag_moe, a1_scale, quant_dtype, per_channel_quant, block_shape,
            )

            invoke_moe_kernel_k_split(
                A=q_chunk, B=w1, C=C_gateup,
                A_scale=a_chunk_scale, B_scale=w1_scale,
                topk_weights=None,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=False, top_k=top_k, config=config,
                compute_type=compute_type,
                use_fp8_w8a8=use_fp8_w8a8, use_int8_w8a8=use_int8_w8a8,
                per_channel_quant=per_channel_quant,
                K_offset=K_offset, K_chunk=K_chunk,
                is_first_chunk=(c == 0), block_shape=block_shape, A_K_offset=0,
            )

        # ---- SiLU ----
        torch.ops._C.silu_and_mul(
            intermediate_cache, C_gateup.view(-1, N_gateup),
        )

        # ---- 量化 Down GEMM 输入 ----
        q_intermediate, a2q_scale = moe_kernel_quantize_input(
            intermediate_cache, a2_scale, quant_dtype, per_channel_quant, block_shape,
        )

        # ---- Down GEMM (N-split) + shared add ----
        for c in range(n_chunks_down):
            N_offset = c * N_down_chunk
            invoke_moe_kernel_n_split(
                A=q_intermediate, B=w2, C=C_down,
                A_scale=a2q_scale, B_scale=w2_scale, B_zp=w2_zp,
                topk_weights=topk_weights_moe,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=True, top_k=1, config=config,
                compute_type=compute_type,
                use_fp8_w8a8=use_fp8_w8a8, use_int8_w8a8=use_int8_w8a8,
                use_int8_w8a16=use_int8_w8a16, use_int4_w4a16=use_int4_w4a16,
                per_channel_quant=per_channel_quant,
                N_offset=N_offset, N_chunk=N_down_chunk, block_shape=block_shape,
            )

            C_down_3d = C_down.view(M_moe, top_k, K_out)
            if top_k == 1:
                rs_inputs[c][:M_moe].copy_(
                    C_down_3d[:, 0, N_offset:N_offset + N_down_chunk]
                )
            else:
                torch.sum(
                    C_down_3d[:, :, N_offset:N_offset + N_down_chunk],
                    dim=1, out=rs_inputs[c][:M_moe],
                )

            if shared_output is not None:
                if c == 0:
                    compute_stream.wait_event(shared_ev)
                rs_inputs[c][:M_moe].add_(
                    shared_output[:, N_offset:N_offset + N_down_chunk]
                )
            comp_evs[c].record(compute_stream)

    # ================================================================
    # Phase 3: 一次性排所有 RS 到 comm_stream
    # ================================================================
    with torch.cuda.stream(comm_stream):
        for c in range(n_chunks_down):
            comm_stream.wait_event(comp_evs[c])
            dist.reduce_scatter_tensor(
                rs_outputs[c], rs_inputs[c], group=tp_group,
            )

    # ================================================================
    # Phase 4: 拼接 + 出口同步
    # ================================================================
    compute_stream.wait_stream(comm_stream)
    with torch.cuda.stream(compute_stream):
        output = torch.cat(rs_outputs, dim=1)

    caller_stream.wait_stream(compute_stream)
    caller_stream.wait_stream(comm_stream)
    caller_stream.wait_stream(shared_stream)

    torch.cuda.nvtx.range_pop()
    return output


def moe_forward_overlap_fp8ag(
    state: MoEOverlapState,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    global_num_experts: int,
    expert_map: Optional[torch.Tensor] = None,
    shared_mlp: Optional[torch.nn.Module] = None,
    M_valid: Optional[int] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
) -> torch.Tensor:
    """
    FP8 AG 版 MoE overlap forward。

    AG 前将 hidden_states 从 BF16 量化为 FP8 (block-wise)，通信量减半。
    使用分离式排队模式减少 stream context switch。
    C_gateup 输出为 bf16（与 baseline 一致）。
    Shared Expert 直接接收 FP8 AG 后的全量 tensor 和对应 block scale。
    """
    torch.cuda.nvtx.range_push("MOE_OVERLAP_FP8AG_FORWARD")

    device = hidden_states.device
    state.ensure_streams(device)
    compute_stream = state._compute_stream
    comm_stream = state._comm_stream
    shared_stream = state._shared_stream
    tp_size = state.tp_size
    tp_group = state.tp_group

    caller_stream = torch.cuda.current_stream()

    M_local, K = hidden_states.shape       # [M/tp, K]
    M = M_local * tp_size
    E, N_gateup, _ = w1.shape
    K_out = w2.shape[1]

    M_moe = M_valid if (M_valid is not None and M_valid < M) else M

    n_chunks_gateup = state.n_chunks_gateup
    n_chunks_down = state.n_chunks_down
    K_chunk = K // n_chunks_gateup
    N_down_chunk = K_out // n_chunks_down

    quant_dtype = torch.float8_e4m3fn
    compute_type = tl.bfloat16

    # ================================================================
    # Step 0: Triton config
    # ================================================================
    config_dtype = _get_config_dtype_str(
        use_fp8_w8a8=True, use_int8_w8a16=False,
        use_int4_w4a16=False, ocp_mx_scheme=None,
        dtype=hidden_states.dtype,
    )
    config = try_get_optimal_moe_config(
        w1.shape, w2.shape, top_k, config_dtype, M_moe, block_shape=block_shape,
    )
    BLOCK_SIZE_M = config["BLOCK_SIZE_M"]

    # ================================================================
    # Step 1: Routing
    # ================================================================
    topk_ids_moe = topk_ids[:M_moe]
    topk_weights_moe = topk_weights[:M_moe]
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids_moe, BLOCK_SIZE_M, global_num_experts, expert_map
    )

    # ================================================================
    # Step 2: BF16→FP8 量化（在 caller stream 上，AG 之前）
    # ================================================================
    h_fp8, h_scale = moe_kernel_quantize_input(
        hidden_states, None, quant_dtype, False, block_shape,
    )
    # h_fp8: [M_local, K], h_scale: [M_local, n_scale_groups_K]

    group_k = block_shape[1]
    n_scale_groups_per_K = (K + group_k - 1) // group_k
    n_scale_groups_per_chunk = (K_chunk + group_k - 1) // group_k

    # ================================================================
    # Step 3: 切 K 维 chunks + 分配 buffer
    # ================================================================
    ag_fp8_inputs = [
        h_fp8[:, c * K_chunk:(c + 1) * K_chunk].contiguous()
        for c in range(n_chunks_gateup)
    ]
    ag_fp8_outputs = [
        torch.empty(M, K_chunk, device=device, dtype=quant_dtype)
        for _ in range(n_chunks_gateup)
    ]
    ag_scale_inputs = [
        h_scale[:, c * n_scale_groups_per_chunk:(c + 1) * n_scale_groups_per_chunk].contiguous()
        for c in range(n_chunks_gateup)
    ]
    ag_scale_outputs = [
        torch.empty(M, n_scale_groups_per_chunk, device=device, dtype=torch.float32)
        for _ in range(n_chunks_gateup)
    ]

    # shared expert 拼接用 buffer
    ag_full_fp8 = torch.empty(M_moe, K, device=device, dtype=quant_dtype)
    ag_full_scale = torch.empty(M_moe, n_scale_groups_per_K, device=device, dtype=torch.float32)

    C_gateup = torch.empty(
        M_moe, top_k, N_gateup, device=device, dtype=torch.bfloat16,
    )
    intermediate_cache = torch.empty(
        M_moe * top_k, N_gateup // 2, device=device, dtype=torch.bfloat16,
    )
    C_down = torch.empty(
        M_moe * top_k, 1, K_out, device=device, dtype=torch.bfloat16,
    )

    rs_inputs = [
        torch.zeros(M, N_down_chunk, device=device, dtype=torch.bfloat16)
        for _ in range(n_chunks_down)
    ]
    rs_outputs = [
        torch.empty(M_local, N_down_chunk, device=device, dtype=torch.bfloat16)
        for _ in range(n_chunks_down)
    ]

    # Events
    entry_ev = caller_stream.record_event()
    ag_evs = [torch.cuda.Event(enable_timing=False) for _ in range(n_chunks_gateup)]
    shared_ev = torch.cuda.Event(enable_timing=False)
    comp_evs = [torch.cuda.Event(enable_timing=False) for _ in range(n_chunks_down)]

    comm_stream.wait_event(entry_ev)
    compute_stream.wait_event(entry_ev)
    shared_stream.wait_event(entry_ev)

    shared_output = None

    # ================================================================
    # Phase 1: 一次性排所有 FP8 AG 到 comm_stream（分离式排队）
    # ================================================================
    with torch.cuda.stream(comm_stream):
        for c in range(n_chunks_gateup):
            # FP8 tensor 通过 .view(torch.uint8) 传输，避免 NCCL sm90 检查
            dist.all_gather_into_tensor(
                ag_fp8_outputs[c].view(torch.uint8),
                ag_fp8_inputs[c].view(torch.uint8),
                group=tp_group,
            )
            dist.all_gather_into_tensor(
                ag_scale_outputs[c], ag_scale_inputs[c], group=tp_group,
            )
            ag_evs[c].record(comm_stream)

    # ================================================================
    # Phase 1.5: Shared Expert 排到 shared_stream
    #
    # 等所有 AG 完成后，将 FP8 chunks 和对应 scale 拼成完整 tensor，
    # shared_mlp 直接使用 FP8 input + block scale 做第一层 GEMM。
    # ================================================================
    if shared_mlp is not None:
        with torch.cuda.stream(shared_stream):
            shared_stream.wait_event(ag_evs[n_chunks_gateup - 1])
            for i in range(n_chunks_gateup):
                ag_full_fp8[:, i * K_chunk:(i + 1) * K_chunk].copy_(
                    ag_fp8_outputs[i][:M_moe])
                sc_start = i * n_scale_groups_per_chunk
                sc_end = sc_start + n_scale_groups_per_chunk
                ag_full_scale[:, sc_start:sc_end].copy_(
                    ag_scale_outputs[i][:M_moe])

            shared_output = shared_mlp(ag_full_fp8, input_scale=ag_full_scale)
            shared_ev.record(shared_stream)

    # ================================================================
    # Phase 2: 一次性排所有 GateUp GEMM + SiLU + Down GEMM 到 compute_stream
    # ================================================================
    with torch.cuda.stream(compute_stream):
        # ---- GateUp GEMM (K-split) ----
        for c in range(n_chunks_gateup):
            compute_stream.wait_event(ag_evs[c])
            K_offset = c * K_chunk
            ag_fp8_moe = ag_fp8_outputs[c][:M_moe]
            ag_sc_moe = ag_scale_outputs[c][:M_moe]

            invoke_moe_kernel_k_split(
                A=ag_fp8_moe, B=w1, C=C_gateup,
                A_scale=ag_sc_moe, B_scale=w1_scale,
                topk_weights=None,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=False, top_k=top_k, config=config,
                compute_type=tl.bfloat16,
                use_fp8_w8a8=True, use_int8_w8a8=False,
                per_channel_quant=False,
                K_offset=K_offset, K_chunk=K_chunk,
                is_first_chunk=(c == 0), block_shape=block_shape, A_K_offset=0,
            )

        # ---- SiLU ----
        torch.ops._C.silu_and_mul(
            intermediate_cache, C_gateup.view(-1, N_gateup),
        )

        # ---- 量化 Down GEMM 输入 ----
        q_inter, a2_sc = moe_kernel_quantize_input(
            intermediate_cache, a2_scale, quant_dtype, False, block_shape,
        )

        # ---- Down GEMM (N-split) + shared add ----
        for c in range(n_chunks_down):
            N_offset = c * N_down_chunk
            invoke_moe_kernel_n_split(
                A=q_inter, B=w2, C=C_down,
                A_scale=a2_sc, B_scale=w2_scale, B_zp=w2_zp,
                topk_weights=topk_weights_moe,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=True, top_k=1, config=config,
                compute_type=compute_type,
                use_fp8_w8a8=True, use_int8_w8a8=False,
                use_int8_w8a16=False, use_int4_w4a16=False,
                per_channel_quant=False,
                N_offset=N_offset, N_chunk=N_down_chunk, block_shape=block_shape,
            )

            C_down_3d = C_down.view(M_moe, top_k, K_out)
            if top_k == 1:
                rs_inputs[c][:M_moe].copy_(
                    C_down_3d[:, 0, N_offset:N_offset + N_down_chunk]
                )
            else:
                torch.sum(
                    C_down_3d[:, :, N_offset:N_offset + N_down_chunk],
                    dim=1, out=rs_inputs[c][:M_moe],
                )

            if shared_output is not None:
                if c == 0:
                    compute_stream.wait_event(shared_ev)
                rs_inputs[c][:M_moe].add_(
                    shared_output[:, N_offset:N_offset + N_down_chunk]
                )
            comp_evs[c].record(compute_stream)

    # ================================================================
    # Phase 3: 一次性排所有 RS 到 comm_stream
    # ================================================================
    with torch.cuda.stream(comm_stream):
        for c in range(n_chunks_down):
            comm_stream.wait_event(comp_evs[c])
            dist.reduce_scatter_tensor(
                rs_outputs[c], rs_inputs[c], group=tp_group,
            )

    # ================================================================
    # Phase 4: 拼接 + 出口同步
    # ================================================================
    compute_stream.wait_stream(comm_stream)
    with torch.cuda.stream(compute_stream):
        output = torch.cat(rs_outputs, dim=1)

    caller_stream.wait_stream(compute_stream)
    caller_stream.wait_stream(comm_stream)
    caller_stream.wait_stream(shared_stream)

    torch.cuda.nvtx.range_pop()
    return output


def moe_forward_overlap_fp8ag_fp8rs(
    state: MoEOverlapState,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    global_num_experts: int,
    expert_map: Optional[torch.Tensor] = None,
    shared_mlp: Optional[torch.nn.Module] = None,
    M_valid: Optional[int] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
) -> torch.Tensor:
    """
    FP8 AG + FP8 RS 版 MoE overlap forward。

    在 C 组 (moe_forward_overlap_fp8ag) 的基础上，把 ReduceScatter 也改为 FP8 传输：
      - AG 前将 hidden_states 从 BF16 量化为 FP8 (block-wise)，通信量减半
      - Down GEMM 后的 ReduceScatter 也量化为 FP8+scale 传输，通信量再减半
      - 接收端反量化后本地求和，得到与 BF16 RS 等价的结果

    FP8 RS 实现：
      1. 将 rs_input [M, N_chunk] 沿 dim=0 分成 tp_size 段，每段 [M_local, N_chunk]
      2. 对每段做 block-wise FP8 量化，得到 FP8 data + float32 scale
      3. 用 all-to-all 交换 FP8 data 和 scale（通信量约为 BF16 RS 的一半）
      4. 接收端将 tp_size 段 FP8 反量化后求和，得到最终 [M_local, N_chunk] BF16 结果
    """
    torch.cuda.nvtx.range_push("MOE_OVERLAP_FP8AG_FP8RS_FORWARD")

    device = hidden_states.device
    state.ensure_streams(device)
    compute_stream = state._compute_stream
    comm_stream = state._comm_stream
    shared_stream = state._shared_stream
    tp_size = state.tp_size
    tp_group = state.tp_group

    caller_stream = torch.cuda.current_stream()

    M_local, K = hidden_states.shape       # [M/tp, K]
    M = M_local * tp_size
    E, N_gateup, _ = w1.shape
    K_out = w2.shape[1]

    M_moe = M_valid if (M_valid is not None and M_valid < M) else M

    n_chunks_gateup = state.n_chunks_gateup
    n_chunks_down = state.n_chunks_down
    K_chunk = K // n_chunks_gateup
    N_down_chunk = K_out // n_chunks_down

    quant_dtype = torch.float8_e4m3fn
    compute_type = tl.bfloat16

    # ================================================================
    # Step 0: Triton config
    # ================================================================
    config_dtype = _get_config_dtype_str(
        use_fp8_w8a8=True, use_int8_w8a16=False,
        use_int4_w4a16=False, ocp_mx_scheme=None,
        dtype=hidden_states.dtype,
    )
    config = try_get_optimal_moe_config(
        w1.shape, w2.shape, top_k, config_dtype, M_moe, block_shape=block_shape,
    )
    BLOCK_SIZE_M = config["BLOCK_SIZE_M"]

    # ================================================================
    # Step 1: Routing
    # ================================================================
    topk_ids_moe = topk_ids[:M_moe]
    topk_weights_moe = topk_weights[:M_moe]
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids_moe, BLOCK_SIZE_M, global_num_experts, expert_map
    )

    # ================================================================
    # Step 2: BF16→FP8 量化（在 caller stream 上，AG 之前）
    # ================================================================
    h_fp8, h_scale = moe_kernel_quantize_input(
        hidden_states, None, quant_dtype, False, block_shape,
    )
    # h_fp8: [M_local, K], h_scale: [M_local, n_scale_groups_K]

    group_k = block_shape[1]
    n_scale_groups_per_K = (K + group_k - 1) // group_k
    n_scale_groups_per_chunk = (K_chunk + group_k - 1) // group_k

    # ================================================================
    # Step 3: 切 K 维 chunks + 分配 buffer
    # ================================================================
    ag_fp8_inputs = [
        h_fp8[:, c * K_chunk:(c + 1) * K_chunk].contiguous()
        for c in range(n_chunks_gateup)
    ]
    ag_fp8_outputs = [
        torch.empty(M, K_chunk, device=device, dtype=quant_dtype)
        for _ in range(n_chunks_gateup)
    ]
    ag_scale_inputs = [
        h_scale[:, c * n_scale_groups_per_chunk:(c + 1) * n_scale_groups_per_chunk].contiguous()
        for c in range(n_chunks_gateup)
    ]
    ag_scale_outputs = [
        torch.empty(M, n_scale_groups_per_chunk, device=device, dtype=torch.float32)
        for _ in range(n_chunks_gateup)
    ]

    # shared expert 拼接用 buffer
    ag_full_fp8 = torch.empty(M_moe, K, device=device, dtype=quant_dtype)
    ag_full_scale = torch.empty(M_moe, n_scale_groups_per_K, device=device, dtype=torch.float32)

    C_gateup = torch.empty(
        M_moe, top_k, N_gateup, device=device, dtype=torch.bfloat16,
    )
    intermediate_cache = torch.empty(
        M_moe * top_k, N_gateup // 2, device=device, dtype=torch.bfloat16,
    )
    C_down = torch.empty(
        M_moe * top_k, 1, K_out, device=device, dtype=torch.bfloat16,
    )

    rs_inputs = [
        torch.zeros(M, N_down_chunk, device=device, dtype=torch.bfloat16)
        for _ in range(n_chunks_down)
    ]
    rs_outputs = [
        torch.empty(M_local, N_down_chunk, device=device, dtype=torch.bfloat16)
        for _ in range(n_chunks_down)
    ]

    # FP8 RS pre-allocated buffers (per-chunk for pipelining)
    group_n_rs = block_shape[1]
    n_scale_groups_rs = (N_down_chunk + group_n_rs - 1) // group_n_rs
    rs_fp8_send_bufs = [
        torch.empty(M, N_down_chunk, device=device, dtype=quant_dtype)
        for _ in range(n_chunks_down)
    ]
    rs_fp8_recv_bufs = [
        torch.empty(M, N_down_chunk, device=device, dtype=quant_dtype)
        for _ in range(n_chunks_down)
    ]
    rs_scale_send_bufs = [
        torch.empty(M, n_scale_groups_rs, device=device, dtype=torch.float32)
        for _ in range(n_chunks_down)
    ]
    rs_scale_recv_bufs = [
        torch.empty(M, n_scale_groups_rs, device=device, dtype=torch.float32)
        for _ in range(n_chunks_down)
    ]
    rs_comm_evs = [torch.cuda.Event(enable_timing=False) for _ in range(n_chunks_down)]

    # Events
    entry_ev = caller_stream.record_event()
    ag_evs = [torch.cuda.Event(enable_timing=False) for _ in range(n_chunks_gateup)]
    shared_ev = torch.cuda.Event(enable_timing=False)
    comp_evs = [torch.cuda.Event(enable_timing=False) for _ in range(n_chunks_down)]

    comm_stream.wait_event(entry_ev)
    compute_stream.wait_event(entry_ev)
    shared_stream.wait_event(entry_ev)

    shared_output = None

    # ================================================================
    # Phase 1: 一次性排所有 FP8 AG 到 comm_stream（分离式排队）
    # ================================================================
    with torch.cuda.stream(comm_stream):
        for c in range(n_chunks_gateup):
            # FP8 tensor 通过 .view(torch.uint8) 传输，避免 NCCL sm90 检查
            dist.all_gather_into_tensor(
                ag_fp8_outputs[c].view(torch.uint8),
                ag_fp8_inputs[c].view(torch.uint8),
                group=tp_group,
            )
            dist.all_gather_into_tensor(
                ag_scale_outputs[c], ag_scale_inputs[c], group=tp_group,
            )
            ag_evs[c].record(comm_stream)

    # ================================================================
    # Phase 1.5: Shared Expert 排到 shared_stream
    # ================================================================
    if shared_mlp is not None:
        with torch.cuda.stream(shared_stream):
            shared_stream.wait_event(ag_evs[n_chunks_gateup - 1])
            for i in range(n_chunks_gateup):
                ag_full_fp8[:, i * K_chunk:(i + 1) * K_chunk].copy_(
                    ag_fp8_outputs[i][:M_moe])
                sc_start = i * n_scale_groups_per_chunk
                sc_end = sc_start + n_scale_groups_per_chunk
                ag_full_scale[:, sc_start:sc_end].copy_(
                    ag_scale_outputs[i][:M_moe])

            shared_output = shared_mlp(ag_full_fp8, input_scale=ag_full_scale)
            shared_ev.record(shared_stream)

    # ================================================================
    # Phase 2: GateUp GEMM + SiLU + Down GEMM + FP8 quantize on compute_stream
    # ================================================================
    with torch.cuda.stream(compute_stream):
        # ---- GateUp GEMM (K-split) ----
        for c in range(n_chunks_gateup):
            compute_stream.wait_event(ag_evs[c])
            K_offset = c * K_chunk
            ag_fp8_moe = ag_fp8_outputs[c][:M_moe]
            ag_sc_moe = ag_scale_outputs[c][:M_moe]

            invoke_moe_kernel_k_split(
                A=ag_fp8_moe, B=w1, C=C_gateup,
                A_scale=ag_sc_moe, B_scale=w1_scale,
                topk_weights=None,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=False, top_k=top_k, config=config,
                compute_type=tl.bfloat16,
                use_fp8_w8a8=True, use_int8_w8a8=False,
                per_channel_quant=False,
                K_offset=K_offset, K_chunk=K_chunk,
                is_first_chunk=(c == 0), block_shape=block_shape, A_K_offset=0,
            )

        # ---- SiLU ----
        torch.ops._C.silu_and_mul(
            intermediate_cache, C_gateup.view(-1, N_gateup),
        )

        # ---- 量化 Down GEMM 输入 ----
        q_inter, a2_sc = moe_kernel_quantize_input(
            intermediate_cache, a2_scale, quant_dtype, False, block_shape,
        )

        # ---- Down GEMM (N-split) + shared add + FP8 quantize for RS ----
        for c in range(n_chunks_down):
            N_offset = c * N_down_chunk
            invoke_moe_kernel_n_split(
                A=q_inter, B=w2, C=C_down,
                A_scale=a2_sc, B_scale=w2_scale, B_zp=w2_zp,
                topk_weights=topk_weights_moe,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=True, top_k=1, config=config,
                compute_type=compute_type,
                use_fp8_w8a8=True, use_int8_w8a8=False,
                use_int8_w8a16=False, use_int4_w4a16=False,
                per_channel_quant=False,
                N_offset=N_offset, N_chunk=N_down_chunk, block_shape=block_shape,
            )

            C_down_3d = C_down.view(M_moe, top_k, K_out)
            if top_k == 1:
                rs_inputs[c][:M_moe].copy_(
                    C_down_3d[:, 0, N_offset:N_offset + N_down_chunk]
                )
            else:
                torch.sum(
                    C_down_3d[:, :, N_offset:N_offset + N_down_chunk],
                    dim=1, out=rs_inputs[c][:M_moe],
                )

            if shared_output is not None:
                if c == 0:
                    compute_stream.wait_event(shared_ev)
                rs_inputs[c][:M_moe].add_(
                    shared_output[:, N_offset:N_offset + N_down_chunk]
                )

            # FP8 量化在 compute_stream 上完成，与 comm_stream 并行
            fp8_data, scale_data = moe_kernel_quantize_input(
                rs_inputs[c], None, quant_dtype, False, block_shape,
            )
            rs_fp8_send_bufs[c].copy_(fp8_data)
            rs_scale_send_bufs[c][:, :scale_data.shape[1]].copy_(scale_data)
            comp_evs[c].record(compute_stream)

    # ================================================================
    # Phase 3: FP8 RS 通信 — all-to-all FP8 data + scale (on comm_stream)
    #
    # 通信量约为 BF16 RS 的一半：
    #   BF16 RS: M * N_chunk * 2 bytes per chunk
    #   FP8 RS:  M * N_chunk * 1 byte + M * (N_chunk/128) * 4 bytes
    # ================================================================
    with torch.cuda.stream(comm_stream):
        for c in range(n_chunks_down):
            comm_stream.wait_event(comp_evs[c])
            dist.all_to_all_single(
                rs_fp8_recv_bufs[c].view(torch.uint8),
                rs_fp8_send_bufs[c].view(torch.uint8),
                group=tp_group,
            )
            dist.all_to_all_single(
                rs_scale_recv_bufs[c],
                rs_scale_send_bufs[c],
                group=tp_group,
            )
            rs_comm_evs[c].record(comm_stream)

    # ================================================================
    # Phase 4: 反量化求和 + 拼接（在 compute_stream）
    # ================================================================
    with torch.cuda.stream(compute_stream):
        for c in range(n_chunks_down):
            compute_stream.wait_event(rs_comm_evs[c])
            # Vectorized dequant + sum
            fp8_3d = rs_fp8_recv_bufs[c].view(tp_size, M_local, N_down_chunk)
            scale_3d = rs_scale_recv_bufs[c][:, :n_scale_groups_rs].view(
                tp_size, M_local, n_scale_groups_rs
            )
            fp8_f32 = fp8_3d.to(torch.float32)
            scale_expanded = scale_3d.repeat_interleave(group_n_rs, dim=2)[:, :, :N_down_chunk]
            dequantized_sum = (fp8_f32 * scale_expanded).sum(dim=0)
            rs_outputs[c].copy_(dequantized_sum.to(torch.bfloat16))

        output = torch.cat(rs_outputs, dim=1)

    caller_stream.wait_stream(compute_stream)
    caller_stream.wait_stream(comm_stream)
    caller_stream.wait_stream(shared_stream)

    torch.cuda.nvtx.range_pop()
    return output
