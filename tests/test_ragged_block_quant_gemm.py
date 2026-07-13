import os

import pytest

from moe_bench.data.ragged import plan_ragged_block_quant_gemm_repro

torch = pytest.importorskip("torch")


def test_ragged_repro_scaffold_covers_und_gen_k192_case_on_cpu():
    plan = plan_ragged_block_quant_gemm_repro(M=128, K=2048, n_gateup=1536, world_size=4, block=128)

    assert plan["intermediate_per_rank"] == 192
    assert plan["requires_l1_requant"] is True
    assert plan["l2_padded_intermediate"] == 256
    assert plan["l2_compute_overhead"] == pytest.approx(256 / 192)
    assert plan["kernel_checks"] == [
        "vllm_fused_moe_group128",
        "tdx_fp8_group_gemm",
    ]


@pytest.mark.skipif(
    os.environ.get("MOE_BENCH_RUN_SERVER_REPRO") != "1",
    reason="manual destructive/server repro; excluded from the automated correctness suite",
)
def test_ragged_block_quant_gemm_server_repro():
    pytest.importorskip("vllm")
    pytest.importorskip("triton_dist")
    plan = plan_ragged_block_quant_gemm_repro(M=128, K=2048, n_gateup=1536, world_size=4, block=128)
    assert plan["requires_l1_requant"] is True
    pytest.fail("SERVER-VERIFY: call vLLM and tdx kernels with this plan and record EXP-004 L1/L2 decision")
