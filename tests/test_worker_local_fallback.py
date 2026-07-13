from pathlib import Path

from moe_bench.config import load_config
from moe_bench.context import DistContext
from moe_bench.worker import _build_row


def test_rank0_result_row_records_distributed_fairness_contract(tmp_path):
    cfg = load_config("configs/smoke.yaml", ["schemes.enabled=[a1]"])
    scheme_cfg = cfg.schemes[0]
    ctx = DistContext(rank=0, world_size=4, local_rank=0, device="cpu")
    verify = {
        "pass": True,
        "vs": "golden_fp8sim_group128",
        "max_abs": 0.0,
        "max_rel": 0.0,
        "cos_sim": 1.0,
        "rel_p99": 0.0,
        "rel_masked_max": 0.0,
        "scale": 1.0,
    }

    row = _build_row(
        Path(tmp_path),
        {"index": 0, "values": {}},
        cfg,
        scheme_cfg,
        {"avg": 1.0, "min": 1.0, "med": 1.0, "p95": 1.0, "std": 0.0},
        [1.0] * cfg.repeat,
        verify,
        1.0,
        ctx,
    )

    assert row["rank"] == 0
    assert row["world_size"] == 4
    assert row["latency_aggregation"] == "per_iteration_max_across_ranks"
    assert row["routing"] == {
        "kind": "none",
        "included_in_timing": False,
        "contract": "gate_and_topk_precomputed",
    }
    assert row["quantization"]["activation"]["group_size"] == 128
    assert row["quantization"]["weight"]["block_shape"] == [128, 128]
