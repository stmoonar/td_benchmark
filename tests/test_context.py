import os

from moe_bench.config import load_config
from moe_bench.context import DistContext, context_from_env, requires_nvshmem


def test_context_from_env_parses_torchrun_environment(monkeypatch):
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "1")

    ctx = context_from_env(device="cpu")

    assert ctx.rank == 2
    assert ctx.world_size == 4
    assert ctx.local_rank == 1
    assert str(ctx.device) == "cpu"
    assert ctx.group is None
    assert ctx.nvshmem_initialized is False


def test_context_from_env_defaults_to_single_process(monkeypatch):
    for key in ["RANK", "WORLD_SIZE", "LOCAL_RANK"]:
        monkeypatch.delenv(key, raising=False)

    ctx = context_from_env(device="cpu")

    assert ctx == DistContext(rank=0, world_size=1, local_rank=0, device="cpu", group=None, nvshmem_initialized=False)


def test_requires_nvshmem_follows_scheme_registry():
    no_nvshmem = load_config("configs/smoke.yaml", ["schemes.enabled=[a1,b1]"])
    yes_nvshmem = load_config("configs/smoke.yaml", ["schemes.enabled=[c1]"])

    assert requires_nvshmem(no_nvshmem) is False
    assert requires_nvshmem(yes_nvshmem) is True
