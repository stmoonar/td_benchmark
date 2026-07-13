import pytest
from pathlib import Path

from moe_bench.config import ConfigError, expand_sweep, load_config, run_cfg_from_dict

REQUIRED_ENV_KEYS = {
    "NVSHMEM_SYMMETRIC_SIZE",
    "NVSHMEM_REMOTE_TRANSPORT",
    "NVSHMEM_DISABLE_CUDA_VMM",
    "C_INCLUDE_PATH",
    "TRITON_PTXAS_PATH",
}


def test_load_config_applies_defaults_and_overrides():
    cfg = load_config(
        "configs/smoke.yaml",
        [
            "run.tag=ci",
            "shape.M=512",
            "schemes.enabled=[a1,b3]",
            "schemes.b3.tunables.n_chunks_down=8",
        ],
    )

    assert cfg.tag == "ci"
    assert cfg.shape.M == 512
    assert cfg.shape.n_down == cfg.shape.n_gateup // 2
    assert [scheme.code for scheme in cfg.schemes] == ["a1", "b3"]
    assert cfg.schemes[1].tunables["n_chunks_down"] == 8
    assert cfg.shape.shared_intermediate == cfg.shape.n_down


def test_resolved_config_round_trip_preserves_run_settings():
    cfg = load_config(
        "configs/smoke.yaml",
        ["run.tag=roundtrip", "run.warmup=7", "run.repeat=13", "run.output_root=/tmp/focused"],
    )

    restored = run_cfg_from_dict(cfg.resolved_dict())

    assert restored.tag == "roundtrip"
    assert restored.warmup == 7
    assert restored.repeat == 13
    assert restored.output_root == "/tmp/focused"


def test_invalid_shape_reports_field_value_and_constraint():
    with pytest.raises(ConfigError) as excinfo:
        load_config("configs/smoke.yaml", ["shape.n_down=123"])

    message = str(excinfo.value)
    assert "shape.n_down" in message
    assert "123" in message
    assert "n_gateup // 2" in message


def test_invalid_intermediate_tp_shard_reports_n_down_divisibility():
    with pytest.raises(ConfigError) as excinfo:
        load_config(
            "configs/smoke.yaml",
            [
                "dist.nproc=8",
                "shape.E=8",
                "shape.n_gateup=24",
                "shape.n_down=12",
                "shape.shared_intermediate=16",
                "dist.cuda_visible_devices=0,1,2,3,4,5,6,7",
            ],
        )

    message = str(excinfo.value)
    assert "shape.n_down" in message
    assert "12" in message
    assert "world_size=8" in message


def test_invalid_dist_device_count_reports_cuda_visible_devices_constraint():
    with pytest.raises(ConfigError) as excinfo:
        load_config("configs/smoke.yaml", ["dist.nproc=2", "dist.cuda_visible_devices=0"])

    message = str(excinfo.value)
    assert "dist.cuda_visible_devices" in message
    assert "0" in message
    assert "nproc=2" in message


def test_env_cuda_visible_devices_is_rejected_because_dist_owns_it():
    with pytest.raises(ConfigError) as excinfo:
        load_config("configs/smoke.yaml", ["env.CUDA_VISIBLE_DEVICES=4"])

    message = str(excinfo.value)
    assert "env.CUDA_VISIBLE_DEVICES" in message
    assert "dist.cuda_visible_devices" in message


def test_cuda_device_max_connections_is_rejected_to_preserve_runtime_default():
    with pytest.raises(ConfigError) as excinfo:
        load_config("configs/smoke.yaml", ["env.CUDA_DEVICE_MAX_CONNECTIONS=1"])

    assert "must remain unset" in str(excinfo.value)


@pytest.mark.parametrize("scheme", ["c1", "c2"])
def test_bf16_triton_dist_schemes_are_disabled(scheme):
    with pytest.raises(ConfigError):
        load_config("configs/smoke.yaml", [f"schemes.enabled=[{scheme}]"])


def test_expand_sweep_uses_cartesian_product_without_mutating_base():
    cfg = load_config(
        "configs/smoke.yaml",
        [
            "sweep_axes=[{path: shape.M, values: [128, 256]}, {path: routing.imbalance_kind, values: [none, zipf]}]",
        ],
    )

    points = expand_sweep(cfg)

    assert [point.index for point in points] == [0, 1, 2, 3]
    assert [(point.cfg.shape.M, point.cfg.routing.imbalance_kind) for point in points] == [
        (128, "none"),
        (128, "zipf"),
        (256, "none"),
        (256, "zipf"),
    ]
    assert cfg.shape.M == 5120


@pytest.mark.parametrize("path", ["default.yaml", "smoke.yaml", "sweep_v1.yaml", "sweep_und.yaml", "sweep_gen.yaml", "tune_c3.yaml"])
def test_shipped_configs_include_required_server_env_defaults(path):
    cfg = load_config(f"configs/{path}")

    assert REQUIRED_ENV_KEYS <= set(cfg.env)
    assert "CUDA_DEVICE_MAX_CONNECTIONS" not in cfg.env
    assert set(scheme.code for scheme in cfg.schemes).isdisjoint({"c1", "c2"})


def test_default_config_is_annotated_template():
    text = Path("configs/default.yaml").read_text(encoding="utf-8")
    comment_lines = [line for line in text.splitlines() if line.strip().startswith("#")]

    assert len(comment_lines) >= 10
    for phrase in [
        "n_gateup",
        "NVSHMEM_SYMMETRIC_SIZE",
        "schemes.enabled",
        "sweep_axes",
    ]:
        assert phrase in text


@pytest.mark.parametrize("path", ["sweep_v1.yaml", "sweep_und.yaml", "sweep_gen.yaml"])
def test_shape_sweep_configs_cover_m_list_and_three_imbalance_distributions(path):
    cfg = load_config(f"configs/{path}")
    axes = {axis.path: axis.values for axis in cfg.sweep_axes}

    assert axes["shape.M"] == [128, 512, 1024, 2048, 3072, 4096, 5120]
    assert axes["routing.imbalance_kind"] == ["none", "zipf", "hotspot"]
    assert len(expand_sweep(cfg)) == 21
