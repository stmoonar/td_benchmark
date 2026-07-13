from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a benchmark config violates the schema contract."""


_PRESETS: dict[str, dict[str, int]] = {
    "v1": {
        "M": 5120,
        "K": 4096,
        "E": 64,
        "top_k": 8,
        "n_gateup": 6144,
        "n_down": 3072,
        "shared_experts": 1,
    },
    "und": {
        "M": 6647,
        "K": 2048,
        "E": 128,
        "top_k": 8,
        "n_gateup": 1536,
        "n_down": 768,
        "shared_experts": 1,
    },
    "gen": {
        "M": 4096,
        "K": 3200,
        "E": 128,
        "top_k": 16,
        "n_gateup": 1536,
        "n_down": 768,
        "shared_experts": 1,
    },
}

_SCHEME_CODES = {"a1", "a2", "b1", "b2", "b3", "c1", "c2", "c3", "c4", "c5"}


@dataclass(frozen=True)
class ShapeCfg:
    M: int
    K: int
    E: int
    top_k: int
    n_gateup: int
    n_down: int
    shared_experts: int = 1
    shared_intermediate: int | None = None

    def __post_init__(self) -> None:
        shared = self.n_down if self.shared_intermediate is None else self.shared_intermediate
        object.__setattr__(self, "shared_intermediate", shared)
        _require("shape.M", self.M, self.M > 0, "> 0")
        _require("shape.K", self.K, self.K > 0, "> 0")
        _require("shape.E", self.E, self.E > 0, "> 0")
        _require("shape.top_k", self.top_k, 0 < self.top_k <= self.E, "0 < top_k <= E")
        _require("shape.n_gateup", self.n_gateup, self.n_gateup > 0 and self.n_gateup % 2 == 0, "> 0 and even")
        _require("shape.n_down", self.n_down, self.n_down == self.n_gateup // 2, "must equal n_gateup // 2")
        _require("shape.shared_experts", self.shared_experts, self.shared_experts in (0, 1), "must be 0 or 1")
        _require("shape.shared_intermediate", shared, shared is not None and shared > 0, "> 0")

    @property
    def intermediate(self) -> int:
        return self.n_down

    def gateup_per_tp(self, world_size: int) -> int:
        _require("shape.n_gateup", self.n_gateup, self.n_gateup % world_size == 0, f"must be divisible by world_size={world_size}")
        return self.n_gateup // world_size

    def intermediate_per_tp(self, world_size: int) -> int:
        _require("shape.n_down", self.n_down, self.n_down % world_size == 0, f"must be divisible by world_size={world_size}")
        return self.n_down // world_size

    def local_E(self, world_size: int) -> int:
        _require("shape.E", self.E, self.E % world_size == 0, f"must be divisible by world_size={world_size}")
        return self.E // world_size

    def M_aligned(self, world_size: int) -> int:
        return ((self.M + world_size - 1) // world_size) * world_size


@dataclass(frozen=True)
class RoutingCfg:
    gate_seed: int = 42
    imbalance_kind: str = "none"
    zipf_s: float = 1.2
    hot_experts: int = 4
    hot_share: float = 0.6
    strength: float = 1.0

    def __post_init__(self) -> None:
        _require("routing.imbalance_kind", self.imbalance_kind, self.imbalance_kind in {"none", "zipf", "hotspot"}, "one of none|zipf|hotspot")
        _require("routing.zipf_s", self.zipf_s, self.zipf_s > 0, "> 0")
        _require("routing.hot_experts", self.hot_experts, self.hot_experts > 0, "> 0")
        _require("routing.hot_share", self.hot_share, 0 < self.hot_share <= 1, "0 < hot_share <= 1")
        _require("routing.strength", self.strength, self.strength >= 0, ">= 0")


@dataclass(frozen=True)
class SchemeCfg:
    code: str
    tunables: dict[str, Any]

    def __post_init__(self) -> None:
        _require("schemes.enabled", self.code, self.code in _SCHEME_CODES, f"scheme code must be one of {sorted(_SCHEME_CODES)}")


@dataclass(frozen=True)
class DistCfg:
    nproc: int = 1
    master_port: int = 29500
    cuda_visible_devices: str = "0"

    def __post_init__(self) -> None:
        _require("dist.nproc", self.nproc, self.nproc > 0, "> 0")
        _require("dist.master_port", self.master_port, 0 < self.master_port < 65536, "1..65535")
        devices_text = str(self.cuda_visible_devices)
        object.__setattr__(self, "cuda_visible_devices", devices_text)
        devices = [item.strip() for item in devices_text.split(",") if item.strip()]
        _require(
            "dist.cuda_visible_devices",
            devices_text,
            len(devices) == self.nproc,
            f"must list exactly nproc={self.nproc} devices",
        )


@dataclass(frozen=True)
class VerifyCfg:
    enabled: bool = True
    tolerances: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.tolerances is None:
            object.__setattr__(self, "tolerances", {})


@dataclass(frozen=True)
class ProfileCfg:
    torch_profiler: bool = False
    intra_kernel: bool = False
    nvtx: bool = True


@dataclass(frozen=True)
class LogCfg:
    level: str = "info"
    diagnostics: bool = False

    def __post_init__(self) -> None:
        _require("log.level", self.level, self.level in {"error", "warn", "warning", "info", "debug"}, "one of error|warn|info|debug")


@dataclass(frozen=True)
class SweepAxis:
    path: str
    values: list[Any]

    def __post_init__(self) -> None:
        _require("sweep_axes.path", self.path, bool(self.path), "non-empty dotted path")
        _require("sweep_axes.values", self.values, bool(self.values), "non-empty list")


@dataclass(frozen=True)
class RunCfg:
    tag: str
    seed: int
    warmup: int
    repeat: int
    output_root: str
    shape: ShapeCfg
    routing: RoutingCfg
    schemes: list[SchemeCfg]
    dist: DistCfg
    env: dict[str, str]
    verify: VerifyCfg
    profile: ProfileCfg
    log: LogCfg
    sweep_axes: list[SweepAxis]

    def __post_init__(self) -> None:
        _require("run.tag", self.tag, bool(self.tag), "non-empty string")
        _require("run.warmup", self.warmup, self.warmup >= 0, ">= 0")
        _require("run.repeat", self.repeat, self.repeat > 0, "> 0")
        _require("schemes.enabled", [s.code for s in self.schemes], bool(self.schemes), "at least one scheme")
        _require(
            "env.CUDA_VISIBLE_DEVICES",
            self.env.get("CUDA_VISIBLE_DEVICES"),
            "CUDA_VISIBLE_DEVICES" not in self.env,
            "must be configured through dist.cuda_visible_devices",
        )
        self.shape.local_E(self.dist.nproc)
        self.shape.gateup_per_tp(self.dist.nproc)
        self.shape.intermediate_per_tp(self.dist.nproc)
        if self.shape.shared_experts:
            _require(
                "shape.shared_intermediate",
                self.shape.shared_intermediate,
                self.shape.shared_intermediate % self.dist.nproc == 0,
                f"must be divisible by world_size={self.dist.nproc}",
            )

    @property
    def verify_enabled(self) -> bool:
        return self.verify.enabled

    def resolved_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["schemes"] = {"enabled": [scheme.code for scheme in self.schemes]}
        for scheme in self.schemes:
            data["schemes"][scheme.code] = {"tunables": scheme.tunables}
        return data


@dataclass(frozen=True)
class SweepPoint:
    index: int
    cfg: RunCfg
    values: dict[str, Any]


def load_config(yaml_path: str | Path, set_overrides: list[str] | None = None) -> RunCfg:
    path = Path(yaml_path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    for override in set_overrides or []:
        _apply_override(raw, override)
    return _build_run_cfg(raw)


def run_cfg_from_dict(raw: dict[str, Any]) -> RunCfg:
    return _build_run_cfg(raw)


def expand_sweep(cfg: RunCfg) -> list[SweepPoint]:
    if not cfg.sweep_axes:
        return [SweepPoint(index=0, cfg=cfg, values={})]
    points: list[SweepPoint] = []
    value_lists = [axis.values for axis in cfg.sweep_axes]
    for index, combo in enumerate(product(*value_lists)):
        raw = cfg.resolved_dict()
        values: dict[str, Any] = {}
        for axis, value in zip(cfg.sweep_axes, combo, strict=True):
            _set_path(raw, axis.path, value)
            values[axis.path] = value
        points.append(SweepPoint(index=index, cfg=_build_run_cfg(raw), values=values))
    return points


def _build_run_cfg(raw: dict[str, Any]) -> RunCfg:
    run = raw.get("run", {})
    shape = _shape_from_raw(raw.get("shape", {}))
    routing = _routing_from_raw(raw.get("routing", {}))
    schemes = _schemes_from_raw(raw.get("schemes", {}))
    cfg = RunCfg(
        tag=str(run.get("tag", "default")),
        seed=int(run.get("seed", 42)),
        warmup=int(run.get("warmup", 20)),
        repeat=int(run.get("repeat", 50)),
        output_root=str(run.get("output_root", "results")),
        shape=shape,
        routing=routing,
        schemes=schemes,
        dist=DistCfg(**(raw.get("dist", {}) or {})),
        env={str(k): str(v) for k, v in (raw.get("env", {}) or {}).items()},
        verify=VerifyCfg(**(raw.get("verify", {}) or {})),
        profile=ProfileCfg(**(raw.get("profile", {}) or {})),
        log=LogCfg(**(raw.get("log", {}) or {})),
        sweep_axes=[SweepAxis(path=str(item["path"]), values=list(item["values"])) for item in raw.get("sweep_axes", []) or []],
    )
    return cfg


def _shape_from_raw(raw: dict[str, Any]) -> ShapeCfg:
    data: dict[str, Any] = {}
    preset = raw.get("preset")
    if preset:
        if preset not in _PRESETS:
            raise ConfigError(f"shape.preset={preset!r} violates constraint: must be one of {sorted(_PRESETS)}")
        data.update(_PRESETS[preset])
    data.update({k: v for k, v in raw.items() if k != "preset"})
    required = ["M", "K", "E", "top_k", "n_gateup", "n_down"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ConfigError(f"shape missing required fields: {missing}")
    return ShapeCfg(
        M=int(data["M"]),
        K=int(data["K"]),
        E=int(data["E"]),
        top_k=int(data["top_k"]),
        n_gateup=int(data["n_gateup"]),
        n_down=int(data["n_down"]),
        shared_experts=int(data.get("shared_experts", 1)),
        shared_intermediate=None if data.get("shared_intermediate") is None else int(data["shared_intermediate"]),
    )


def _routing_from_raw(raw: dict[str, Any]) -> RoutingCfg:
    data = dict(raw)
    imbalance = data.pop("imbalance", None)
    if isinstance(imbalance, dict):
        data["imbalance_kind"] = imbalance.get("kind", data.get("imbalance_kind", "none"))
        for key, value in imbalance.items():
            if key != "kind":
                data[key] = value
    return RoutingCfg(**data)


def _schemes_from_raw(raw: dict[str, Any]) -> list[SchemeCfg]:
    enabled = raw.get("enabled", [])
    schemes: list[SchemeCfg] = []
    for code in enabled:
        scheme_raw = raw.get(code, {}) or {}
        schemes.append(SchemeCfg(code=str(code), tunables=dict(scheme_raw.get("tunables", {}) or {})))
    return schemes


def _apply_override(raw: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ConfigError(f"--set {override!r} violates constraint: must be path=value")
    path, value_text = override.split("=", 1)
    value = yaml.safe_load(value_text)
    _set_path(raw, path, value)


def _set_path(raw: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor = raw
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if next_value is None:
            next_value = {}
            cursor[part] = next_value
        if not isinstance(next_value, dict):
            raise ConfigError(f"{path} violates constraint: parent {part!r} is not a mapping")
        cursor = next_value
    cursor[parts[-1]] = value


def _require(field: str, value: Any, ok: bool, constraint: str) -> None:
    if not ok:
        raise ConfigError(f"{field}={value!r} violates constraint: {constraint}")


def replace_cfg(cfg: RunCfg, **changes: Any) -> RunCfg:
    return replace(cfg, **changes)
