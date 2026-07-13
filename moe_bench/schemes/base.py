from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


class SchemeConfigError(ValueError):
    """Raised when scheme tunables do not match the scheme schema."""


@dataclass(frozen=True)
class TunableSpec:
    typ: type
    default: Any
    description: str


@dataclass(frozen=True)
class SchemeSpec:
    code: str
    name: str
    family: str
    parallel: str
    comm: str
    weight_dtype: str
    act_quant: str
    requires_nvshmem: bool
    tunables_schema: dict[str, TunableSpec]


@dataclass
class StageResult:
    output: Any
    stage_ms: dict[str, float]


@dataclass
class SchemeInstance:
    run: Callable[[], Any]
    run_staged: Callable[[], StageResult] | None
    diagnostics: dict[str, Any]
    close: Callable[[], None]


BuilderFn = Callable[[Any, Any, Any, Any], SchemeInstance]


def validate_tunables(spec: SchemeSpec, provided: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(provided) - set(spec.tunables_schema))
    if unknown:
        raise SchemeConfigError(f"{spec.code} tunables {unknown} violate schema: allowed keys are {sorted(spec.tunables_schema)}")
    values = {name: item.default for name, item in spec.tunables_schema.items()}
    values.update(provided)
    for name, value in values.items():
        expected = spec.tunables_schema[name].typ
        if expected is bool:
            ok = isinstance(value, bool)
        else:
            ok = isinstance(value, expected) and not isinstance(value, bool)
        if not ok:
            raise SchemeConfigError(f"{spec.code}.{name}={value!r} violates schema: expected {expected.__name__}")
    return values


def make_lazy_instance(
    spec: SchemeSpec,
    tunables: dict[str, Any],
    run_impl: Callable[[], Any],
    diagnostics: dict[str, Any] | None = None,
    close_impl: Callable[[], None] | None = None,
) -> SchemeInstance:
    diag = {
        "scheme": spec.code,
        "name": spec.name,
        "family": spec.family,
        "parallel": spec.parallel,
        "weight_dtype": spec.weight_dtype,
        "act_quant": spec.act_quant,
        "requires_nvshmem": spec.requires_nvshmem,
        "tunables": tunables,
    }
    if diagnostics:
        diag.update(diagnostics)

    def run_staged() -> StageResult:
        return StageResult(output=run_impl(), stage_ms={})

    return SchemeInstance(
        run=run_impl,
        run_staged=run_staged,
        diagnostics=diag,
        close=close_impl or (lambda: None),
    )


def missing_dependency(message: str) -> None:
    raise RuntimeError(message)
