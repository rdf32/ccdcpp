"""The preprocessing steps of a configuration, by name and in physical terms.

A configuration's ``preprocess`` section says, per procedure (``standard``, ``snow``,
``insufficient_clear``), whether it reads the thermal row through the input's thermal transform
and which steps build its processing mask, in order. A step's parameters are physical: QA codes
by name, bands by role, bounds in reflectance and degrees Celsius. The run resolves them against
its input with the rest of the configuration; ``ccdcpp.plan_of(config, input)`` shows the
result. The classes here write a chain; a field left None is the step's default
(``ccdcpp.steps()`` lists them)::

    from ccdcpp import BandRange, Dedupe, KeepQa, ProcedureChain, ThermalRange

    keeps_snow = ccdcpp.config("ccdc").replace(preprocess__standard=ProcedureChain(
        steps=(KeepQa(("clear", "water", "snow")), ThermalRange(), BandRange(), Dedupe()),
        thermal_transform=True,
    ))
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

__all__ = ["AddQa", "BandRange", "Dedupe", "GreenMedian", "KeepQa", "PROCEDURES", "Preprocess",
           "ProcedureChain", "QA_CODE_NAMES", "REFLECTIVE", "ThermalRange"]

#: The procedures of a configuration's preprocess section.
PROCEDURES = ("standard", "snow", "insufficient_clear")

#: The band list of every row whose role is not thermal or other, in row order.
REFLECTIVE = "reflective"

#: The QA code names a step takes; each is the ``ccdcpp.Input`` field ``qa_<name>``.
QA_CODE_NAMES = ("fill", "clear", "water", "shadow", "snow", "cloud", "cirrus1", "cirrus2",
                 "occlusion")


def _codes(value: str | int | Iterable[str | int], where: str) -> tuple[str | int, ...]:
    """``value`` as a tuple of QA code names, or of ints 0..255."""
    codes = (value,) if isinstance(value, (str, int)) else tuple(value)
    names = [c for c in codes if isinstance(c, str)]
    if names and len(names) != len(codes):
        raise ValueError(f"{where}: codes are all names or all ints, not {codes!r}")
    for code in codes:
        if isinstance(code, str):
            if code not in QA_CODE_NAMES:
                raise ValueError(f"{where}: unknown QA code name {code!r}; the names are "
                                 f"{QA_CODE_NAMES}, or pass int codes")
        elif isinstance(code, bool) or not hasattr(code, "__index__") or not 0 <= code <= 255:
            raise ValueError(f"{where}: QA code {code!r} is not a name or an int in 0..255")
    return tuple(c if isinstance(c, str) else int(c) for c in codes)


def _pair(value: Iterable[bool] | None, name: str) -> tuple[bool, bool] | None:
    if value is None:
        return None
    pair = tuple(value)
    if len(pair) != 2 or not all(isinstance(v, bool) or type(v).__name__ == "bool_"
                                 for v in pair):
        raise ValueError(f"{name} is {value!r}; it must be two bools (min, max)")
    return (bool(pair[0]), bool(pair[1]))


def _given(step: str, **params: Any) -> dict[str, Any]:
    """The step as a configuration holds it: {"step": name, param: value} for every
    parameter given (not None)."""
    out = {"step": step}
    out.update({k: v for k, v in params.items() if v is not None})
    return out


def _range(step: str, min: float | None, max: float | None,
           inclusive: tuple[bool, bool] | None) -> dict[str, Any]:
    flags = {} if inclusive is None else {"min_inclusive": inclusive[0],
                                         "max_inclusive": inclusive[1]}
    return dict(_given(step, min=min, max=max), **flags)


@dataclass(frozen=True)
class KeepQa:
    """keep_qa: keep only the observations whose QA code is in ``codes``, names from
    :data:`QA_CODE_NAMES` (the input's ``qa_*`` codes) or ints. None: clear and water."""

    codes: tuple[str | int, ...] | None = None

    def __post_init__(self) -> None:
        if self.codes is not None:
            object.__setattr__(self, "codes", _codes(self.codes, "KeepQa"))

    def to_dict(self) -> dict[str, Any]:
        return _given("keep_qa", codes=None if self.codes is None else list(self.codes))


@dataclass(frozen=True)
class AddQa:
    """add_qa: also keep every observation whose QA code is in ``codes``, whatever an earlier
    step decided; a :class:`Dedupe` must follow it. None: snow."""

    codes: tuple[str | int, ...] | None = None

    def __post_init__(self) -> None:
        if self.codes is not None:
            object.__setattr__(self, "codes", _codes(self.codes, "AddQa"))

    def to_dict(self) -> dict[str, Any]:
        return _given("add_qa", codes=None if self.codes is None else list(self.codes))


@dataclass(frozen=True)
class ThermalRange:
    """thermal_range: keep only the observations whose thermal row lies in ``(min, max)``,
    degrees Celsius; an inclusive end also keeps a value equal to it. None: pyccd's -93.2 and
    70.7, exclusive. A procedure with ``thermal_transform`` tests the transformed row; one
    without it tests the row as the input holds it against the bounds taken back through the
    input's transform, so the range is in degrees Celsius on every input. An input without a
    ``"thermal"`` row skips the step."""

    min: float | None = None
    max: float | None = None
    inclusive: tuple[bool, bool] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "inclusive", _pair(self.inclusive, "inclusive"))

    def to_dict(self) -> dict[str, Any]:
        return _range("thermal_range", self.min, self.max, self.inclusive)


@dataclass(frozen=True)
class BandRange:
    """band_range: keep only the observations on which every band of ``bands`` lies in
    ``(min, max)``, reflectance; an inclusive end also keeps a value equal to it. ``bands`` is
    roles (or rows), tested in order, or ``"reflective"``: every row whose role is not thermal
    or other, in row order. An empty list is refused. None: ``"reflective"``, 0 and 1,
    exclusive."""

    bands: str | tuple[str | int, ...] | None = None
    min: float | None = None
    max: float | None = None
    inclusive: tuple[bool, bool] | None = None

    def __post_init__(self) -> None:
        if self.bands is not None and self.bands != REFLECTIVE:
            bands = (self.bands,) if isinstance(self.bands, str) else tuple(self.bands)
            if not bands:
                raise ValueError(f"BandRange: bands is empty; name the bands tested, or "
                                 f"{REFLECTIVE!r}")
            object.__setattr__(self, "bands", bands)
        object.__setattr__(self, "inclusive", _pair(self.inclusive, "inclusive"))

    def to_dict(self) -> dict[str, Any]:
        out = _range("band_range", self.min, self.max, self.inclusive)
        if self.bands is not None:
            out["bands"] = self.bands if self.bands == REFLECTIVE else list(self.bands)
        return out


@dataclass(frozen=True)
class GreenMedian:
    """green_median: InsufficientClear's cloud screen. Drop the observations whose green row is
    at or above the median green of the kept observations dated up to the shared
    ``statistics_end``, plus ``offset`` (reflectance), at the dates ``scope`` names: ``"all"``,
    ``"stat_window"`` or ``"off"``. None: 0.04 and ``"stat_window"``."""

    offset: float | None = None
    scope: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _given("green_median", offset=self.offset, scope=self.scope)


@dataclass(frozen=True)
class Dedupe:
    """dedupe: keep the first kept observation of each date."""

    def to_dict(self) -> dict[str, Any]:
        return {"step": "dedupe"}


_STEP_TYPES = (KeepQa, AddQa, ThermalRange, BandRange, GreenMedian, Dedupe)


def step_dict(step: object) -> dict[str, Any]:
    """A step as a configuration holds it: one of the classes here, or a dict
    {"step": name, ...}."""
    if isinstance(step, _STEP_TYPES):
        return step.to_dict()
    if isinstance(step, dict) and "step" in step:
        return dict(step)
    raise TypeError(f"{step!r} is not a step; steps are "
                    f"{', '.join(t.__name__ for t in _STEP_TYPES)} or a dict {{'step': ...}}")


@dataclass(frozen=True)
class ProcedureChain:
    """One procedure's preprocessing: its ``steps`` in order (at most
    ``ccdcpp.CHAIN_CAPACITY``), and whether it reads the thermal row through the input's
    thermal transform (``thermal_transform``; "ccdc" sets it for standard only, as pyccd)."""

    steps: tuple[object, ...] = ()
    thermal_transform: bool = False

    def __post_init__(self) -> None:
        steps = tuple(self.steps)
        for step in steps:
            step_dict(step)
        object.__setattr__(self, "steps", steps)
        if not isinstance(self.thermal_transform, bool):
            raise TypeError(f"ProcedureChain: thermal_transform is {self.thermal_transform!r}; "
                            f"it is True or False")

    def to_dict(self) -> dict[str, Any]:
        return {"thermal_transform": self.thermal_transform,
                "steps": [step_dict(s) for s in self.steps]}


@dataclass(frozen=True)
class Preprocess:
    """A preprocess section: a :class:`ProcedureChain` per procedure; None keeps the base
    configuration's (``Config.replace(preprocess=...)``)."""

    standard: ProcedureChain | None = None
    snow: ProcedureChain | None = None
    insufficient_clear: ProcedureChain | None = None

    def __post_init__(self) -> None:
        for name in PROCEDURES:
            value = getattr(self, name)
            if value is not None and not isinstance(value, ProcedureChain):
                raise TypeError(f"Preprocess: {name} is {value!r}, not a ProcedureChain")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name).to_dict() for name in PROCEDURES
                if getattr(self, name) is not None}
