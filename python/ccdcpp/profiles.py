"""Sensor profiles: one sensor's band layout, input units and QA, by role, and the
configuration it runs (``SensorProfile.config()``).

A :class:`SensorProfile` names the rows of the ``spectral`` array and says
which role each one plays. Its ``input()`` is the ``ccdcpp.Input`` of the
data; the detection and Tmask lists are role names the configuration holds,
resolved to rows against the input when a run is built, and the saturation
list is resolved when the input is, so the same profile text holds for any
band order. ``ccdcpp.detect``,
``ccdcpp.detect_cube`` and ``ccdcpp.detect_cube_columns`` take one as
``profile=``.

A new sensor is one constructor call; ``ccdcpp.presets`` writes its profiles
that way. NOT SHIPPED: the profile below is an example of the shape, its
values are not validated, and ccdcpp.presets does not contain it::

    from ccdcpp import CELSIUS_HUNDREDTHS, REFLECTANCE, SensorProfile

    def landsat_oli_c2():
        # Landsat 8/9 Collection 2 Level-2 with the coastal band, in the
        # units of presets.landsat_c2: reflectance and degrees Celsius / 100.
        # Raw QA_PIXEL.
        return SensorProfile(
            name="landsat_oli_c2",
            band_names=("SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6",
                        "SR_B7", "ST_B10"),
            roles=("coastal", "blue", "green", "red", "nir", "swir1", "swir2",
                   "thermal"),
            detection=("green", "red", "nir", "swir1", "swir2"),
            tmask=("green", "swir1"),
            saturation=("coastal", "blue", "green", "red", "nir", "swir1",
                        "swir2"),
            units=REFLECTANCE,
            alpha=1e-4,
            coef_floor=1e-4,
            thermal=CELSIUS_HUNDREDTHS,
            stat_ord=736694,  # 2017-12-31, the Landsat statistics cutoff
            qa_table_builder="landsat_c2_qa_pixel",
        )

The detection count sets the chi-squared degrees of freedom of the change and
outlier thresholds: the run computes them from the configuration's
probabilities for the count it resolves.

A profile's ``preprocess=`` is a :class:`Preprocess`: per procedure, the
filter steps that build the processing mask and the transform before them,
written with roles and QA code names. None, the default, is the default plan
of the profile's input and configuration (``ccdcpp.default_plan``). The Standard procedure
keeping snow observations::

    from ccdcpp import BandRange, Dedupe, KeepQa, Preprocess, ProcedureChain
    from ccdcpp import ThermalRange

    keeps_snow = Preprocess(standard=ProcedureChain(steps=(
        KeepQa(("clear", "water", "snow")), ThermalRange(), BandRange(), Dedupe(),
    )))  # and the default transform, the profile's thermal one
    profile = dataclasses.replace(presets.landsat_c1_dn(), preprocess=keeps_snow)
"""
from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping, Optional, Tuple, Union

import numpy as np

from .ccdcpp import (
    CCD_MAX_BANDS,
    NO_STAT_CUTOFF,
    ROLES,
    Affine,
    Chain,
    GreenFilterScope,
    Input,
    PreprocessPlan,
    ProcedurePlan,
    Step,
    default_plan,
    qa_tables,
    validate_plan,
)

#: ``green_filter_scope`` names -> ``ccdcpp.GreenFilterScope`` values.
GREEN_FILTER_SCOPES = {
    "all": GreenFilterScope.All,
    "stat_window": GreenFilterScope.StatWindow,
    "off": GreenFilterScope.Off,
}

#: The Landsat statistics cutoff, 2017-12-31 (the "ccdc" preset's shared stat_ord).
LANDSAT_STAT_ORD = 736694


def _flag_pair(value, name):
    """``value`` as a ``(bool, bool)``, for the inclusive-bound fields."""
    pair = tuple(value)
    if len(pair) != 2 or not all(isinstance(v, (bool, np.bool_)) for v in pair):
        raise ValueError(f"{name} is {value!r}; it must be two bools (lo, hi)")
    return (bool(pair[0]), bool(pair[1]))


@dataclass(frozen=True)
class Units:
    """The units of the reflective bands.

    refl_min, refl_max
        ``Input.refl_min``, ``Input.refl_max``: the saturation bounds, tested
        on the saturation bands.
    median_green_filter
        The offset of InsufficientClear's green median cloud screen, in the
        data's units (the configuration's shared ``green_filter`` times
        ``reflectance_unit``; :meth:`SensorProfile.config` writes it).
    reflectance_unit
        ``Input.reflectance_unit``: the value one unit of reflectance has in
        the data (Collection-1 DN 10000, reflectance 1).
    refl_inclusive
        ``(refl_min_inclusive, refl_max_inclusive)``: whether a value equal
        to each bound is kept (default ``(False, False)``, pyccd's exclusive
        range, which rejects NaN). NaN passes only with both True.
    """

    refl_min: float
    refl_max: float
    median_green_filter: float
    reflectance_unit: float
    refl_inclusive: Tuple[bool, bool] = (False, False)

    def __post_init__(self):
        object.__setattr__(self, "refl_inclusive", _flag_pair(self.refl_inclusive, "refl_inclusive"))


@dataclass(frozen=True)
class Thermal:
    """The thermal band's transform and bounds.

    scale, offset
        ``Input.thermal_scale``, ``Input.thermal_offset``: the Standard
        procedure reads ``thermal * scale + offset``. ``(1.0, 0.0)`` for input
        already in the bounds' unit.
    min, max
        ``Input.therm_min``, ``Input.therm_max``: the valid range, after the
        transform.
    inclusive
        ``(therm_min_inclusive, therm_max_inclusive)``: whether a value equal
        to each bound is kept (default ``(False, False)``, pyccd's exclusive
        range, which rejects NaN).
    """

    scale: float
    offset: float
    min: float
    max: float
    inclusive: Tuple[bool, bool] = (False, False)

    def __post_init__(self):
        object.__setattr__(self, "inclusive", _flag_pair(self.inclusive, "inclusive"))


#: Collection 1 DN: reflectance x 10000 (``Input()`` defaults).
DN = Units(refl_min=0.0, refl_max=10000.0, median_green_filter=400.0, reflectance_unit=10000.0)

#: Surface reflectance, 0..1: :data:`DN` divided by 10000.
REFLECTANCE = Units(refl_min=0.0, refl_max=1.0, median_green_filter=0.04, reflectance_unit=1.0)

#: Thermal as Kelvin x 10, transformed to degrees Celsius x 100
#: (``Input()`` defaults).
KELVIN_TENTHS = Thermal(scale=10.0, offset=-27315.0, min=-9320.0, max=7070.0)

#: Thermal already in degrees Celsius / 100, the unit that shares the
#: reflectance scale factor of 10000 with the optical bands (one Lasso alpha
#: serves every band only if every band has the same scale).
CELSIUS_HUNDREDTHS = Thermal(scale=1.0, offset=0.0, min=-0.9320, max=0.7070)


#------------------------------------------------------------------------------
# Preprocessing steps, by role and QA code name.
#
# Each step resolves to a ccdcpp.Step against a profile, the Input of the run
# and the shared values its configuration resolves to for it
# (SensorProfile.preprocess_plan). A field left None takes the value
# default_plan() would use from those, so a step with every field None is the
# default plan's step.
#------------------------------------------------------------------------------

#: QA code names -> the ``ccdcpp.Input`` fields that hold the codes.
QA_CODE_FIELDS = {
    "fill": "qa_fill",
    "clear": "qa_clear",
    "water": "qa_water",
    "shadow": "qa_shadow",
    "snow": "qa_snow",
    "cloud": "qa_cloud",
}


def _codes(value, where):
    """``value`` as a tuple of QA code names and ints 0..255."""
    codes = (value,) if isinstance(value, (str, int)) else tuple(value)
    for code in codes:
        if isinstance(code, str):
            if code not in QA_CODE_FIELDS:
                raise ValueError(
                    f"{where}: unknown QA code name {code!r}; names are "
                    f"{tuple(QA_CODE_FIELDS)}, or pass an int code")
        elif isinstance(code, (bool, np.bool_)) or not isinstance(code, (int, np.integer)) \
                or not 0 <= code <= 255:
            raise ValueError(f"{where}: QA code {code!r} is not a name or an int in 0..255")
    return tuple(c if isinstance(c, str) else int(c) for c in codes)


def _resolve_codes(codes, input):
    return [int(getattr(input, QA_CODE_FIELDS[c])) if isinstance(c, str) else c
            for c in codes]


def _optional_pair(value, name):
    return None if value is None else _flag_pair(value, name)


def _role_index(profile, role, where):
    if role == "other" or role not in profile.roles:
        profile._fail(
            f"{where} names {role!r}, which is not a role of this profile's bands "
            f"{dict(zip(profile.band_names, profile.roles))}")
    return profile.index(role)


@dataclass(frozen=True)
class KeepQa:
    """Keep only the observations whose QA code is in ``codes``: names from
    :data:`QA_CODE_FIELDS` (the run's ``qa_*`` codes) or ints."""

    codes: Tuple[Union[str, int], ...] = ("clear", "water")

    def __post_init__(self):
        object.__setattr__(self, "codes", _codes(self.codes, "KeepQa"))

    def _roles(self):
        return ()

    def _step(self, profile, input, shared):
        return Step.keep_qa(_resolve_codes(self.codes, input))


@dataclass(frozen=True)
class AddQa:
    """Also keep every observation whose QA code is in ``codes``, whatever an
    earlier step decided. Must be followed by :class:`Dedupe`."""

    codes: Tuple[Union[str, int], ...] = ("snow",)

    def __post_init__(self):
        object.__setattr__(self, "codes", _codes(self.codes, "AddQa"))

    def _roles(self):
        return ()

    def _step(self, profile, input, shared):
        return Step.add_qa(_resolve_codes(self.codes, input))


@dataclass(frozen=True)
class ThermalRange:
    """Keep only the observations whose thermal band lies in
    ``(min, max)``; an inclusive end also keeps a value equal to it.

    None takes ``therm_min``, ``therm_max`` and their flags from the run's
    input, which a profile sets from its :class:`Thermal`. The band is the
    profile's ``"thermal"`` band; a profile without one raises ValueError.
    The bounds are in the unit of the band after its procedure's transform,
    if any.
    """

    min: Optional[float] = None
    max: Optional[float] = None
    inclusive: Optional[Tuple[bool, bool]] = None

    def __post_init__(self):
        object.__setattr__(self, "inclusive", _optional_pair(self.inclusive, "inclusive"))

    def _roles(self):
        return ("thermal",)

    def _step(self, profile, input, shared):
        lo = input.therm_min if self.min is None else self.min
        hi = input.therm_max if self.max is None else self.max
        flags = ((input.therm_min_inclusive, input.therm_max_inclusive)
                 if self.inclusive is None else self.inclusive)
        return Step.thermal_range(_role_index(profile, "thermal", "ThermalRange"),
                                  lo, hi, flags)


@dataclass(frozen=True)
class BandRange:
    """Keep only the observations on which every band of ``roles`` lies in
    ``(min, max)``, the bands tested in order; an inclusive end also keeps a
    value equal to it.

    None takes the run's ``saturation_bands`` (a profile's ``saturation``),
    ``refl_min``, ``refl_max`` and their flags (a profile's :class:`Units`).
    """

    roles: Optional[Tuple[str, ...]] = None
    min: Optional[float] = None
    max: Optional[float] = None
    inclusive: Optional[Tuple[bool, bool]] = None

    def __post_init__(self):
        if self.roles is not None:
            roles = (self.roles,) if isinstance(self.roles, str) else tuple(self.roles)
            object.__setattr__(self, "roles", roles)
        object.__setattr__(self, "inclusive", _optional_pair(self.inclusive, "inclusive"))

    def _roles(self):
        return () if self.roles is None else self.roles

    def _step(self, profile, input, shared):
        bands = (input.saturation_bands if self.roles is None else
                 [_role_index(profile, r, "BandRange") for r in self.roles])
        lo = input.refl_min if self.min is None else self.min
        hi = input.refl_max if self.max is None else self.max
        flags = ((input.refl_min_inclusive, input.refl_max_inclusive)
                 if self.inclusive is None else self.inclusive)
        return Step.band_range(bands, lo, hi, flags)


@dataclass(frozen=True)
class GreenMedian:
    """InsufficientClear's cloud screen: drop the observations whose band is
    at or above the median of that band over the kept observations dated
    ``<= max_date``, plus ``offset``, at the dates ``scope`` names.

    None takes the run's shared ``green_filter`` (in the input's units),
    ``green_filter_scope`` and ``stat_ord``, as its configuration resolves
    them for the input.
    ``scope`` is a name from :data:`GREEN_FILTER_SCOPES` or a
    ``ccdcpp.GreenFilterScope``. ``role`` is the band, ``"green"``.
    """

    offset: Optional[float] = None
    scope: Optional[str] = None
    max_date: Optional[int] = None
    role: str = "green"

    def __post_init__(self):
        if isinstance(self.scope, GreenFilterScope):
            names = {v: k for k, v in GREEN_FILTER_SCOPES.items()}
            object.__setattr__(self, "scope", names[self.scope])
        if self.scope is not None and self.scope not in GREEN_FILTER_SCOPES:
            raise ValueError(
                f"GreenMedian: scope is {self.scope!r}; it is one of "
                f"{tuple(GREEN_FILTER_SCOPES)}")

    def _roles(self):
        return (self.role,)

    def _step(self, profile, input, shared):
        return Step.green_median(
            _role_index(profile, self.role, "GreenMedian"),
            shared["green_filter"] if self.offset is None else self.offset,
            shared["stat_ord"] if self.max_date is None else self.max_date,
            GREEN_FILTER_SCOPES[shared["green_filter_scope"] if self.scope is None
                                else self.scope],
        )


@dataclass(frozen=True)
class Dedupe:
    """Keep the first kept observation of each date."""

    def _roles(self):
        return ()

    def _step(self, profile, input, shared):
        return Step.dedupe()


_STEP_TYPES = (KeepQa, AddQa, ThermalRange, BandRange, GreenMedian, Dedupe)


@dataclass(frozen=True)
class Transform:
    """``band <- band * scale + offset`` on the per-pixel copy of the band
    with ``role``, before its procedure's steps run. The caller's array is
    never written."""

    role: str
    scale: float = 1.0
    offset: float = 0.0

    def _affine(self, profile):
        return Affine(_role_index(profile, self.role, "Transform"),
                      float(self.scale), float(self.offset))


#: ``ProcedureChain.transform``'s default: the procedure's transform in the
#: default plan.
DEFAULT_TRANSFORM = "default"


@dataclass(frozen=True)
class ProcedureChain:
    """One procedure's preprocessing.

    steps
        The steps, in order (at most ``ccdcpp.CHAIN_CAPACITY``), or None
        for the procedure's steps in the default plan.
    transform
        :data:`DEFAULT_TRANSFORM` (the default) for the procedure's
        transform in the default plan, a :class:`Transform`, or None for
        none. The default plan transforms the Standard procedure's thermal
        band (a profile's ``thermal.scale``/``thermal.offset``, none when
        that is the identity) and nothing else.
    """

    steps: Optional[Tuple[object, ...]] = None
    transform: Union[Transform, None, str] = "default"

    def __post_init__(self):
        if self.steps is not None:
            steps = tuple(self.steps)
            for step in steps:
                if not isinstance(step, _STEP_TYPES):
                    raise TypeError(
                        f"ProcedureChain: {step!r} is not a step; steps are "
                        f"{', '.join(t.__name__ for t in _STEP_TYPES)}")
            object.__setattr__(self, "steps", steps)
        if not (self.transform is None or self.transform == DEFAULT_TRANSFORM
                or isinstance(self.transform, Transform)):
            raise TypeError(
                f"ProcedureChain: transform is {self.transform!r}; it is a Transform, None "
                f"or {DEFAULT_TRANSFORM!r}")

    def _roles(self):
        roles = (self.transform.role,) if isinstance(self.transform, Transform) else ()
        for step in self.steps or ():
            roles += step._roles()
        return roles

    def _plan(self, default, profile, input, shared):
        chain = (default.chain if self.steps is None else
                 Chain([s._step(profile, input, shared) for s in self.steps]))
        if self.transform == DEFAULT_TRANSFORM:
            transform = default.transform
        else:
            transform = None if self.transform is None else self.transform._affine(profile)
        return ProcedurePlan(transform, chain)


#: The procedures of a :class:`Preprocess`, as ``ccdcpp.PreprocessPlan``
#: names them.
PROCEDURES = ("standard", "snow", "insufficient_clear")


@dataclass(frozen=True)
class Preprocess:
    """A preprocessing plan by role: a :class:`ProcedureChain` per procedure.

    standard, snow, insufficient_clear
        The Standard, PermanentSnow and InsufficientClear procedures. None
        keeps that procedure's default plan, transform and steps.

    :meth:`plan` resolves it against a profile, the run's input and its
    configuration into a ``ccdcpp.PreprocessPlan``: roles become the
    profile's row indices, QA code names the input's codes, and fields left
    None the input's values and the configuration's shared ones.
    ``Preprocess()`` is ``ccdcpp.default_plan(input, config)``.
    """

    standard: Optional[ProcedureChain] = None
    snow: Optional[ProcedureChain] = None
    insufficient_clear: Optional[ProcedureChain] = None

    def __post_init__(self):
        for name in PROCEDURES:
            value = getattr(self, name)
            if value is not None and not isinstance(value, ProcedureChain):
                raise TypeError(
                    f"Preprocess: {name} is {value!r}, not a ProcedureChain")

    def _roles(self):
        roles = ()
        for name in PROCEDURES:
            chain = getattr(self, name)
            if chain is not None:
                roles += chain._roles()
        return roles

    def plan(self, profile, input=None, config=None):
        """The ``ccdcpp.PreprocessPlan`` of this plan for ``profile``, with
        the ``input`` of the run (default: the profile's own) and its
        configuration (a ``ccdcpp.Config``; default: the profile's). Raises
        ValueError when the plan does not fit the profile
        (``ccdcpp.validate_plan``)."""
        h = profile.input() if input is None else input
        cfg = profile.config() if config is None else config
        default = default_plan(h, cfg)
        shared = cfg.resolve(h)["shared"]
        plan = PreprocessPlan(
            *(getattr(default, name) if getattr(self, name) is None
              else getattr(self, name)._plan(getattr(default, name), profile, h, shared)
              for name in PROCEDURES)
        )
        validate_plan(plan, len(profile.band_names))
        return plan


Cadence = Union[Mapping[str, float], Tuple[Tuple[str, float], ...]]
QaTableOptions = Union[Mapping[str, object], Tuple[Tuple[str, object], ...]]

#: The keyword arguments each ``ccdcpp.qa_tables`` builder takes beyond
#: ``input``, as its binding declares them: builder -> {name: (type,
#: default)}. ``SensorProfile.qa_table_options`` is checked against it.
QA_TABLE_OPTIONS = MappingProxyType({
    "landsat_c2_qa_pixel": MappingProxyType({}),
    "landsat_c1_pixel_qa": MappingProxyType({}),
    "hls_fmask": MappingProxyType({"exclude_high_aerosol": (bool, False)}),
    "sentinel2_scl": MappingProxyType({}),
})

#: The ``ccdcpp.Input`` QA code fields a table is built with, in the order
#: ``_build_table`` takes them.
_TABLE_CODE_FIELDS = ("qa_fill", "qa_clear", "qa_water", "qa_shadow", "qa_snow", "qa_cloud")


@dataclass(frozen=True)
class SensorProfile:
    """One sensor's input: band names and roles, units and QA, and the
    configuration values that follow the sensor.

    name
        A label for the profile, for messages and provenance.
    band_names
        The rows of ``spectral``, in order.
    roles
        One role from :data:`ROLES` per band, in the same order. ``"green"``
        is required; ``"thermal"`` requires ``thermal=``.
    detection, tmask, saturation
        Role names. ``detection`` is the change-magnitude bands (the
        configuration's shared ``detection_bands``, :meth:`config`),
        ``tmask`` the Tmask outlier screen (``window_screen``'s ``bands``),
        ``saturation`` the bands tested against the saturation bounds
        (``Input.saturation_bands``; may be empty, must not hold thermal).
        Each list keeps its order, which is the order the procedures
        accumulate in.
    units
        The reflective bands' units, a :class:`Units`.
    alpha, coef_floor
        The Lasso's ``alpha`` and ``coef_floor`` in the data's units; the
        configuration's shared values are these over
        ``units.reflectance_unit`` (:meth:`config`). Both are absolute and
        statements about the input unit; a wrong value does not raise, it
        fits sparser or wronger coefficients, so neither has a default.
    thermal
        A :class:`Thermal`, or None for an input without a thermal band
        (no ``"thermal"`` role, which skips the transform and filter).
    cadence
        ``stage__parameter`` -> value, the configuration's parameters that
        follow the sensor's revisit (``{"change_test__peek_reference_days":
        10.0}``), which :meth:`config` sets with ``Config.replace()``.
        Parameters not named keep the base configuration's values (pyccd's
        16-day Landsat values in "ccdc").
    stat_ord
        The configuration's shared ``stat_ord`` (:meth:`config`), the last
        date of the statistics window, as a proleptic Gregorian ordinal; None
        for no cutoff (:data:`NO_STAT_CUTOFF`). The Landsat value,
        2017-12-31, would put a series that starts later entirely outside
        the window.
    qa_table_builder
        The name of a ``ccdcpp.qa_tables`` builder for the sensor's raw QA
        band, or None for pre-classified uint8 codes.
    green_filter_scope
        The configuration's shared ``green_filter_scope``, by name from
        :data:`GREEN_FILTER_SCOPES`:
        ``"all"``, ``"stat_window"`` (the default) or ``"off"``; a
        ``ccdcpp.GreenFilterScope`` is accepted and stored as its name. Which
        observations InsufficientClear's green median filter drops.
    preprocess
        A :class:`Preprocess`, or None (the default) for the default plan of
        the profile's input. The entry points run it in place of the
        default plan; see :meth:`preprocess_plan`.
    qa_table_options
        Keyword arguments of ``qa_table_builder`` beyond the input, as a
        mapping or ``(name, value)`` pairs; empty (the default) for none.
        ``{"exclude_high_aerosol": True}`` makes ``hls_fmask`` decode high
        aerosol as cloud. Each name and the type of its value are those
        :data:`QA_TABLE_OPTIONS` lists for the builder, and the value is of
        exactly that type (``exclude_high_aerosol`` takes True or False, not
        None or 1). Stored sorted by name with every argument the builder
        declares, its default written out where none is given: with
        ``hls_fmask``, ``()`` and ``{"exclude_high_aerosol": False}`` both
        store ``(("exclude_high_aerosol", False),)``; without a builder
        nothing is stored. Anything else (``qa_table_options`` without
        ``qa_table_builder``, a name given twice, a name the builder does not
        take, a value of another type, input that is not a mapping or pairs)
        raises ValueError when the profile is constructed.

    Profiles are immutable, hashable and picklable, so one can be handed to
    worker processes where input objects cannot.
    """

    name: str
    band_names: Tuple[str, ...]
    roles: Tuple[str, ...]
    detection: Tuple[str, ...]
    tmask: Tuple[str, ...]
    saturation: Tuple[str, ...]
    units: Units
    alpha: float
    coef_floor: float
    thermal: Optional[Thermal] = None
    cadence: Cadence = field(default=())
    stat_ord: Optional[int] = LANDSAT_STAT_ORD
    qa_table_builder: Optional[str] = None
    green_filter_scope: str = "stat_window"
    preprocess: Optional[Preprocess] = None
    qa_table_options: QaTableOptions = field(default=())

    def __post_init__(self):
        for name in ("band_names", "roles", "detection", "tmask", "saturation"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if isinstance(self.green_filter_scope, GreenFilterScope):
            names = {v: k for k, v in GREEN_FILTER_SCOPES.items()}
            object.__setattr__(self, "green_filter_scope", names[self.green_filter_scope])
        items = self.cadence.items() if isinstance(self.cadence, Mapping) else self.cadence
        object.__setattr__(self, "cadence", tuple(sorted((str(k), v) for k, v in items)))
        object.__setattr__(self, "qa_table_options", self._canonical_qa_table_options())
        self._validate()

    def _fail(self, message):
        raise ValueError(f"SensorProfile {self.name!r}: {message}")

    def _validate(self):
        n = len(self.band_names)
        if n == 0:
            self._fail("band_names is empty")
        if n > CCD_MAX_BANDS:
            self._fail(f"has {n} bands; ccdcpp accepts at most {CCD_MAX_BANDS}")
        if len(set(self.band_names)) != n:
            self._fail(f"band_names repeats a name: {self.band_names}")
        if len(self.roles) != n:
            self._fail(f"has {n} band_names and {len(self.roles)} roles")
        unknown = [r for r in self.roles if r not in ROLES]
        if unknown:
            self._fail(f"unknown roles {unknown}; roles are {ROLES}")
        named = [r for r in self.roles if r != "other"]
        if len(set(named)) != len(named):
            self._fail(f"roles repeats a role other than 'other': {self.roles}")
        if "green" not in self.roles:
            self._fail("has no 'green' band; InsufficientClear's green filter reads it")
        if ("thermal" in self.roles) != (self.thermal is not None):
            self._fail("a 'thermal' band and thermal= go together")
        for list_name in ("detection", "tmask", "saturation"):
            entries = getattr(self, list_name)
            for role in entries:
                if role == "other" or role not in self.roles:
                    self._fail(
                        f"{list_name} names {role!r}, which is not a role of this "
                        f"profile's bands {dict(zip(self.band_names, self.roles))}"
                    )
            if len(set(entries)) != len(entries):
                self._fail(f"{list_name} repeats a role: {entries}")
        if not self.detection:
            self._fail("detection is empty")
        if not self.tmask:
            self._fail("tmask is empty")
        if "thermal" in self.saturation:
            self._fail("saturation holds 'thermal'; the thermal bounds test it")
        for key, _ in self.cadence:
            if "__" not in key:
                self._fail(f"cadence names {key!r}; its keys are stage__parameter")
        if self.green_filter_scope not in GREEN_FILTER_SCOPES:
            self._fail(
                f"green_filter_scope is {self.green_filter_scope!r}; it is one of "
                f"{tuple(GREEN_FILTER_SCOPES)}"
            )
        if self.preprocess is not None:
            if not isinstance(self.preprocess, Preprocess):
                self._fail(f"preprocess is {self.preprocess!r}, not a ccdcpp.Preprocess")
            for role in self.preprocess._roles():
                if role == "other" or role not in self.roles:
                    self._fail(
                        f"preprocess names {role!r}, which is not a role of this "
                        f"profile's bands {dict(zip(self.band_names, self.roles))}"
                    )
        if self.qa_table_builder is not None and not callable(
            getattr(qa_tables, self.qa_table_builder, None)
        ):
            self._fail(f"ccdcpp.qa_tables has no builder {self.qa_table_builder!r}")
        self._validate_qa_table_options()

    def _canonical_qa_table_options(self):
        """``qa_table_options`` checked as given, then stored canonically:
        every argument :data:`QA_TABLE_OPTIONS` declares for the builder,
        sorted by name, its default written out where none is given."""
        builder = self.qa_table_builder
        if builder is not None and not isinstance(builder, str):
            self._fail(f"qa_table_builder is {builder!r}; it is a str or None")
        given = self.qa_table_options
        if isinstance(given, Mapping):
            pairs = tuple(given.items())
        elif isinstance(given, (tuple, list)):
            pairs = tuple(given)
            for pair in pairs:
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    self._fail(f"qa_table_options entry {pair!r} is not a (name, value) pair")
        else:
            self._fail(
                f"qa_table_options is {given!r}; it is a mapping or (name, value) pairs")
        names = [name for name, _ in pairs]
        for name in names:
            if not isinstance(name, str):
                self._fail(f"qa_table_options name {name!r} is not a str")
        if len(set(names)) != len(names):
            self._fail(f"qa_table_options repeats a name: {names}")
        if builder is None:
            if pairs:
                self._fail(f"qa_table_options {dict(pairs)} needs a qa_table_builder")
            return ()
        declared = QA_TABLE_OPTIONS.get(builder)
        if declared is None:
            if not pairs:
                return ()
            self._fail(
                f"qa_table_options {dict(pairs)}: ccdcpp.qa_tables has no builder "
                f"{self.qa_table_builder!r} in QA_TABLE_OPTIONS")
        unknown = {name: value for name, value in pairs if name not in declared}
        if unknown:
            self._fail(
                f"ccdcpp.qa_tables.{self.qa_table_builder} does not take qa_table_options "
                f"{unknown}; it takes {tuple(declared) or 'none'}")
        for name, value in pairs:
            kind = declared[name][0]
            if type(value) is not kind:
                hint = ""
                if isinstance(value, np.generic) and type(value.item()) is kind:
                    hint = f"; pass {kind.__name__}(value) for a numpy scalar"
                self._fail(
                    f"qa_table_options {name!r} is {value!r} ({type(value).__name__}); "
                    f"ccdcpp.qa_tables.{self.qa_table_builder} takes a {kind.__name__}{hint}")
        given = dict(pairs)
        return tuple(sorted((name, given.get(name, default))
                            for name, (_, default) in declared.items()))

    def _validate_qa_table_options(self):
        options = self.qa_table_options
        if not options:
            return
        defaults = Input()
        codes = tuple(int(getattr(defaults, name)) for name in _TABLE_CODE_FIELDS)
        try:
            _build_table(self.qa_table_builder, codes, options)
        except TypeError as error:
            self._fail(
                f"ccdcpp.qa_tables.{self.qa_table_builder} rejects qa_table_options "
                f"{dict(options)}, which QA_TABLE_OPTIONS allows: {error}")

    def index(self, role):
        """The row of the band with ``role``."""
        return self.roles.index(role)

    def input(self):
        """The ``ccdcpp.Input`` of this profile's data, built once per profile
        and shared: copy it (``copy.copy``) before changing it.

        Roles are the profile's; ``saturation`` becomes row indices, in its
        own order. Every field the profile does not describe keeps its
        ``Input()`` default.
        """
        return _input_of(self)

    def config(self, base="ccdc"):
        """``base`` (a ``ccdcpp.config()`` source) with this profile's values, a
        ``ccdcpp.Config``: its ``cadence``; its detection bands, statistics cutoff, green
        filter scope and value, and Lasso ``alpha`` and ``coef_floor`` (the data's values
        over ``units.reflectance_unit``) as shared values; and its Tmask bands (window_screen
        ``bands``), where ``base``'s strategy has that parameter. A run with ``profile=`` and
        no ``config=`` uses ``profile.config()``. Built once per profile and preset name.
        """
        if isinstance(base, str):
            return _config_of(self, base)
        return self._config(base)

    def _config(self, base):
        from .config import config as _config

        cfg = _config(base).replace(**dict(self.cadence))
        unit = self.units.reflectance_unit
        params = {
            "shared__detection_bands": list(self.detection),
            "shared__stat_ord": self.stat_ord,
            "shared__green_filter_scope": self.green_filter_scope,
            "shared__green_filter": self.units.median_green_filter / unit,
            "shared__alpha": self.alpha / unit,
            "shared__coef_floor": self.coef_floor / unit,
        }
        if "window_screen" in cfg.stages() and "bands" in cfg.params("window_screen"):
            params["window_screen__bands"] = list(self.tmask)
        return cfg.replace(**params)

    def qa_table(self, input=None):
        """The ``ccdcpp.QaTable`` of ``qa_table_builder``, or None for
        pre-classified input.

        Built with the ``qa_*`` codes of ``input`` (default: this profile's
        own), so pass the input the run uses, and with ``qa_table_options``
        as keyword arguments. Tables are cached per builder, codes and
        builder arguments, and are read-only.
        """
        if self.qa_table_builder is None:
            return None
        h = self.input() if input is None else input
        codes = tuple(int(getattr(h, name)) for name in _TABLE_CODE_FIELDS)
        return _build_table(self.qa_table_builder, codes, self.qa_table_options)

    def preprocess_plan(self, input=None, config=None):
        """The ``ccdcpp.PreprocessPlan`` this profile runs, a new object.

        ``preprocess`` resolved against this profile, ``input`` (default: the
        profile's own; pass the input the run uses) and ``config`` (a
        ``ccdcpp.Config``; default: the profile's), or
        ``ccdcpp.default_plan(input, config)`` when ``preprocess`` is None.
        """
        h = self.input() if input is None else input
        cfg = self.config() if config is None else config
        if self.preprocess is None:
            return default_plan(h, cfg)
        return self.preprocess.plan(self, h, cfg)

    def check(self, spectral, band_names=None):
        """Raise ValueError unless the band axis of ``spectral`` has one row
        per band of the profile, and, with ``band_names``, unless those names
        are the profile's ``band_names`` in the profile's order.

        The band axis is second from last: ``(B, T)`` for ``detect``,
        ``(H, W, B, T)`` for the cube entry points. ``band_names`` names the
        rows of ``spectral`` as the caller holds them (a store's band list,
        say), a sequence of str; a single str raises ``TypeError``. Without
        it only the count is checked, and a stack of the right count in
        another order runs as if it were in the profile's order.
        """
        shape = np.shape(spectral)
        if len(shape) < 2:
            self._fail(
                f"spectral has shape {shape}, with no band axis; it needs "
                f"(..., {len(self.band_names)}, T)"
            )
        if shape[-2] != len(self.band_names):
            self._fail(
                f"spectral has {shape[-2]} bands on its band axis (shape {shape}); "
                f"the profile has {len(self.band_names)}: {', '.join(self.band_names)}"
            )
        if band_names is not None:
            self._check_band_names(band_names)

    def _check_band_names(self, band_names):
        if isinstance(band_names, (str, bytes)):
            raise TypeError(
                f"SensorProfile {self.name!r}: band_names is a sequence of "
                f"names, one per row, not a single {type(band_names).__name__}"
            )
        given = tuple(band_names)
        if not all(isinstance(b, str) for b in given):
            raise TypeError(
                f"SensorProfile {self.name!r}: band_names holds "
                f"{', '.join(sorted({type(b).__name__ for b in given if not isinstance(b, str)}))}; "
                f"every name is a str"
            )
        given = tuple(str(b) for b in given)
        if given == self.band_names:
            return
        expected = ", ".join(self.band_names)
        if sorted(given) == sorted(self.band_names):
            row = next(i for i, (a, b) in enumerate(zip(given, self.band_names)) if a != b)
            self._fail(
                f"band_names are the profile's bands in another order: row {row} "
                f"is {given[row]!r}, where the profile has {self.band_names[row]!r}; "
                f"reorder the band axis of spectral to {expected}"
            )
        missing = [b for b in self.band_names if b not in given]
        extra = [b for b in given if b not in self.band_names]
        repeated = sorted({b for b in given if given.count(b) > 1})
        parts = [f"band_names are {', '.join(given)}"]
        if missing:
            parts.append(f"missing {', '.join(missing)}")
        if extra:
            parts.append(f"not in the profile: {', '.join(extra)}")
        if repeated:
            parts.append(f"repeated: {', '.join(repeated)}")
        self._fail("; ".join(parts) + f"; the profile has {expected}")


@lru_cache(maxsize=None)
def _build_table(builder, codes, builder_options=()):
    h = Input()
    for name, code in zip(_TABLE_CODE_FIELDS, codes):
        setattr(h, name, code)
    return getattr(qa_tables, builder)(h, **dict(builder_options))


@lru_cache(maxsize=None)
def _input_of(profile):
    h = Input()
    h.roles = list(profile.roles)
    h.saturation_bands = [profile.index(r) for r in profile.saturation]
    h.reflectance_unit = profile.units.reflectance_unit
    h.refl_min = profile.units.refl_min
    h.refl_max = profile.units.refl_max
    h.refl_min_inclusive, h.refl_max_inclusive = profile.units.refl_inclusive
    if profile.thermal is not None:
        h.thermal_scale = profile.thermal.scale
        h.thermal_offset = profile.thermal.offset
        h.therm_min = profile.thermal.min
        h.therm_max = profile.thermal.max
        h.therm_min_inclusive, h.therm_max_inclusive = profile.thermal.inclusive
    return h


@lru_cache(maxsize=None)
def _config_of(profile, base):
    return profile._config(base)
