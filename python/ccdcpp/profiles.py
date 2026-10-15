"""Sensor profiles: one sensor's band layout, input units and options, by role.

A :class:`SensorProfile` names the rows of the ``spectral`` array and says
which role each one plays. The detection, Tmask and saturation lists are
written as role names and resolved to row indices when the options are built,
so the same profile text holds for any band order. ``ccdcpp.detect``,
``ccdcpp.detect_cube`` and ``ccdcpp.detect_cube_columns`` take one as
``profile=``.

A new sensor is one constructor call. NOT SHIPPED: the profile below is an
example of the shape, its values are not validated, and ccdcpp.presets does
not contain it::

    from ccdcpp import CELSIUS_HUNDREDTHS, REFLECTANCE, SensorProfile

    def hls_l30():
        # HLS v2.0 L30 surface reflectance divided by 10000, thermal B10 in
        # degrees Celsius / 100. Fmask QA.
        return SensorProfile(
            name="hls_l30",
            band_names=("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B10"),
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
            stat_ord=None,
            qa_table_builder="hls_fmask",
        )

The detection count sets the chi-squared degrees of freedom of the change and
outlier thresholds; a profile with other than five detection bands gets
thresholds computed for its count (see :meth:`SensorProfile.options`).

A profile's ``preprocess=`` is a :class:`Preprocess`: per procedure, the
filter steps that build the processing mask and the transform before them,
written with roles and QA code names. None, the default, is the default plan
of the profile's options (``ccdcpp.default_plan``). The Standard procedure
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
from typing import Mapping, Optional, Tuple, Union
import warnings

import numpy as np

from .ccdcpp import (
    CCD_MAX_BANDS,
    NO_BAND,
    Affine,
    Chain,
    GreenFilterScope,
    HarmonicOptions,
    LassoOptions,
    PreprocessPlan,
    ProcedurePlan,
    Step,
    chi2_quantile,
    default_plan,
    qa_tables,
    validate_plan,
)

#: The roles a band can play. Every role but ``"other"`` names at most one
#: band of a profile; ``"other"`` bands are fitted and reported, and no list
#: can name them. ``"nir"`` is the Landsat near infrared: TM/ETM+ band 4,
#: OLI band 5 (865 nm), and on Sentinel-2 the 865 nm B8A; the broad 842 nm
#: Sentinel-2 B08 is ``"nir_broad"``.
ROLES = (
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
    "thermal",
    "coastal",
    "red_edge_1",
    "red_edge_2",
    "red_edge_3",
    "nir_broad",
    "other",
)

#: The informational ``HarmonicOptions`` index fields a profile fills in
#: from its roles (``ccdcpp.NO_BAND`` for a role the profile lacks).
#: ``GREEN_IDX`` and ``THERMAL_IDX`` are the ones the procedures read.
_INDEX_FIELDS = {
    "blue": "BLUE_IDX",
    "green": "GREEN_IDX",
    "red": "RED_IDX",
    "nir": "NIR_IDX",
    "swir1": "SWIR1_IDX",
    "swir2": "SWIR2_IDX",
    "thermal": "THERMAL_IDX",
}

#: The ``HarmonicOptions`` fields ``cadence=`` may set: the cadence and
#: change-probability options and the two window sizes.
CADENCE_FIELDS = (
    "MEOW_SIZE",
    "PEEK_SIZE",
    "PEEK_REFERENCE_DAYS",
    "VARIOGRAM_MIN_LAG_DAYS",
    "RETRAIN_MIN_OBS",
    "RETRAIN_SPAN_FACTOR",
    "SEASONAL_RMSE_COUNT",
    "SEASONAL_RMSE_DIVISOR",
    "CHANGE_PROBABILITY",
)

#: The probability behind the ``OUTLIER_THRESHOLD`` default:
#: ``chi2_quantile(OUTLIER_PROBABILITY, 5)`` is that default to the last bit.
OUTLIER_PROBABILITY = 1.0 - 1e-6

#: ``green_filter_scope`` names -> ``ccdcpp.GreenFilterScope`` values.
GREEN_FILTER_SCOPES = {
    "all": GreenFilterScope.All,
    "stat_window": GreenFilterScope.StatWindow,
    "off": GreenFilterScope.Off,
}

#: ``stat_ord=None`` puts every date inside the statistics window.
NO_STAT_CUTOFF = 2**63 - 1

#: The Landsat statistics cutoff, 2017-12-31 (``HarmonicOptions().STAT_ORD``).
LANDSAT_STAT_ORD = 736694


def _flag_pair(value, name):
    """``value`` as a ``(bool, bool)``, for the inclusive-bound fields."""
    pair = tuple(value)
    if len(pair) != 2 or not all(isinstance(v, (bool, np.bool_)) for v in pair):
        raise ValueError(f"{name} is {value!r}; it must be two bools (lo, hi)")
    return (bool(pair[0]), bool(pair[1]))


@dataclass(frozen=True)
class Units:
    """The unit-coupled thresholds of the reflective bands.

    refl_min, refl_max
        ``REFL_MIN``, ``REFL_MAX``: the saturation bounds, tested on the
        saturation bands.
    median_green_filter
        ``MEDIAN_GREEN_FILTER``: the offset of InsufficientClear's green
        median cloud screen.
    refl_inclusive
        ``(REFL_MIN_INCLUSIVE, REFL_MAX_INCLUSIVE)``: whether a value equal
        to each bound is kept (default ``(False, False)``, pyccd's exclusive
        range, which rejects NaN). NaN passes only with both True.
    """

    refl_min: float
    refl_max: float
    median_green_filter: float
    refl_inclusive: Tuple[bool, bool] = (False, False)

    def __post_init__(self):
        object.__setattr__(self, "refl_inclusive", _flag_pair(self.refl_inclusive, "refl_inclusive"))


@dataclass(frozen=True)
class Thermal:
    """The thermal band's transform and bounds.

    scale, offset
        ``THERMAL_SCALE``, ``THERMAL_OFFSET``: the Standard procedure reads
        ``thermal * scale + offset``. ``(1.0, 0.0)`` for input already in the
        bounds' unit.
    min, max
        ``THERM_MIN``, ``THERM_MAX``: the valid range, after the transform.
    inclusive
        ``(THERM_MIN_INCLUSIVE, THERM_MAX_INCLUSIVE)``: whether a value equal
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


#: Collection 1 DN: reflectance x 10000 (``HarmonicOptions()`` defaults).
DN = Units(refl_min=0.0, refl_max=10000.0, median_green_filter=400.0)

#: Surface reflectance, 0..1: :data:`DN` divided by 10000.
REFLECTANCE = Units(refl_min=0.0, refl_max=1.0, median_green_filter=0.04)

#: Thermal as Kelvin x 10, transformed to degrees Celsius x 100
#: (``HarmonicOptions()`` defaults).
KELVIN_TENTHS = Thermal(scale=10.0, offset=-27315.0, min=-9320.0, max=7070.0)

#: Thermal already in degrees Celsius / 100, the unit that shares the
#: reflectance scale factor of 10000 with the optical bands (one Lasso alpha
#: serves every band only if every band has the same scale).
CELSIUS_HUNDREDTHS = Thermal(scale=1.0, offset=0.0, min=-0.9320, max=0.7070)


#------------------------------------------------------------------------------
# Preprocessing steps, by role and QA code name.
#
# Each step resolves to a ccdcpp.Step against a profile and the
# HarmonicOptions of the run (SensorProfile.preprocess_plan). A field left
# None takes the value default_plan() would use from those options, so a
# step with every field None is the default plan's step.
#------------------------------------------------------------------------------

#: QA code names -> the ``HarmonicOptions`` fields that hold the codes.
QA_CODE_FIELDS = {
    "fill": "QA_FILL",
    "clear": "QA_CLEAR",
    "water": "QA_WATER",
    "shadow": "QA_SHADOW",
    "snow": "QA_SNOW",
    "cloud": "QA_CLOUD",
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


def _resolve_codes(codes, options):
    return [int(getattr(options, QA_CODE_FIELDS[c])) if isinstance(c, str) else c
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
    :data:`QA_CODE_FIELDS` (the run's ``QA_*`` codes) or ints."""

    codes: Tuple[Union[str, int], ...] = ("clear", "water")

    def __post_init__(self):
        object.__setattr__(self, "codes", _codes(self.codes, "KeepQa"))

    def _roles(self):
        return ()

    def _step(self, profile, options):
        return Step.keep_qa(_resolve_codes(self.codes, options))


@dataclass(frozen=True)
class AddQa:
    """Also keep every observation whose QA code is in ``codes``, whatever an
    earlier step decided. Must be followed by :class:`Dedupe`."""

    codes: Tuple[Union[str, int], ...] = ("snow",)

    def __post_init__(self):
        object.__setattr__(self, "codes", _codes(self.codes, "AddQa"))

    def _roles(self):
        return ()

    def _step(self, profile, options):
        return Step.add_qa(_resolve_codes(self.codes, options))


@dataclass(frozen=True)
class ThermalRange:
    """Keep only the observations whose thermal band lies in
    ``(min, max)``; an inclusive end also keeps a value equal to it.

    None takes ``THERM_MIN``, ``THERM_MAX`` and their flags from the run's
    options, which a profile sets from its :class:`Thermal`. The band is the
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

    def _step(self, profile, options):
        lo = options.THERM_MIN if self.min is None else self.min
        hi = options.THERM_MAX if self.max is None else self.max
        flags = ((options.THERM_MIN_INCLUSIVE, options.THERM_MAX_INCLUSIVE)
                 if self.inclusive is None else self.inclusive)
        return Step.thermal_range(_role_index(profile, "thermal", "ThermalRange"),
                                  lo, hi, flags)


@dataclass(frozen=True)
class BandRange:
    """Keep only the observations on which every band of ``roles`` lies in
    ``(min, max)``, the bands tested in order; an inclusive end also keeps a
    value equal to it.

    None takes the run's ``SATURATION_BANDS`` (a profile's ``saturation``),
    ``REFL_MIN``, ``REFL_MAX`` and their flags (a profile's :class:`Units`).
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

    def _step(self, profile, options):
        bands = (options.SATURATION_BANDS if self.roles is None else
                 [_role_index(profile, r, "BandRange") for r in self.roles])
        lo = options.REFL_MIN if self.min is None else self.min
        hi = options.REFL_MAX if self.max is None else self.max
        flags = ((options.REFL_MIN_INCLUSIVE, options.REFL_MAX_INCLUSIVE)
                 if self.inclusive is None else self.inclusive)
        return Step.band_range(bands, lo, hi, flags)


@dataclass(frozen=True)
class GreenMedian:
    """InsufficientClear's cloud screen: drop the observations whose band is
    at or above the median of that band over the kept observations dated
    ``<= max_date``, plus ``offset``, at the dates ``scope`` names.

    None takes the run's ``MEDIAN_GREEN_FILTER`` (a profile's
    ``units.median_green_filter``), ``GREEN_FILTER_SCOPE`` and ``STAT_ORD``.
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

    def _step(self, profile, options):
        return Step.green_median(
            _role_index(profile, self.role, "GreenMedian"),
            options.MEDIAN_GREEN_FILTER if self.offset is None else self.offset,
            options.STAT_ORD if self.max_date is None else self.max_date,
            (options.GREEN_FILTER_SCOPE if self.scope is None
             else GREEN_FILTER_SCOPES[self.scope]),
        )


@dataclass(frozen=True)
class Dedupe:
    """Keep the first kept observation of each date."""

    def _roles(self):
        return ()

    def _step(self, profile, options):
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

    def _plan(self, default, profile, options):
        chain = (default.chain if self.steps is None else
                 Chain([s._step(profile, options) for s in self.steps]))
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

    :meth:`plan` resolves it against a profile and the run's options into
    a ``ccdcpp.PreprocessPlan``: roles become the profile's row indices, QA
    code names the options' codes, and fields left None the options'
    values. ``Preprocess()`` is ``ccdcpp.default_plan(options)``.
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

    def plan(self, profile, options=None):
        """The ``ccdcpp.PreprocessPlan`` of this plan for ``profile``, with
        the ``options`` of the run (default: the profile's own). Raises
        ValueError when the plan does not fit the profile
        (``ccdcpp.validate_plan``)."""
        h = profile.options()[0] if options is None else options
        default = default_plan(h)
        plan = PreprocessPlan(
            *(getattr(default, name) if getattr(self, name) is None
              else getattr(self, name)._plan(getattr(default, name), profile, h)
              for name in PROCEDURES)
        )
        validate_plan(plan, len(profile.band_names))
        return plan


Cadence = Union[Mapping[str, float], Tuple[Tuple[str, float], ...]]
QaTableOptions = Union[Mapping[str, object], Tuple[Tuple[str, object], ...]]

#: The ``HarmonicOptions`` QA code fields a table is built with, in the order
#: ``_build_table`` takes them.
_TABLE_CODE_FIELDS = ("QA_FILL", "QA_CLEAR", "QA_WATER", "QA_SHADOW", "QA_SNOW", "QA_CLOUD")


@dataclass(frozen=True)
class SensorProfile:
    """One sensor's input: band names and roles, units and options.

    name
        A label for the profile, for messages and provenance.
    band_names
        The rows of ``spectral``, in order.
    roles
        One role from :data:`ROLES` per band, in the same order. ``"green"``
        is required; ``"thermal"`` requires ``thermal=``.
    detection, tmask, saturation
        Role names. ``detection`` is the change-magnitude bands
        (``DETECTION_BANDS``), ``tmask`` the Tmask outlier screen
        (``TMASK_BANDS``), ``saturation`` the bands tested against the
        saturation bounds (``SATURATION_BANDS``; may be empty, must not hold
        thermal). Each list keeps its order, which is the order the
        procedures accumulate in.
    units
        The reflective thresholds, a :class:`Units`.
    alpha, coef_floor
        ``LassoOptions.alpha`` and ``coef_floor``. Both are absolute and
        statements about the input unit; a wrong value does not raise, it
        fits sparser or wronger coefficients, so neither has a default.
    thermal
        A :class:`Thermal`, or None for an input without a thermal band
        (``THERMAL_IDX = NO_BAND``, which skips the transform and filter).
    cadence
        ``HarmonicOptions`` field -> value, from :data:`CADENCE_FIELDS`.
        Fields not named keep pyccd's 16-day Landsat values.
    stat_ord
        ``STAT_ORD``, the last date of the statistics window, as a proleptic
        Gregorian ordinal; None for no cutoff (:data:`NO_STAT_CUTOFF`). The
        Landsat value, 2017-12-31, would put a series that starts later
        entirely outside the window.
    change_threshold, outlier_threshold
        None (the default) computes ``chi2_quantile(CHANGE_PROBABILITY, n)``
        and ``chi2_quantile(OUTLIER_PROBABILITY, n)`` for ``n`` detection
        bands, which for five bands are the ``HarmonicOptions`` defaults to
        the last bit. A number is used as given.
    qa_table_builder
        The name of a ``ccdcpp.qa_tables`` builder for the sensor's raw QA
        band, or None for pre-classified uint8 codes.
    green_filter_scope
        ``GREEN_FILTER_SCOPE``, by name from :data:`GREEN_FILTER_SCOPES`:
        ``"all"``, ``"stat_window"`` (the default) or ``"off"``; a
        ``ccdcpp.GreenFilterScope`` is accepted and stored as its name. Which
        observations InsufficientClear's green median filter drops.
    preprocess
        A :class:`Preprocess`, or None (the default) for the default plan of
        the profile's options. The entry points run it in place of the
        default plan; see :meth:`preprocess_plan`.
    qa_table_options
        Keyword arguments of ``qa_table_builder`` beyond the options, as a
        mapping or ``(name, value)`` pairs, stored sorted by name; empty
        (the default) for none. ``{"exclude_high_aerosol": True}`` makes
        ``hls_fmask`` decode high aerosol as cloud. Values must be hashable.
        Requires ``qa_table_builder``; a name the builder does not take
        raises ValueError when the profile is constructed.

    Profiles are immutable, hashable and picklable, so one can be handed to
    worker processes where option objects cannot.
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
    change_threshold: Optional[float] = None
    outlier_threshold: Optional[float] = None
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
        items = (self.qa_table_options.items() if isinstance(self.qa_table_options, Mapping)
                 else self.qa_table_options)
        object.__setattr__(self, "qa_table_options",
                           tuple(sorted((str(k), v) for k, v in items)))
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
            if key not in CADENCE_FIELDS:
                self._fail(f"cadence names {key!r}; it may set {CADENCE_FIELDS}")
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

    def _validate_qa_table_options(self):
        options = self.qa_table_options
        if not options:
            return
        if self.qa_table_builder is None:
            self._fail(f"qa_table_options {dict(options)} needs a qa_table_builder")
        names = [name for name, _ in options]
        if len(set(names)) != len(names):
            self._fail(f"qa_table_options repeats a name: {names}")
        try:
            hash(options)
        except TypeError:
            self._fail(f"qa_table_options {options} holds an unhashable value")
        defaults = HarmonicOptions()
        codes = tuple(int(getattr(defaults, name)) for name in _TABLE_CODE_FIELDS)
        try:
            _build_table(self.qa_table_builder, codes, options)
        except TypeError as error:
            self._fail(
                f"ccdcpp.qa_tables.{self.qa_table_builder} does not take "
                f"qa_table_options {dict(options)}: {error}")

    def index(self, role):
        """The row of the band with ``role``."""
        return self.roles.index(role)

    def options(self):
        """``(HarmonicOptions, LassoOptions)`` for this profile, new objects.

        Roles become row indices: each list in its own order, each index
        the role's position in ``band_names``. Every field the profile does
        not describe keeps its ``HarmonicOptions()`` / ``LassoOptions()``
        default.

        Warns (``UserWarning``) when the detection count is not 5 and a
        threshold is still the chi2(5) default, which tests the change
        magnitude against the wrong distribution.
        """
        h, lo = HarmonicOptions(), LassoOptions()
        defaults = HarmonicOptions()

        h.NUM_BANDS = len(self.band_names)
        h.DETECTION_BANDS = [self.index(r) for r in self.detection]
        h.TMASK_BANDS = [self.index(r) for r in self.tmask]
        h.SATURATION_BANDS = [self.index(r) for r in self.saturation]
        for role, attr in _INDEX_FIELDS.items():
            setattr(h, attr, self.index(role) if role in self.roles else NO_BAND)

        h.REFL_MIN = self.units.refl_min
        h.REFL_MAX = self.units.refl_max
        h.REFL_MIN_INCLUSIVE, h.REFL_MAX_INCLUSIVE = self.units.refl_inclusive
        h.MEDIAN_GREEN_FILTER = self.units.median_green_filter
        h.GREEN_FILTER_SCOPE = GREEN_FILTER_SCOPES[self.green_filter_scope]
        if self.thermal is not None:
            h.THERMAL_SCALE = self.thermal.scale
            h.THERMAL_OFFSET = self.thermal.offset
            h.THERM_MIN = self.thermal.min
            h.THERM_MAX = self.thermal.max
            h.THERM_MIN_INCLUSIVE, h.THERM_MAX_INCLUSIVE = self.thermal.inclusive

        for key, value in self.cadence:
            setattr(h, key, value)
        h.STAT_ORD = NO_STAT_CUTOFF if self.stat_ord is None else self.stat_ord

        df = len(self.detection)
        h.CHANGE_THRESHOLD = (
            chi2_quantile(h.CHANGE_PROBABILITY, df)
            if self.change_threshold is None else self.change_threshold
        )
        h.OUTLIER_THRESHOLD = (
            chi2_quantile(OUTLIER_PROBABILITY, df)
            if self.outlier_threshold is None else self.outlier_threshold
        )
        if df != 5:
            stale = [
                name for name in ("CHANGE_THRESHOLD", "OUTLIER_THRESHOLD")
                if getattr(h, name) == getattr(defaults, name)
            ]
            if stale:
                warnings.warn(
                    f"SensorProfile {self.name!r} has {df} detection bands but "
                    f"{' and '.join(stale)} is the chi2(5) default; leave it None "
                    f"to compute it for {df} degrees of freedom",
                    UserWarning,
                    stacklevel=2,
                )

        lo.alpha = self.alpha
        lo.coef_floor = self.coef_floor
        return h, lo

    def qa_table(self, options=None):
        """The ``ccdcpp.QaTable`` of ``qa_table_builder``, or None for
        pre-classified input.

        Built with the ``QA_*`` codes of ``options`` (default: this
        profile's own), so pass the options the run uses, and with
        ``qa_table_options`` as keyword arguments. Tables are cached per
        builder, codes and builder arguments, and are read-only.
        """
        if self.qa_table_builder is None:
            return None
        h = self.options()[0] if options is None else options
        codes = tuple(int(getattr(h, name)) for name in _TABLE_CODE_FIELDS)
        return _build_table(self.qa_table_builder, codes, self.qa_table_options)

    def preprocess_plan(self, options=None):
        """The ``ccdcpp.PreprocessPlan`` this profile runs, a new object.

        ``preprocess`` resolved against this profile and ``options``
        (default: the profile's own; pass the options the run uses), or
        ``ccdcpp.default_plan(options)`` when ``preprocess`` is None.
        """
        h = self.options()[0] if options is None else options
        if self.preprocess is None:
            return default_plan(h)
        return self.preprocess.plan(self, h)

    def check(self, spectral):
        """Raise ValueError unless the band axis of ``spectral`` has one row
        per band of the profile.

        The band axis is second from last: ``(B, T)`` for ``detect``,
        ``(H, W, B, T)`` for the cube entry points.
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


@lru_cache(maxsize=None)
def _build_table(builder, codes, builder_options=()):
    h = HarmonicOptions()
    for name, code in zip(_TABLE_CODE_FIELDS, codes):
        setattr(h, name, code)
    return getattr(qa_tables, builder)(h, **dict(builder_options))
