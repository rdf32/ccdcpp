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
"""
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Mapping, Optional, Tuple, Union
import warnings

import numpy as np

from .ccdcpp import (
    CCD_MAX_BANDS,
    NO_BAND,
    HarmonicOptions,
    LassoOptions,
    chi2_quantile,
    qa_tables,
)

#: The roles a band can play. Every role but ``"other"`` names at most one
#: band of a profile; ``"other"`` bands are fitted and reported, and no list
#: can name them.
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
    "nir_narrow",
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

#: ``stat_ord=None`` puts every date inside the statistics window.
NO_STAT_CUTOFF = 2**63 - 1

#: The Landsat statistics cutoff, 2017-12-31 (``HarmonicOptions().STAT_ORD``).
LANDSAT_STAT_ORD = 736694


@dataclass(frozen=True)
class Units:
    """The unit-coupled thresholds of the reflective bands.

    refl_min, refl_max
        ``REFL_MIN``, ``REFL_MAX``: the saturation bounds, inclusive, tested
        on the saturation bands.
    median_green_filter
        ``MEDIAN_GREEN_FILTER``: the offset of InsufficientClear's green
        median cloud screen.
    """

    refl_min: float
    refl_max: float
    median_green_filter: float


@dataclass(frozen=True)
class Thermal:
    """The thermal band's transform and bounds.

    scale, offset
        ``THERMAL_SCALE``, ``THERMAL_OFFSET``: the Standard procedure reads
        ``thermal * scale + offset``. ``(1.0, 0.0)`` for input already in the
        bounds' unit.
    min, max
        ``THERM_MIN``, ``THERM_MAX``: the valid range, exclusive, after the
        transform.
    """

    scale: float
    offset: float
    min: float
    max: float


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


Cadence = Union[Mapping[str, float], Tuple[Tuple[str, float], ...]]


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

    def __post_init__(self):
        for name in ("band_names", "roles", "detection", "tmask", "saturation"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        items = self.cadence.items() if isinstance(self.cadence, Mapping) else self.cadence
        object.__setattr__(self, "cadence", tuple(sorted((str(k), v) for k, v in items)))
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
        if self.qa_table_builder is not None and not callable(
            getattr(qa_tables, self.qa_table_builder, None)
        ):
            self._fail(f"ccdcpp.qa_tables has no builder {self.qa_table_builder!r}")

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
        h.MEDIAN_GREEN_FILTER = self.units.median_green_filter
        if self.thermal is not None:
            h.THERMAL_SCALE = self.thermal.scale
            h.THERMAL_OFFSET = self.thermal.offset
            h.THERM_MIN = self.thermal.min
            h.THERM_MAX = self.thermal.max

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
        profile's own), so pass the options the run uses. Tables are cached
        per builder and codes, and are read-only.
        """
        if self.qa_table_builder is None:
            return None
        h = self.options()[0] if options is None else options
        codes = tuple(
            int(getattr(h, name))
            for name in ("QA_FILL", "QA_CLEAR", "QA_WATER", "QA_SHADOW",
                         "QA_SNOW", "QA_CLOUD")
        )
        return _build_table(self.qa_table_builder, codes)

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
def _build_table(builder, codes):
    h = HarmonicOptions()
    (h.QA_FILL, h.QA_CLEAR, h.QA_WATER, h.QA_SHADOW, h.QA_SNOW, h.QA_CLOUD) = codes
    return getattr(qa_tables, builder)(h)
