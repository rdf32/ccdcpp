"""Sensor profiles: one sensor's band layout, input units and QA, by role.

A :class:`SensorProfile` describes the data, and only the data: it names the
rows of the ``spectral`` array, says which role each one plays, and gives the
units and the QA decoding. Its ``input()`` is the ``ccdcpp.Input`` of the
data, roles included, so a configuration that names bands by role holds for
any band order. ``ccdcpp.detect`` and ``ccdcpp.detect_cube`` take one as
``input=``; the configuration (``config=``, "ccdc" by default) is the same
for every profile.

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
            units=REFLECTANCE,
            thermal=CELSIUS_HUNDREDTHS,
            qa_table_builder="landsat_c2_qa_pixel",
        )

The configuration names its detection and Tmask bands by role ("green",
"swir1", ...); a profile whose roles lack one is refused when it runs.
"""
from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping, Optional, Tuple, Union

import numpy as np

from .ccdcpp import CCD_MAX_BANDS, ROLES, Input, qa_tables


@dataclass(frozen=True)
class Units:
    """The units of the reflective bands: reflectance r is held as
    ``r * reflectance_unit + offset``.

    reflectance_unit
        ``Input.reflectance_unit``: the value one unit of reflectance has in
        the data (Collection-1 DN 10000, reflectance 1).
    offset
        ``Input.reflectance_offset``: the data's value of reflectance 0
        (0, the default, for both).
    """

    reflectance_unit: float
    offset: float = 0.0


@dataclass(frozen=True)
class Thermal:
    """The thermal band's transform and the unit it lands on.

    scale, offset
        ``Input.thermal_scale``, ``Input.thermal_offset``: a procedure that
        transforms thermal reads ``thermal * scale + offset``. ``(1.0, 0.0)``
        for input already in the unit.
    unit
        ``Input.thermal_unit``: the value 100 degrees Celsius has after the
        transform (10000 for Celsius x 100, 1 for Celsius / 100). The
        configuration's thermal bounds in degrees Celsius resolve to
        ``c * 100 * unit / 10000``.
    """

    scale: float
    offset: float
    unit: float


#: Collection 1 DN: reflectance x 10000 (``Input()`` defaults).
DN = Units(reflectance_unit=10000.0)

#: Surface reflectance, 0..1: :data:`DN` divided by 10000.
REFLECTANCE = Units(reflectance_unit=1.0)

#: Thermal as Kelvin x 10, transformed to degrees Celsius x 100, where 100
#: degrees is 10000 (``Input()`` defaults).
KELVIN_TENTHS = Thermal(scale=10.0, offset=-27315.0, unit=10000.0)

#: Thermal already in degrees Celsius / 100, where 100 degrees is 1: the unit
#: that shares the reflectance scale factor of 10000 with the optical bands
#: (one Lasso alpha serves every band only if every band has the same scale).
CELSIUS_HUNDREDTHS = Thermal(scale=1.0, offset=0.0, unit=1.0)


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
    """One sensor's data: band names and roles, units and QA.

    The input says only how to read the data. The configuration says what
    to do with it, in physical units. The run translates once: 0–1
    reflectance × 10000 = the 0–10000 DN check, exactly as today.

    Example: ``ccdcpp.detect(dates, spectral, qas,
    input=ccdcpp.presets.landsat_c2())``

    name
        A label for the profile, for messages and provenance.
    band_names
        The rows of ``spectral``, in order.
    roles
        One role from :data:`ROLES` per band, in the same order. ``"green"``
        is required; ``"thermal"`` requires ``thermal=``.
    units
        The reflective bands' units, a :class:`Units`.
    thermal
        A :class:`Thermal`, or None for an input without a thermal band
        (no ``"thermal"`` role, which skips the transform and the thermal
        range).
    saturated
        ``Input.saturated``: the value a saturated observation holds, as the
        sensor writes it, or None (the default) when the profile does not
        declare one. A step or strategy that reads it refuses an input
        without one.
    qa_table_builder
        The name of a ``ccdcpp.qa_tables`` builder for the sensor's raw QA
        band, or None for pre-classified uint8 codes.
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
    units: Units
    thermal: Optional[Thermal] = None
    saturated: Optional[float] = None
    qa_table_builder: Optional[str] = None
    qa_table_options: QaTableOptions = field(default=())

    def __post_init__(self):
        for name in ("band_names", "roles"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.saturated is not None:
            object.__setattr__(self, "saturated", float(self.saturated))
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
        and shared (it is immutable; ``replace()`` gives a changed copy).

        Roles are the profile's; ``qa_table`` is the ``ccdcpp.QaTable`` of
        ``qa_table_builder`` with ``qa_table_options``, built with the
        input's codes, or None for pre-classified uint8 codes. Every field
        the profile does not describe keeps its ``Input()`` default.
        """
        return _input_of(self)

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
    h = Input(**dict(zip(_TABLE_CODE_FIELDS, codes)))
    return getattr(qa_tables, builder)(h, **dict(builder_options))


@lru_cache(maxsize=None)
def _input_of(profile):
    h = Input(
        roles=list(profile.roles),
        reflectance_unit=profile.units.reflectance_unit,
        reflectance_offset=profile.units.offset,
        saturated=profile.saturated,
    )
    if profile.thermal is not None:
        h = h.replace(thermal_scale=profile.thermal.scale,
                      thermal_offset=profile.thermal.offset,
                      thermal_unit=profile.thermal.unit)
    if profile.qa_table_builder is not None:
        codes = tuple(int(getattr(h, name)) for name in _TABLE_CODE_FIELDS)
        h = h.replace(qa_table=_build_table(profile.qa_table_builder, codes,
                                            profile.qa_table_options))
    return h

