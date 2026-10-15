import sys

from .ccdcpp import *

# A star import skips underscore names.
from .ccdcpp import __version__

# `import ccdcpp.qa_tables`, as well as the attribute.
from .ccdcpp import qa_tables
sys.modules[__name__ + ".qa_tables"] = qa_tables

del sys

from . import ccdcpp as _extension
from . import presets
from .profiles import (
    CADENCE_FIELDS,
    CELSIUS_HUNDREDTHS,
    DEFAULT_TRANSFORM,
    DN,
    GREEN_FILTER_SCOPES,
    KELVIN_TENTHS,
    NO_STAT_CUTOFF,
    OUTLIER_PROBABILITY,
    PROCEDURES,
    QA_CODE_FIELDS,
    QA_TABLE_OPTIONS,
    REFLECTANCE,
    ROLES,
    AddQa,
    BandRange,
    Dedupe,
    GreenMedian,
    KeepQa,
    Preprocess,
    ProcedureChain,
    SensorProfile,
    Thermal,
    ThermalRange,
    Transform,
    Units,
)


#------------------------------------------------------------------------------
# The entry points, with profile=.
#
# Each forwards to the extension function of the same name
# (ccdcpp.ccdcpp.detect, ...). Without profile= the arguments reach it as
# given, so a call is the extension call. hoptions and loptions are optional
# only together with a profile, and band_names is accepted only with one.
#------------------------------------------------------------------------------

class _FromProfile:
    """The qa_table and preprocess default: the profile's, or None without
    one."""

    def __repr__(self):
        return "<from profile>"

    def __reduce__(self):
        return "_FROM_PROFILE"


_FROM_PROFILE = _FromProfile()


def _plan(entry, preprocess, profile, hoptions):
    """preprocess= as the extension takes it: a PreprocessPlan or None."""
    if preprocess is _FROM_PROFILE:
        if profile is None or profile.preprocess is None:
            return None
        return profile.preprocess.plan(profile, hoptions)
    if isinstance(preprocess, Preprocess):
        if profile is None:
            raise TypeError(
                f"{entry}(): a ccdcpp.Preprocess names roles and needs profile=; "
                f"pass a ccdcpp.PreprocessPlan without one"
            )
        return preprocess.plan(profile, hoptions)
    return preprocess


def _resolve(entry, spectral, hoptions, loptions, qa_table, preprocess, profile,
             band_names=None):
    if profile is None:
        if band_names is not None:
            raise TypeError(
                f"{entry}(): band_names= is checked against a profile's band "
                f"names and needs profile="
            )
        if hoptions is None or loptions is None:
            raise TypeError(
                f"{entry}() needs hoptions and loptions, or profile="
            )
        return (hoptions, loptions, (None if qa_table is _FROM_PROFILE else qa_table),
                _plan(entry, preprocess, None, hoptions))

    if not isinstance(profile, SensorProfile):
        raise TypeError(
            f"{entry}(): profile must be a ccdcpp.SensorProfile, not "
            f"{type(profile).__name__}"
        )
    profile.check(spectral, band_names)
    if hoptions is None or loptions is None:
        h, lo = profile.options()
        hoptions = h if hoptions is None else hoptions
        loptions = lo if loptions is None else loptions
    if qa_table is _FROM_PROFILE:
        qa_table = profile.qa_table(hoptions)
    return hoptions, loptions, qa_table, _plan(entry, preprocess, profile, hoptions)


def detect(dates, spectral, qas, hoptions=None, loptions=None, *,
           qa_table=_FROM_PROFILE, preprocess=_FROM_PROFILE, profile=None,
           band_names=None):
    """CCD on one pixel.

    dates
        (T,) int64 proleptic Gregorian ordinals, non-decreasing (ValueError
        otherwise). Equal dates are allowed; the first observation of each
        date is kept.
    spectral
        (B, T) float64, one row per band.
    qas
        (T,) uint8 QA codes, or raw uint8/uint16 QA with qa_table.
    hoptions, loptions
        ccdcpp.HarmonicOptions and ccdcpp.LassoOptions. Required unless
        profile is given.
    qa_table
        A ccdcpp.QaTable that decodes raw qas, or None for codes.
    preprocess
        A ccdcpp.PreprocessPlan, or None for ccdcpp.default_plan(hoptions):
        per procedure, the transform and filter steps that build the
        processing mask. With a profile, a ccdcpp.Preprocess is accepted
        too. Checked against the band count first (ValueError).
    profile
        A ccdcpp.SensorProfile (see ccdcpp.presets) that supplies hoptions,
        loptions, qa_table and preprocess. Any of them passed explicitly is
        used instead, whole; qa_table=None for pre-classified codes,
        preprocess=None for the default plan. The table and the plan are
        built with the hoptions the run uses. spectral is checked against
        the profile's band count first (ValueError).
    band_names
        The names of spectral's rows, in the caller's order (a store's band
        list, say), or None. Given, they must be the profile's band_names
        in the profile's order: a stack in another order raises ValueError
        before anything runs, where without it only the count is checked
        and a misordered stack runs. Needs profile= (TypeError otherwise).

    Returns a dict: models (one dict per segment, each with a bands list of
    B entries in row order), processing_mask and num_bands, plus band_names
    (a tuple) with a profile.
    """
    hoptions, loptions, qa_table, preprocess = _resolve(
        "detect", spectral, hoptions, loptions, qa_table, preprocess, profile,
        band_names)
    out = _extension.detect(
        dates, spectral, qas, hoptions, loptions, qa_table=qa_table,
        preprocess=preprocess)
    if profile is not None:
        out["band_names"] = profile.band_names
    return out


def detect_cube(dates, spectral, qas, hoptions=None, loptions=None, threads=0,
                *, qa_table=_FROM_PROFILE, preprocess=_FROM_PROFILE, profile=None,
                band_names=None):
    """CCD on every pixel of a cube, in parallel.

    dates
        (T,) int64, shared by every pixel, non-decreasing (ValueError
        otherwise, checked once per call).
    spectral
        (H, W, B, T) float64, float32 or uint16, any non-negative strides.
    qas
        (H, W, T) uint8 QA codes, or raw uint8/uint16 QA with qa_table.
    hoptions, loptions
        ccdcpp.HarmonicOptions and ccdcpp.LassoOptions. Required unless
        profile is given.
    threads
        OpenMP threads; 0 for the library's default.
    qa_table
        A ccdcpp.QaTable that decodes raw qas, or None for codes.
    preprocess
        A ccdcpp.PreprocessPlan, or None for ccdcpp.default_plan(hoptions):
        per procedure, the transform and filter steps that build the
        processing mask. With a profile, a ccdcpp.Preprocess is accepted
        too. Checked against the band count first (ValueError).
    profile
        A ccdcpp.SensorProfile (see ccdcpp.presets) that supplies hoptions,
        loptions, qa_table and preprocess. Any of them passed explicitly is
        used instead, whole; qa_table=None for pre-classified codes,
        preprocess=None for the default plan. The table and the plan are
        built with the hoptions the run uses. spectral is checked against
        the profile's band count first (ValueError).
    band_names
        The names of spectral's rows, in the caller's order (a store's band
        list, say), or None. Given, they must be the profile's band_names
        in the profile's order: a stack in another order raises ValueError
        before anything runs, where without it only the count is checked
        and a misordered stack runs. Needs profile= (TypeError otherwise).

    Returns a list of per-pixel dicts in raster order: row, col, models and
    processing_mask, as detect's. They carry no band_names.
    """
    hoptions, loptions, qa_table, preprocess = _resolve(
        "detect_cube", spectral, hoptions, loptions, qa_table, preprocess, profile,
        band_names)
    return _extension.detect_cube(
        dates, spectral, qas, hoptions, loptions, threads, qa_table=qa_table,
        preprocess=preprocess)


def detect_cube_columns(dates, spectral, qas, hoptions=None, loptions=None,
                        threads=0, *, qa_table=_FROM_PROFILE, preprocess=_FROM_PROFILE,
                        profile=None, band_names=None):
    """CCD on every pixel of a cube, in parallel, as columns.

    Arguments as detect_cube. Returns a dict of arrays with one entry per
    segment: row, col, start_day, end_day, break_day, curve_qa,
    observation_count, change_probability, intercept [n, B], coefficients
    [n, B, 7], rmse [n, B], magnitude [n, B]; and num_bands, failures,
    threads, pixels_total and pixels_with_segments, plus band_names (a
    tuple) with a profile.
    """
    hoptions, loptions, qa_table, preprocess = _resolve(
        "detect_cube_columns", spectral, hoptions, loptions, qa_table, preprocess,
        profile, band_names)
    out = _extension.detect_cube_columns(
        dates, spectral, qas, hoptions, loptions, threads, qa_table=qa_table,
        preprocess=preprocess)
    if profile is not None:
        out["band_names"] = profile.band_names
    return out
