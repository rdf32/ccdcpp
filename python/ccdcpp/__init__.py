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
    DN,
    KELVIN_TENTHS,
    NO_STAT_CUTOFF,
    OUTLIER_PROBABILITY,
    REFLECTANCE,
    ROLES,
    SensorProfile,
    Thermal,
    Units,
)


#------------------------------------------------------------------------------
# The entry points, with profile=.
#
# Each forwards to the extension function of the same name
# (ccdcpp.ccdcpp.detect, ...). Without profile= the arguments reach it as
# given, so a call is the extension call. hoptions and loptions are optional
# only together with a profile.
#------------------------------------------------------------------------------

class _FromProfile:
    """The qa_table default: the profile's table, or None without one."""

    def __repr__(self):
        return "<from profile>"

    def __reduce__(self):
        return "_FROM_PROFILE"


_FROM_PROFILE = _FromProfile()


def _resolve(entry, spectral, hoptions, loptions, qa_table, profile):
    if profile is None:
        if hoptions is None or loptions is None:
            raise TypeError(
                f"{entry}() needs hoptions and loptions, or profile="
            )
        return hoptions, loptions, (None if qa_table is _FROM_PROFILE else qa_table)

    if not isinstance(profile, SensorProfile):
        raise TypeError(
            f"{entry}(): profile must be a ccdcpp.SensorProfile, not "
            f"{type(profile).__name__}"
        )
    profile.check(spectral)
    if hoptions is None or loptions is None:
        h, lo = profile.options()
        hoptions = h if hoptions is None else hoptions
        loptions = lo if loptions is None else loptions
    if qa_table is _FROM_PROFILE:
        qa_table = profile.qa_table(hoptions)
    return hoptions, loptions, qa_table


def detect(dates, spectral, qas, hoptions=None, loptions=None, *,
           qa_table=_FROM_PROFILE, profile=None):
    """CCD on one pixel.

    dates
        (T,) int64 proleptic Gregorian ordinals, ascending.
    spectral
        (B, T) float64, one row per band.
    qas
        (T,) uint8 QA codes, or raw uint8/uint16 QA with qa_table.
    hoptions, loptions
        ccdcpp.HarmonicOptions and ccdcpp.LassoOptions. Required unless
        profile is given.
    qa_table
        A ccdcpp.QaTable that decodes raw qas, or None for codes.
    profile
        A ccdcpp.SensorProfile (see ccdcpp.presets) that supplies hoptions,
        loptions and qa_table. Any of the three passed explicitly is used
        instead, whole; qa_table=None for pre-classified codes. The table is
        built with the QA codes of the hoptions the run uses. spectral is
        checked against the profile's band count first (ValueError).

    Returns a dict: models (one dict per segment, each with a bands list of
    B entries in row order), processing_mask and num_bands, plus band_names
    (a tuple) with a profile.
    """
    hoptions, loptions, qa_table = _resolve(
        "detect", spectral, hoptions, loptions, qa_table, profile)
    out = _extension.detect(
        dates, spectral, qas, hoptions, loptions, qa_table=qa_table)
    if profile is not None:
        out["band_names"] = profile.band_names
    return out


def detect_cube(dates, spectral, qas, hoptions=None, loptions=None, threads=0,
                *, qa_table=_FROM_PROFILE, profile=None):
    """CCD on every pixel of a cube, in parallel.

    dates
        (T,) int64, shared by every pixel.
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
    profile
        A ccdcpp.SensorProfile (see ccdcpp.presets) that supplies hoptions,
        loptions and qa_table. Any of the three passed explicitly is used
        instead, whole; qa_table=None for pre-classified codes. The table is
        built with the QA codes of the hoptions the run uses. spectral is
        checked against the profile's band count first (ValueError).

    Returns a list of per-pixel dicts in raster order: row, col, models and
    processing_mask, as detect's. They carry no band_names.
    """
    hoptions, loptions, qa_table = _resolve(
        "detect_cube", spectral, hoptions, loptions, qa_table, profile)
    return _extension.detect_cube(
        dates, spectral, qas, hoptions, loptions, threads, qa_table=qa_table)


def detect_cube_columns(dates, spectral, qas, hoptions=None, loptions=None,
                        threads=0, *, qa_table=_FROM_PROFILE, profile=None):
    """CCD on every pixel of a cube, in parallel, as columns.

    Arguments as detect_cube. Returns a dict of arrays with one entry per
    segment: row, col, start_day, end_day, break_day, curve_qa,
    observation_count, change_probability, intercept [n, B], coefficients
    [n, B, 7], rmse [n, B], magnitude [n, B]; and num_bands, failures,
    threads, pixels_total and pixels_with_segments, plus band_names (a
    tuple) with a profile.
    """
    hoptions, loptions, qa_table = _resolve(
        "detect_cube_columns", spectral, hoptions, loptions, qa_table, profile)
    out = _extension.detect_cube_columns(
        dates, spectral, qas, hoptions, loptions, threads, qa_table=qa_table)
    if profile is not None:
        out["band_names"] = profile.band_names
    return out
