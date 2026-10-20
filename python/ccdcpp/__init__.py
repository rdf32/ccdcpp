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
from .config import Config, config, shared_values, strategies
from .config import config as _config
from .ccdcpp import Input
from .profiles import (
    CELSIUS_HUNDREDTHS,
    DEFAULT_TRANSFORM,
    DN,
    GREEN_FILTER_SCOPES,
    KELVIN_TENTHS,
    NO_STAT_CUTOFF,
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
# The two calls: detect() on one pixel, detect_cube() on a cube, and
# to_pixels() to read a cube's columns per pixel.
#
# Each takes the description of the data as input= (a ccdcpp.Input, or a
# SensorProfile that builds one) and the algorithm as config= (a
# ccdcpp.Config, or a preset's name, built once), and calls the extension
# function of the same name with the Input and the Config's compiled
# Algorithm.
#------------------------------------------------------------------------------

def _input(entry, input, spectral, band_names):
    """input= as the extension takes it: (the ccdcpp.Input, the profile or None)."""
    if isinstance(input, SensorProfile):
        input.check(spectral, band_names)
        return input.input(), input
    if isinstance(input, Input):
        if band_names is not None:
            raise TypeError(
                f"{entry}(): band_names= is checked against a profile's band names; "
                f"pass input= a ccdcpp.SensorProfile")
        return input, None
    raise TypeError(
        f"{entry}(): input is a ccdcpp.Input or a ccdcpp.SensorProfile, not "
        f"{type(input).__name__}")


def _configuration(entry, config):
    """config= as a ccdcpp.Config: itself, or the preset of that name."""
    if isinstance(config, Config):
        return config
    if isinstance(config, str):
        return _config(config)
    raise TypeError(
        f"{entry}(): config is a ccdcpp.Config or a preset's name, not "
        f"{type(config).__name__}; ccdcpp.config() reads a file or a dict")


def _plan(profile, input, cfg):
    """The plan a profile's preprocess= gives, or None for the configuration's."""
    if profile is None or profile.preprocess is None:
        return None
    return profile.preprocess.plan(profile, input, cfg)


def detect(dates, spectral, qas, *, input, config="ccdc", record=False, resume=None,
           band_names=None, trace=False):
    """CCD on one pixel.

    dates
        (T,) int64 proleptic Gregorian ordinals, non-decreasing (ValueError
        otherwise). Equal dates are allowed; the first observation of each
        date is kept.
    spectral
        (B, T) float64, one row per band.
    qas
        (T,) uint8 QA codes, or raw uint8/uint16 QA when the input has a
        qa_table.
    input
        What the data is: a ccdcpp.Input, or a ccdcpp.SensorProfile (see
        ccdcpp.presets), whose input() is used. spectral is checked against a
        profile's band count first (ValueError).
    config
        The algorithm: a ccdcpp.Config (ccdcpp.config()), or a preset's name
        ("ccdc", the default, built once and shared). The same configuration
        means the same on every input. Checked when it is built (ValueError).
    record
        True to also return the run's resume record: "resume", the pixel's
        record (a dict; see resume_schema()), and "resume_header", the
        call's. What resume= takes back to continue the run on a longer
        series.
    resume
        An earlier detect() output made with record=True (or by a resume):
        continue that run on this longer series, the stored run's
        observations followed by new ones (ValueError if it ends sooner).
        Refused (ValueError) unless the input and the configuration are the
        earlier call's, as are another record format or results version. The
        result is detect()'s on the whole series, bit for bit, with every
        statistics cutoff capped at the first run's series end, plus "resume"
        and "resume_header" for the next resume and "status": "resumed", or
        "rerun_input_changed" when the pixel's observations up to the stored
        series end differ from the stored run's and it was rerun in full.
    band_names
        The names of spectral's rows, in the caller's order (a store's band
        list, say), or None. Given, they must be the profile's band_names in
        the profile's order: a stack in another order raises ValueError
        before anything runs, where without it only the count is checked and
        a misordered stack runs. Needs input= a profile (TypeError otherwise).
    trace
        True to also return "trace": the pixel's run step by step, one dict
        per event (kind: transition, test, drop, segment or exit; phase and
        next, the window, prev_break, observations, day, day_to, outcome,
        values), ready for pandas.DataFrame. Empty in a build with
        CCD_OBSERVE=OFF.

    Returns a dict: models (one dict per segment, each with a bands list of
    B entries in row order), processing_mask and num_bands, plus band_names
    (a tuple) with a profile.
    """
    h, profile = _input("detect", input, spectral, band_names)
    cfg = _configuration("detect", config)
    out = _extension.detect(dates, spectral, qas, input=h, config=cfg, record=record,
                            resume=resume, trace=trace, _plan=_plan(profile, h, cfg))
    if profile is not None:
        out["band_names"] = profile.band_names
    return out


def detect_cube(dates, spectral, qas, *, input, config="ccdc", threads=0, record=False,
                resume=None, band_names=None, trace_pixels=None):
    """CCD on every pixel of a cube, in parallel, as columns.

    dates
        (T,) int64, shared by every pixel, non-decreasing (ValueError
        otherwise, checked once per call).
    spectral
        (H, W, B, T) float64, float32 or uint16, any non-negative strides.
    qas
        (H, W, T) uint8 QA codes, or raw uint8/uint16 QA when the input has a
        qa_table.
    input, config, band_names
        As detect().
    threads
        OpenMP threads; 0 for the library's default.
    record
        True to also return "resume", every pixel's resume record as columns
        over the H x W pixels in raster order (see resume_schema(); each
        pixel's excluded days are excluded_days[excluded_offsets[i]:
        excluded_offsets[i + 1]]), and "resume_header", the call's.
    resume
        An earlier detect_cube() output made with record=True (or by a
        resume), of the same cube (H, W): continue it on this longer series,
        as detect()'s resume=, each pixel equal bit for bit to detect() on
        it. The output then also holds "resume" (with a "status" column: 0
        resumed, 1 rerun because its input changed, 2 failed),
        "resume_header" and "counts" (resumed, rerun_input_changed, failed).
        A pixel that fails is in "failures"; its record is a default one, so
        the next resume reruns it. The record columns must cover the cube in
        raster order, each of its length and shape (ValueError otherwise); a
        record that does not fit its pixel's series is that pixel's failure.
        resume_header["options_digest"] and resume["input_fingerprint"] are
        uint64 (store them unsigned, or as int64 bits: both are read back).
    trace_pixels
        (row, col) pairs whose runs to return step by step, as detect()'s
        trace, in "traces": {(row, col): [event, ...]}.

    Returns a dict of arrays with one entry per segment: row, col,
    start_day, end_day, break_day, curve_qa, observation_count,
    change_probability, intercept [n, B], coefficients [n, B, 7], rmse
    [n, B], magnitude [n, B]; and num_bands, failures, threads, pixels_total
    and pixels_with_segments, plus band_names (a tuple) with a profile. A
    pixel with no segment has no row (pixels_total - pixels_with_segments
    counts them). ccdcpp.to_pixels() reads it per pixel.
    """
    h, profile = _input("detect_cube", input, spectral, band_names)
    cfg = _configuration("detect_cube", config)
    out = _extension.detect_cube(dates, spectral, qas, input=h, config=cfg, threads=threads,
                                 record=record, resume=resume, trace_pixels=trace_pixels,
                                 _plan=_plan(profile, h, cfg))
    if profile is not None:
        out["band_names"] = profile.band_names
    return out


def to_pixels(out):
    """detect_cube()'s columns as one dict per pixel with a segment, in raster
    order: {"row", "col", "models"}, the models as detect() returns them (each
    with a bands list in row order, every value the column's). A pixel with no
    segment has no entry, as it has no row in the columns."""
    pixels = []
    n = len(out["row"])
    B = int(out["num_bands"])
    i = 0
    while i < n:
        row, col = int(out["row"][i]), int(out["col"][i])
        models = []
        while i < n and int(out["row"][i]) == row and int(out["col"][i]) == col:
            models.append({
                "start_day": int(out["start_day"][i]),
                "end_day": int(out["end_day"][i]),
                "break_day": int(out["break_day"][i]),
                "observation_count": int(out["observation_count"][i]),
                "change_probability": float(out["change_probability"][i]),
                "curve_qa": int(out["curve_qa"][i]),
                "bands": [{
                    "rmse": float(out["rmse"][i, b]),
                    "magnitude": float(out["magnitude"][i, b]),
                    "coefficients": tuple(float(c) for c in out["coefficients"][i, b]),
                    "intercept": float(out["intercept"][i, b]),
                } for b in range(B)],
            })
            i += 1
        pixels.append({"row": row, "col": col, "models": models})
    return pixels
