from __future__ import annotations

import os as _os
import sys
from collections.abc import Mapping as _Mapping, Sequence as _Sequence
from typing import Any as _Any

from numpy.typing import ArrayLike as _ArrayLike

# The extension's names, but for the calls this module wraps (detect, detect_cube, plan_of).
from .ccdcpp import (
    CCD_MAX_BANDS,
    CHAIN_CAPACITY,
    NO_BAND,
    NO_STATISTICS_END,
    QA_TABLE_SIZE,
    RESULTS_VERSION,
    RESUME_FORMAT,
    ROLES,
    Affine,
    Algorithm,
    Chain,
    GreenFilterScope,
    Input,
    PreprocessPlan,
    ProcedurePlan,
    QaTable,
    Step,
    StepKind,
    __version__,
    chi2_quantile,
    resume_schema,
)

# `import ccdcpp.qa_tables`, as well as the attribute.
from .ccdcpp import qa_tables
sys.modules[__name__ + ".qa_tables"] = qa_tables

del sys

from . import ccdcpp as _extension
from . import presets
from .config import Config, config, shared_values, steps, strategies
from .config import _NAMED
from .config import config as _config
from .preprocess import (
    PROCEDURES,
    QA_CODE_NAMES,
    REFLECTIVE,
    AddQa,
    BandRange,
    Dedupe,
    GreenMedian,
    KeepQa,
    Preprocess,
    ProcedureChain,
    ThermalRange,
)
from .profiles import (
    CELSIUS_OVER_100,
    DN,
    KELVIN_TENTHS,
    QA_TABLE_OPTIONS,
    REFLECTANCE,
    SensorProfile,
    Thermal,
    Units,
)
from . import results


#------------------------------------------------------------------------------
# The two calls: detect() on one pixel, detect_cube() on a cube, and
# to_pixels() to read a cube's columns per pixel.
#
# Each takes the description of the data as encoding= (a ccdcpp.Input, or a
# SensorProfile that builds one) and the algorithm as config= (a
# ccdcpp.Config, or a preset's name, built once), and calls the extension
# function of the same name with the Input and the Config's compiled
# Algorithm.
#------------------------------------------------------------------------------

def _encoding(entry: str, encoding: object, spectral: _ArrayLike,
              band_names: _Sequence[str] | None) -> tuple[Input, SensorProfile | None]:
    """encoding= as the extension takes it: (the ccdcpp.Input, the profile or None)."""
    if isinstance(encoding, SensorProfile):
        encoding.check(spectral, band_names)
        return encoding.input(), encoding
    if isinstance(encoding, Input):
        if band_names is not None:
            raise TypeError(
                f"{entry}(): band_names= is checked against a profile's band names; "
                f"pass encoding= a ccdcpp.SensorProfile")
        return encoding, None
    raise TypeError(
        f"{entry}(): encoding is a ccdcpp.Input or a ccdcpp.SensorProfile, not "
        f"{type(encoding).__name__}")


def _configuration(entry: str, config: object) -> Config:
    """config= as a ccdcpp.Config: itself, or the preset of that name (built once)."""
    if isinstance(config, str):
        named = _NAMED.get(config)
        if named is not None:
            return named
        if config.endswith(".toml") or _os.sep in config:
            raise TypeError(
                f"{entry}(): config is a ccdcpp.Config or a preset's name, not a path; "
                f"pass ccdcpp.config({config!r}), built once")
        return _config(config)
    if isinstance(config, Config):
        return config
    raise TypeError(
        f"{entry}(): config is a ccdcpp.Config or a preset's name, not "
        f"{type(config).__name__}; ccdcpp.config() reads a file or a dict")


def detect(dates: _ArrayLike, spectral: _ArrayLike, qas: _ArrayLike, *,
           encoding: Input | SensorProfile, config: Config | str = "ccdc", record: bool = False,
           resume: _Mapping[str, _Any] | None = None, band_names: _Sequence[str] | None = None,
           trace: bool = False) -> results.PixelResult:
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
    encoding
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
        a misordered stack runs. Needs encoding= a profile (TypeError otherwise).
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
    h, profile = _encoding("detect", encoding, spectral, band_names)
    cfg = _configuration("detect", config)
    out = _extension.detect(dates, spectral, qas, encoding=h, config=cfg._algorithm,
                            record=record, resume=resume, trace=trace)
    if profile is not None:
        out["band_names"] = profile.band_names
    return out


def detect_cube(dates: _ArrayLike, spectral: _ArrayLike, qas: _ArrayLike, *,
                encoding: Input | SensorProfile, config: Config | str = "ccdc", threads: int = 0,
                record: bool = False, resume: _Mapping[str, _Any] | None = None,
                band_names: _Sequence[str] | None = None,
                trace_pixels: _Sequence[tuple[int, int]] | None = None,
                masks: bool = False) -> results.CubeColumns:
    """CCD on every pixel of a cube, in parallel, as columns.

    dates
        (T,) int64, shared by every pixel, non-decreasing (ValueError
        otherwise, checked once per call).
    spectral
        (H, W, B, T) float64, float32 or uint16, any non-negative strides.
    qas
        (H, W, T) uint8 QA codes, or raw uint8/uint16 QA when the input has a
        qa_table.
    encoding, config, band_names
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
        resume_header["digest"] and resume["input_fingerprint"] are
        uint64 (store them unsigned, or as int64 bits: both are read back).
    trace_pixels
        (row, col) pairs whose runs to return step by step, as detect()'s
        trace, in "traces": {(row, col): [event, ...]}.
    masks
        True to also return "processing_mask", every pixel's processing mask
        as detect() returns it, in one contiguous uint8 array (H, W, T):
        [r, c] is pixel (r, c)'s (zeros for a failed pixel). T bytes per
        pixel, kept while the cube runs.

    Returns a dict of arrays with one entry per segment: row, col,
    start_day, end_day, break_day, curve_qa, observation_count,
    change_probability, intercept [n, B], coefficients [n, B, 7], rmse
    [n, B], magnitude [n, B]; and num_bands, failures, threads, pixels_total
    and pixels_with_segments, plus band_names (a tuple) with a profile. A
    pixel with no segment has no row (pixels_total - pixels_with_segments
    counts them). ccdcpp.to_pixels() reads it per pixel.
    """
    h, profile = _encoding("detect_cube", encoding, spectral, band_names)
    cfg = _configuration("detect_cube", config)
    out = _extension.detect_cube(dates, spectral, qas, encoding=h, config=cfg._algorithm,
                                 threads=threads, record=record, resume=resume,
                                 trace_pixels=trace_pixels, masks=masks)
    if profile is not None:
        out["band_names"] = profile.band_names
    return out


def plan_of(config: Config | str, input: Input | SensorProfile) -> PreprocessPlan:
    """The ccdcpp.PreprocessPlan ``config`` (a ccdcpp.Config or a preset's name) runs on
    ``input`` (a ccdcpp.Input or a SensorProfile): its preprocess section resolved against the
    input, the QA code names as its codes, the bands by role as its rows, the bounds in its
    units. For inspection: ``print(ccdcpp.plan_of("ccdc", ccdcpp.Input()))``."""
    h = input.input() if isinstance(input, SensorProfile) else input
    return _extension.plan_of(_configuration("plan_of", config)._algorithm, h)


def to_pixels(out: results.CubeColumns) -> list[results.Pixel]:
    """detect_cube()'s columns as one dict per pixel with a segment, in raster
    order: {"row", "col", "models"}, the models as detect() returns them (each
    with a bands list in row order, every value the column's), and with
    masks=True its "processing_mask" (its row of the cube's array). A pixel
    with no segment has no entry, as it has no row in the columns."""
    pixels: list[results.Pixel] = []
    n = len(out["row"])
    B = int(out["num_bands"])
    masks = out.get("processing_mask")
    i = 0
    while i < n:
        row, col = int(out["row"][i]), int(out["col"][i])
        models: list[results.Model] = []
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
        pixel: results.Pixel = {"row": row, "col": col, "models": models}
        if masks is not None:
            pixel["processing_mask"] = masks[row, col]
        pixels.append(pixel)
    return pixels
