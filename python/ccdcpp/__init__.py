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
# The entry points, with profile=.
#
# Each forwards to the extension function of the same name
# (ccdcpp.ccdcpp.detect, ...). Without profile= the arguments reach it as
# given, so a call is the extension call. input is optional only together
# with a profile, and band_names is accepted only with one. A configuration
# reaches the extension as a Config, whose compiled Algorithm it runs.
#------------------------------------------------------------------------------

class _FromProfile:
    """The qa_table and preprocess default: the profile's, or None without
    one."""

    def __repr__(self):
        return "<from profile>"

    def __reduce__(self):
        return "_FROM_PROFILE"


_FROM_PROFILE = _FromProfile()


def _plan(entry, preprocess, profile, input, cfg):
    """preprocess= as the extension takes it: a PreprocessPlan or None."""
    if preprocess is _FROM_PROFILE:
        if profile is None or profile.preprocess is None:
            return None
        return profile.preprocess.plan(profile, input, cfg)
    if isinstance(preprocess, Preprocess):
        if profile is None:
            raise TypeError(
                f"{entry}(): a ccdcpp.Preprocess names roles and needs profile=; "
                f"pass a ccdcpp.PreprocessPlan without one"
            )
        return preprocess.plan(profile, input, cfg)
    return preprocess


def _resolve(entry, spectral, input, qa_table, preprocess, profile, band_names, config):
    """The arguments as the extension takes them: (input, qa_table, plan, the
    Config)."""
    if profile is None:
        if band_names is not None:
            raise TypeError(
                f"{entry}(): band_names= is checked against a profile's band "
                f"names and needs profile="
            )
        if input is None:
            raise TypeError(f"{entry}() needs input, or profile=")
        cfg = _config(config if config is not None else "ccdc")
        return (input, (None if qa_table is _FROM_PROFILE else qa_table),
                _plan(entry, preprocess, None, input, cfg), cfg)

    if not isinstance(profile, SensorProfile):
        raise TypeError(
            f"{entry}(): profile must be a ccdcpp.SensorProfile, not "
            f"{type(profile).__name__}"
        )
    profile.check(spectral, band_names)
    if input is None:
        input = profile.input()
    if qa_table is _FROM_PROFILE:
        qa_table = profile.qa_table(input)
    cfg = profile.config() if config is None else _config(config)
    return input, qa_table, _plan(entry, preprocess, profile, input, cfg), cfg


def detect(dates, spectral, qas, input=None, *,
           qa_table=_FROM_PROFILE, preprocess=_FROM_PROFILE, profile=None,
           band_names=None, resume_record=False, config=None, trace=False):
    """CCD on one pixel.

    dates
        (T,) int64 proleptic Gregorian ordinals, non-decreasing (ValueError
        otherwise). Equal dates are allowed; the first observation of each
        date is kept.
    spectral
        (B, T) float64, one row per band.
    qas
        (T,) uint8 QA codes, or raw uint8/uint16 QA with qa_table.
    input
        A ccdcpp.Input, the description of the data. Required unless
        profile is given.
    qa_table
        A ccdcpp.QaTable that decodes raw qas, or None for codes.
    preprocess
        A ccdcpp.PreprocessPlan, or None for ccdcpp.default_plan(input, config):
        per procedure, the transform and filter steps that build the
        processing mask. With a profile, a ccdcpp.Preprocess is accepted
        too. Checked against the band count first (ValueError).
    profile
        A ccdcpp.SensorProfile (see ccdcpp.presets) that supplies input
        (profile.input()), qa_table, preprocess and the configuration
        (profile.config()). Any of them passed explicitly is
        used instead, whole; qa_table=None for pre-classified codes,
        preprocess=None for the default plan. The table and the plan are
        built with the input and configuration the run uses. spectral is checked against
        the profile's band count first (ValueError).
    band_names
        The names of spectral's rows, in the caller's order (a store's band
        list, say), or None. Given, they must be the profile's band_names
        in the profile's order: a stack in another order raises ValueError
        before anything runs, where without it only the count is checked
        and a misordered stack runs. Needs profile= (TypeError otherwise).
    config
        A ccdcpp.Config: the algorithm, the shared values and one strategy
        and its parameters per stage (ccdcpp.config(), ccdcpp.strategies()),
        in physical terms; or any ccdcpp.config() source (a preset's name, a
        TOML file's path, a dict). None, the default, runs profile.config()
        with a profile and the "ccdc" preset without one. Checked before the pixel runs (ValueError); a resume
        record of a configured run continues only under the same one.
    trace
        True to also return "trace": the pixel's run step by step, one dict
        per event (kind: transition, test, drop, segment or exit; phase and
        next, the window, prev_break, observations, day, day_to, outcome,
        values), ready for pandas.DataFrame. Empty in a build with
        CCD_OBSERVE=OFF.
    resume_record
        True to also return the run's resume record: "resume", the pixel's
        record (a dict; see resume_schema()), and "resume_header", the
        call's. What resume() takes back to continue the run on a longer
        series.

    Returns a dict: models (one dict per segment, each with a bands list of
    B entries in row order), processing_mask and num_bands, plus band_names
    (a tuple) with a profile.
    """
    input, qa_table, preprocess, cfg = _resolve(
        "detect", spectral, input, qa_table, preprocess, profile, band_names,
        config)
    out = _extension.detect(
        dates, spectral, qas, input, qa_table=qa_table,
        preprocess=preprocess, resume_record=resume_record, config=cfg, trace=trace)
    if profile is not None:
        out["band_names"] = profile.band_names
    return out


def detect_cube(dates, spectral, qas, input=None, threads=0,
                *, qa_table=_FROM_PROFILE, preprocess=_FROM_PROFILE, profile=None,
                band_names=None, config=None):
    """CCD on every pixel of a cube, in parallel.

    dates
        (T,) int64, shared by every pixel, non-decreasing (ValueError
        otherwise, checked once per call).
    spectral
        (H, W, B, T) float64, float32 or uint16, any non-negative strides.
    qas
        (H, W, T) uint8 QA codes, or raw uint8/uint16 QA with qa_table.
    input
        A ccdcpp.Input, the description of the data. Required unless
        profile is given.
    threads
        OpenMP threads; 0 for the library's default.
    qa_table
        A ccdcpp.QaTable that decodes raw qas, or None for codes.
    preprocess
        A ccdcpp.PreprocessPlan, or None for ccdcpp.default_plan(input, config):
        per procedure, the transform and filter steps that build the
        processing mask. With a profile, a ccdcpp.Preprocess is accepted
        too. Checked against the band count first (ValueError).
    profile
        A ccdcpp.SensorProfile (see ccdcpp.presets) that supplies input
        (profile.input()), qa_table, preprocess and the configuration
        (profile.config()). Any of them passed explicitly is
        used instead, whole; qa_table=None for pre-classified codes,
        preprocess=None for the default plan. The table and the plan are
        built with the input and configuration the run uses. spectral is checked against
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
    input, qa_table, preprocess, cfg = _resolve(
        "detect_cube", spectral, input, qa_table, preprocess, profile, band_names,
        config)
    return _extension.detect_cube(
        dates, spectral, qas, input, threads, qa_table=qa_table,
        preprocess=preprocess, config=cfg)


def detect_cube_columns(dates, spectral, qas, input=None, threads=0, *, qa_table=_FROM_PROFILE, preprocess=_FROM_PROFILE,
                        profile=None, band_names=None, resume_record=False,
                        config=None, trace_pixels=None):
    """CCD on every pixel of a cube, in parallel, as columns.

    Arguments as detect_cube. Returns a dict of arrays with one entry per
    segment: row, col, start_day, end_day, break_day, curve_qa,
    observation_count, change_probability, intercept [n, B], coefficients
    [n, B, 7], rmse [n, B], magnitude [n, B]; and num_bands, failures,
    threads, pixels_total and pixels_with_segments, plus band_names (a
    tuple) with a profile.

    resume_record=True also returns "resume", every pixel's resume record
    as columns over the H x W pixels in raster order (see resume_schema();
    each pixel's excluded days are excluded_days[excluded_offsets[i]:
    excluded_offsets[i + 1]]), and "resume_header", the call's: what
    resume_cube_columns() takes back.
    """
    input, qa_table, preprocess, cfg = _resolve(
        "detect_cube_columns", spectral, input, qa_table, preprocess, profile, band_names,
        config)
    out = _extension.detect_cube_columns(
        dates, spectral, qas, input, threads, qa_table=qa_table,
        preprocess=preprocess, resume_record=resume_record, config=cfg,
        trace_pixels=trace_pixels)
    if profile is not None:
        out["band_names"] = profile.band_names
    return out


def resume(dates, spectral, qas, input=None, *, resume,
           resume_header, models, qa_table=_FROM_PROFILE, preprocess=_FROM_PROFILE,
           profile=None, band_names=None, config=None):
    """Continue a recorded run on one pixel's longer series.

    dates, spectral, qas
        The whole series: the stored run's input with the observations after
        its series end (ValueError if it ends sooner).
    resume, resume_header
        detect(resume_record=True)'s "resume" and "resume_header" (or an
        earlier resume's).
    models
        The stored run's segments, as detect() returned them; the first
        resume["final_segments"] are kept as given.
    input, qa_table, preprocess, profile, band_names, config
        As detect(), and the same as the stored run's: another input,
        configuration, plan or QA table raise ValueError, as do another record format or results
        version.

    Returns detect()'s dict for the whole series, equal bit for bit to
    detect() with every statistics cutoff capped at the first run's series
    end, plus "resume" and "resume_header" for the next resume and
    "status": "resumed", or "rerun_input_changed" when the pixel's
    observations up to the stored series end differ from the stored run's
    and it was rerun in full.
    """
    input, qa_table, preprocess, cfg = _resolve(
        "resume", spectral, input, qa_table, preprocess, profile, band_names,
        config)
    out = _extension.resume(
        dates, spectral, qas, input, resume=resume,
        resume_header=resume_header, models=models, qa_table=qa_table,
        preprocess=preprocess, config=cfg)
    if profile is not None:
        out["band_names"] = profile.band_names
    return out


def resume_cube_columns(dates, spectral, qas, input=None,
                        threads=0, *, resume, resume_header, segments,
                        qa_table=_FROM_PROFILE, preprocess=_FROM_PROFILE,
                        profile=None, band_names=None, config=None):
    """Continue a recorded cube run on its longer series, as columns.

    dates, spectral, qas
        The whole series of the same cube (H, W): the stored run's input
        with the observations after its series end.
    resume, resume_header
        detect_cube_columns(resume_record=True)'s "resume" and
        "resume_header" (or an earlier resume's).
    segments
        The stored run's segment columns, as detect_cube_columns() returned
        them (row, col, start_day, end_day, break_day, curve_qa,
        observation_count, change_probability, intercept, coefficients,
        rmse, magnitude); each pixel's first final_segments are kept as
        given.
    The other arguments as detect_cube_columns, and the same as the stored
    run's (ValueError otherwise).

    Returns detect_cube_columns()'s dict for the whole series, each pixel
    equal bit for bit to resume() on it, plus "resume" (with a "status"
    column: 0 resumed, 1 rerun because its input changed, 2 failed),
    "resume_header" and "counts" (resumed, rerun_input_changed, failed). A
    pixel that fails is in "failures"; its record is a default one, so the
    next resume reruns it.

    The record columns must cover the cube in raster order, each of its
    length and shape, or the call raises ValueError; a record that does not
    fit its pixel's series (a cursor past it) is that pixel's failure.
    resume_header["options_digest"] and resume["input_fingerprint"] are
    uint64 (store them unsigned, or as int64 bits: both are read back).
    """
    input, qa_table, preprocess, cfg = _resolve(
        "resume_cube_columns", spectral, input, qa_table, preprocess, profile, band_names,
        config)
    out = _extension.resume_cube_columns(
        dates, spectral, qas, input, threads, resume=resume,
        resume_header=resume_header, segments=segments, qa_table=qa_table,
        preprocess=preprocess, config=cfg)
    if profile is not None:
        out["band_names"] = profile.band_names
    return out
