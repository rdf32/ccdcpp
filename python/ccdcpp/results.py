"""The types of what ``ccdcpp.detect``, ``ccdcpp.detect_cube`` and ``ccdcpp.to_pixels``
return: dicts of a fixed shape, as ``TypedDict``s for type checkers. At run time each is a
plain ``dict``.

A key marked ``NotRequired`` is present only when the call asks for it: ``band_names`` with a
profile, ``resume`` and ``resume_header`` with ``record=True`` or ``resume=``, ``status`` and
``counts`` with ``resume=``, ``trace`` / ``traces`` with ``trace=True`` / ``trace_pixels=``, the
cube's ``processing_mask`` with ``masks=True``, ``mask_reasons`` with ``reasons=True``.
"""
from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

import numpy as np
from numpy.typing import NDArray

__all__ = ["BandFit", "CubeColumns", "CubeRecord", "Model", "Pixel", "PixelRecord",
           "PixelResult", "ResumeCounts", "ResumeHeader", "TraceEvent"]


class BandFit(TypedDict):
    """One band's fit of a segment."""

    rmse: float
    magnitude: float
    #: slope, cos1, sin1, cos2, sin2, cos3, sin3 (the intercept is separate)
    coefficients: tuple[float, ...]
    intercept: float


class Model(TypedDict):
    """One segment: its days, its observation count, its change probability and curve_qa,
    and one fit per band, in row order."""

    start_day: int
    end_day: int
    break_day: int
    observation_count: int
    change_probability: float
    curve_qa: int
    bands: list[BandFit]


class ResumeHeader(TypedDict):
    """A call's resume header (``ccdcpp.resume_schema()`` describes the record)."""

    format: int
    results_version: int
    #: uint64; store it unsigned, or as int64 bits
    digest: int
    series_end: int
    statistics_end: int


class PixelRecord(TypedDict):
    """One pixel's resume record, as detect() returns it."""

    input_fingerprint: int
    procedure: int
    exit: int
    window_start: int
    window_stop: int
    prev_break: int
    fit_stop: int
    grow: bool
    num_coef: int
    magnitude: NDArray[np.float64]
    final_segments: int
    excluded_days: NDArray[np.int64]


class CubeRecord(TypedDict):
    """Every pixel's resume record as columns over the cube's pixels in raster order; pixel
    i's excluded days are ``excluded_days[excluded_offsets[i]:excluded_offsets[i + 1]]``."""

    row: NDArray[np.int32]
    col: NDArray[np.int32]
    input_fingerprint: NDArray[np.uint64]
    procedure: NDArray[np.uint8]
    exit: NDArray[np.uint8]
    window_start: NDArray[np.int64]
    window_stop: NDArray[np.int64]
    prev_break: NDArray[np.int64]
    fit_stop: NDArray[np.int64]
    grow: NDArray[np.bool_]
    num_coef: NDArray[np.uint8]
    magnitude: NDArray[np.float64]
    final_segments: NDArray[np.int64]
    excluded_offsets: NDArray[np.int64]
    excluded_days: NDArray[np.int64]
    #: with resume=: 0 resumed, 1 rerun because its input changed, 2 failed
    status: NotRequired[NDArray[np.uint8]]


class ResumeCounts(TypedDict):
    """A cube resume's pixels by outcome."""

    resumed: int
    rerun_input_changed: int
    failed: int


class TraceEvent(TypedDict):
    """One event of a traced pixel's run."""

    kind: str
    machine: str
    phase: str
    next: str
    window_start: int
    window_stop: int
    prev_break: int
    observations: int
    day: int
    day_to: int
    outcome: str
    values: list[float]


class PixelResult(TypedDict):
    """``ccdcpp.detect()``'s result."""

    models: list[Model]
    #: (T,) uint8: 1 where the observation is kept
    processing_mask: NDArray[np.uint8]
    #: (T,) uint8, with reasons=True: each observation's code in ccdcpp.REASONS
    mask_reasons: NotRequired[NDArray[np.uint8]]
    num_bands: int
    band_names: NotRequired[tuple[str, ...]]
    resume: NotRequired[PixelRecord]
    resume_header: NotRequired[ResumeHeader]
    status: NotRequired[Literal["resumed", "rerun_input_changed"]]
    trace: NotRequired[list[TraceEvent]]


class CubeColumns(TypedDict):
    """``ccdcpp.detect_cube()``'s result: one row per (pixel, segment)."""

    row: NDArray[np.int32]
    col: NDArray[np.int32]
    start_day: NDArray[np.int64]
    end_day: NDArray[np.int64]
    break_day: NDArray[np.int64]
    curve_qa: NDArray[np.uint8]
    observation_count: NDArray[np.int32]
    change_probability: NDArray[np.float64]
    #: [n, B]
    intercept: NDArray[np.float64]
    #: [n, B, 7]
    coefficients: NDArray[np.float64]
    #: [n, B]
    rmse: NDArray[np.float64]
    #: [n, B]
    magnitude: NDArray[np.float64]
    num_bands: int
    #: (row, col, message) per pixel that raised
    failures: list[tuple[int, int, str]]
    threads: int
    pixels_total: int
    pixels_with_segments: int
    #: (H, W, T) uint8, with masks=True
    processing_mask: NotRequired[NDArray[np.uint8]]
    #: (H, W, T) uint8, with reasons=True
    mask_reasons: NotRequired[NDArray[np.uint8]]
    band_names: NotRequired[tuple[str, ...]]
    resume: NotRequired[CubeRecord]
    resume_header: NotRequired[ResumeHeader]
    counts: NotRequired[ResumeCounts]
    traces: NotRequired[dict[tuple[int, int], list[TraceEvent]]]


class Pixel(TypedDict):
    """One pixel of ``ccdcpp.to_pixels()``: its address and models, and with masks=True its
    processing mask, with reasons=True its reason codes."""

    row: int
    col: int
    models: list[Model]
    processing_mask: NotRequired[NDArray[np.uint8]]
    mask_reasons: NotRequired[NDArray[np.uint8]]
