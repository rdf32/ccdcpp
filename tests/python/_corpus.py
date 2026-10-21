"""The inputs the Python tests share, and the runs they compare.

Import it after _harness.use_package(): it builds its inputs with the package.

  * The default Input's QA codes, and START.
  * pyccd's reference pixel (tests/ccd/data) and a corpus cube on its dates, one pixel per
    procedure, in DN (CUBE_DN) and in the physical presets' units (CUBE_PHYS), with its
    QA codes (CODES).
  * PYCCD: the "ccdc" preset with pyccd's statistics cutoff.
  * A ten-band profile without thermal, not shipped, with its configuration and series.
  * every_entry(): detect() on every pixel of a cube and detect_cube() on it, and the
    helpers that strip what two such runs may differ in.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

import ccdcpp

from _harness import SOURCE_DIR

START = 724276  # 1984-01-01
SCALE = 10000.0
FILL, CLEAR, WATER, SHADOW, SNOW, CLOUD = 255, 0, 1, 2, 3, 4


Series = tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.uint8]]


def reference_pixel() -> Series:
    """pyccd's reference pixel: (dates, spectral (7, T) DN, qas)."""
    path = os.path.join(SOURCE_DIR, "tests", "ccd", "data", "test_3657_3610_observations.csv")
    data = np.loadtxt(path, delimiter=",", dtype=np.int64)
    return (data[:, 0].copy(), np.ascontiguousarray(data[:, 1:8].T, dtype=np.float64),
            data[:, 8].astype(np.uint8))


def corpus_cube(H: int = 2, W: int = 3, seed: int = 11) -> Series:
    """A (H, W, 7, T) DN cube on the reference pixel's dates: the reference
    pixel itself, a snow-dominated pixel (PermanentSnow), a cloud-dominated
    one (InsufficientClear), and clear pixels with a break (Standard).

    Returns (dates, spectral, codes), codes free of FILL-only rows and of
    code 8, so each has a QA_PIXEL input."""
    dates, ref, ref_qa = reference_pixel()
    T = dates.size
    rng = np.random.default_rng(seed)
    phase = 2 * np.pi * (dates - START) / 365.25
    spectral = np.empty((H, W, 7, T))
    codes = np.empty((H, W, T), dtype=np.uint8)
    for p in range(H * W):
        r, c = divmod(p, W)
        if p == 0:
            spectral[r, c], codes[r, c] = ref, ref_qa
            continue
        base = 600 + 300 * np.arange(7, dtype=float)[:, None]
        season = (150 + 20 * p) * np.sin(phase + p)[None, :]
        level = np.where(np.arange(T) > T // 2 + 7 * p, 900.0, 0.0)[None, :]
        s = base + season + level + rng.normal(0, 40, (7, T))
        s[6] = 2900 + 80 * np.sin(phase) + rng.normal(0, 10, T)
        spectral[r, c] = np.round(s)
        probs = {1: [0.02, 0.05, 0.01, 0.05, 0.85, 0.02],
                 2: [0.02, 0.17, 0.03, 0.05, 0.02, 0.71]}.get(
                     p, [0.04, 0.74, 0.06, 0.05, 0.03, 0.08])
        codes[r, c] = rng.choice(
            np.array([FILL, CLEAR, WATER, SHADOW, SNOW, CLOUD], dtype=np.uint8), size=T, p=probs)
    codes[codes == 8] = SHADOW
    return dates, spectral, codes


def to_physical(spectral: NDArray[np.float64]) -> NDArray[np.float64]:
    """DN -> the physical preset's units: reflectance, degrees Celsius / 100."""
    s = spectral.copy()
    s[..., :6, :] = s[..., :6, :] / SCALE
    s[..., 6, :] = (s[..., 6, :] * 10.0 - 27315.0) / SCALE
    return s


DATES, CUBE_DN, CODES = corpus_cube()
CUBE_PHYS = to_physical(CUBE_DN)


#: The "ccdc" preset with pyccd's statistics cutoff, 2017-12-31: what the Landsat profiles run.
PYCCD = ccdcpp.config("ccdc").replace(shared__statistics_end=736694)


def every_entry(dates: NDArray[np.int64], spectral: NDArray[Any], qas: NDArray[Any],
                threads: int = 8, **kw: Any) -> dict[str, Any]:
    """{entry: output} for detect on every pixel and detect_cube (its columns,
    without the thread count)."""
    H, W = spectral.shape[:2]
    return {
        "detect": [ccdcpp.detect(dates, spectral[r, c], qas[r, c], **kw)
                   for r in range(H) for c in range(W)],
        "detect_cube": without(
            ccdcpp.detect_cube(dates, spectral, qas, threads=threads, **kw), "threads"),
    }


def without(out: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    kept = dict(out)
    for k in keys:
        kept.pop(k)
    return kept


def strip_band_names(outs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "detect": [without(o, "band_names") for o in outs["detect"]],
        "detect_cube": without(outs["detect_cube"], "band_names"),
    }


#------------------------------------------------------------------------------
# A ten-band, thermal-less, non-Landsat profile, defined here and not shipped:
# a Sentinel-2-like layout in reflectance with SCL QA.
#------------------------------------------------------------------------------

def ten_band_profile() -> ccdcpp.SensorProfile:
    return ccdcpp.SensorProfile(
        name="test_ten_band",
        band_names=("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"),
        roles=("blue", "green", "red", "red_edge_1", "red_edge_2", "red_edge_3", "nir_broad",
               "nir", "swir1", "swir2"),
        units=ccdcpp.REFLECTANCE,
        qa_table_builder="sentinel2_scl",
    )


#: The configuration the ten-band tests run: six detection bands by role, a
#: 10-day reference cadence.
TEN_BAND_CONFIG = ccdcpp.config("ccdc").replace(
    change_test__peek_reference_days=10.0,
    shared__detection_bands=["green", "red", "red_edge_2", "nir", "swir1", "swir2"],
)


#: Raw SCL values: 4 vegetation, 5 bare (clear); 6 water; 8, 9 cloud; 3 shadow;
#: 0 no data.
SCL_VALUES = np.array([4, 5, 6, 8, 9, 3, 0], dtype=np.uint8)


def ten_band_inputs(H: int = 2, W: int = 2, seed: int = 5) -> Series:
    """(dates, spectral (H, W, 10, T) reflectance, raw SCL (H, W, T)): a 5-day
    series from 2018, after the Landsat statistics_end, with a break per pixel."""
    rng = np.random.default_rng(seed)
    dates = (736695 + 5 * np.arange(4 * 73)).astype(np.int64)
    T = dates.size
    phase = 2 * np.pi * (dates - dates[0]) / 365.25
    spectral = np.empty((H, W, 10, T))
    for p in range(H * W):
        r, c = divmod(p, W)
        base = (0.04 + 0.03 * np.arange(10))[:, None]
        season = (0.02 + 0.005 * p) * np.sin(phase + p)[None, :]
        level = np.where(np.arange(T) > T // 2 + 9 * p, 0.08, 0.0)[None, :]
        spectral[r, c] = base + season + level + rng.normal(0, 0.004, (10, T))
    scl = rng.choice(SCL_VALUES, size=(H, W, T), p=[0.5, 0.2, 0.05, 0.1, 0.05, 0.05, 0.05])
    return dates, spectral, scl
