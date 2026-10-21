"""The helpers the profile tests share (test_profiles_*.py): what a profile's input and
configuration resolve to, a profile's rows permuted and the outputs put back, and synthetic
HLS and Sentinel-2 series with each preset's input written out by hand.

Import it after _harness.use_package().
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

import numpy as np
from numpy.typing import NDArray

import ccdcpp
from ccdcpp import presets
from ccdcpp.results import Model, PixelResult

from _corpus import CLEAR, CLOUD, FILL, SHADOW, SNOW, WATER, Series, without


#: One raw QA_PIXEL value per class (Landsat Collection 2).
C2_RAW = {FILL: 1, CLEAR: 21824, WATER: 21952, SHADOW: 23888, SNOW: 30048, CLOUD: 22280}


def fields(obj: object) -> dict[str, Any]:
    return {
        name: getattr(obj, name) for name in dir(obj)
        if not name.startswith("_") and not callable(getattr(obj, name))
    }


def qa_table_of(input: ccdcpp.Input) -> ccdcpp.QaTable:
    """The QA table of an input that has one."""
    assert input.qa_table is not None
    return input.qa_table


def doc_of(obj: object) -> str:
    """The docstring of an object that has one."""
    assert obj.__doc__ is not None, obj
    return obj.__doc__


def resolved(cfg: ccdcpp.Config, input: ccdcpp.Input | None = None) -> dict[str, Any]:
    """``cfg`` resolved against ``input`` (default ccdcpp.Input())."""
    return cfg.resolve(ccdcpp.Input() if input is None else input)


def change_threshold(cfg: ccdcpp.Config, input: ccdcpp.Input | None = None) -> float:
    return resolved(cfg, input)["change_test"]["change_threshold"]


def outlier_threshold(cfg: ccdcpp.Config, input: ccdcpp.Input | None = None) -> float:
    return resolved(cfg, input)["outlier"]["threshold"]


# The parameters one shared value fills in, as a run resolves them.
GROUPS: dict[str, tuple[tuple[str, str], ...]] = {
    "alpha": (("model", "lasso.alpha"), ("stability_fit", "lasso.alpha"),
              ("edge_segments", "lasso.alpha")),
    "coef_floor": (("model", "lasso.coef_floor"), ("stability_fit", "lasso.coef_floor"),
                   ("edge_segments", "lasso.coef_floor")),
    "change_threshold": (("change_test", "change_threshold"),
                         ("stability_rule", "change_threshold"),
                         ("lookback", "change_threshold")),
    "outlier_threshold": (("outlier", "threshold"), ("lookback", "outlier_threshold")),
    "tmask": (("window_screen", "bands"),),
}


def option_params(cfg: ccdcpp.Config, name: str,
                  input: ccdcpp.Input | None = None) -> dict[tuple[str, str], Any]:
    """{(stage, parameter): value} of group ``name`` (GROUPS) in ``cfg`` resolved
    against ``input``, or of one ``stage__parameter``."""
    stage, _, param = name.partition("__")
    pairs = GROUPS.get(name) or ((stage, param),)
    stages = resolved(cfg, input)
    return {(stage, param): stages[stage][param]
            for stage, param in pairs
            if stage in stages and param in stages[stage]}


def band_range_rows(input: ccdcpp.Input | ccdcpp.SensorProfile,
                    config: ccdcpp.Config | str = "ccdc") -> list[int]:
    """The rows the Standard chain's band_range step tests on ``input``."""
    (step,) = [s for s in ccdcpp.plan_of(config, input).standard.chain
               if s.kind == ccdcpp.StepKind.BandRange]
    return step.bands


def with_table(input: ccdcpp.Input, qa_table: ccdcpp.QaTable | None) -> ccdcpp.Input:
    """``input`` (a ccdcpp.Input) decoding its QA through ``qa_table``."""
    return input.replace(qa_table=qa_table)


def permuted(profile: ccdcpp.SensorProfile, order: Sequence[int]) -> ccdcpp.SensorProfile:
    return dataclasses.replace(
        profile,
        band_names=tuple(profile.band_names[i] for i in order),
        roles=tuple(profile.roles[i] for i in order),
    )


def unpermute(outs: Mapping[str, Any], order: Sequence[int]) -> dict[str, Any]:
    """Outputs of a run on rows `order` -> the outputs of the run in row order."""
    inv = np.argsort(order)

    def models(ms: Sequence[Model]) -> list[dict[str, Any]]:
        return [dict(m, bands=[m["bands"][i] for i in inv]) for m in ms]

    cols = dict(outs["detect_cube"])
    for key in ("intercept", "coefficients", "rmse", "magnitude"):
        cols[key] = np.ascontiguousarray(cols[key][:, inv])
    cols["band_names"] = tuple(cols["band_names"][i] for i in inv)
    return {
        "detect": [dict(o, models=models(o["models"]),
                        band_names=tuple(o["band_names"][i] for i in inv))
                   for o in outs["detect"]],
        "detect_cube": cols,
    }


#------------------------------------------------------------------------------
# The HLS and Sentinel-2 presets, on synthetic series in each preset's units.
#------------------------------------------------------------------------------

#: Raw HLS v2.0 Fmask values the HLS User Guide (Table 9) defines, with their
#: decoded code when the aerosol bits are ignored: 0, 64, 128, 192 clear at
#: each aerosol level; 32 water; 2 cloud; 4 adjacent to cloud/shadow; 8
#: shadow; 16 snow; 100 water + adjacent + low aerosol (the guide's example,
#: cloud by precedence); 255 fill.
FMASK_VALUES = np.array([0, 64, 128, 192, 32, 2, 4, 8, 16, 100, 255], dtype=np.uint8)


FMASK_CODES = np.array([CLEAR, CLEAR, CLEAR, CLEAR, WATER, CLOUD, CLOUD, SHADOW, SNOW, CLOUD,
                        FILL], dtype=np.uint8)


FMASK_P = [0.45, 0.15, 0.08, 0.04, 0.03, 0.08, 0.05, 0.04, 0.02, 0.02, 0.04]


#: Raw Sentinel-2 L2A SCL classes and their decoded codes: 4 vegetation, 5
#: not vegetated (clear); 6 water; 7 unclassified, 8, 9 cloud, 10 cirrus; 2
#: cast shadows, 3 cloud shadows; 11 snow; 0 no data, 1 saturated.
SCL_ALL = np.array([4, 5, 6, 7, 8, 9, 10, 2, 3, 11, 0, 1], dtype=np.uint8)


SCL_CODES = np.array([CLEAR, CLEAR, WATER, CLOUD, CLOUD, CLOUD, CLOUD, SHADOW, SHADOW, SNOW,
                      FILL, FILL], dtype=np.uint8)


SCL_P = [0.45, 0.17, 0.03, 0.05, 0.08, 0.06, 0.04, 0.02, 0.03, 0.02, 0.03, 0.02]


L30_START = date(2013, 4, 11).toordinal()


S2_START = date(2015, 7, 4).toordinal()


S2_OFFSET_DATE = date(2022, 1, 25).toordinal()  # processing baseline 04.00


def optical_cube(dates: NDArray[np.int64], bands: int, H: int, W: int, rng: np.random.Generator,
                 breaks: bool = True) -> NDArray[np.float64]:
    """(H, W, bands, T) reflectance: a seasonal curve per band, a level
    change per pixel past the middle when ``breaks``."""
    T = dates.size
    phase = 2 * np.pi * (dates - dates[0]) / 365.25
    cube = np.empty((H, W, bands, T))
    for p in range(H * W):
        r, c = divmod(p, W)
        base = (0.03 + 0.025 * np.arange(bands))[:, None]
        season = (0.02 + 0.004 * p) * np.sin(phase + p)[None, :]
        level = np.where(np.arange(T) > T // 2 + 11 * p, 0.07, 0.0)[None, :] if breaks else 0.0
        cube[r, c] = base + season + level + rng.normal(0, 0.004, (bands, T))
    return cube


def hls_inputs(bands: int, thermal: bool = False, step: int = 8, start: int = L30_START,
               years: float = 7, H: int = 2, W: int = 2, seed: int = 3
               ) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.uint8], NDArray[np.int16]]:
    """(dates, spectral, fmask, raw): an HLS series as the preset takes it,
    every layer the int16 the product stores times 0.0001. ``raw`` is the
    int16 cube; with ``thermal`` the last row is B10, degrees Celsius x 100.
    """
    rng = np.random.default_rng(seed)
    dates = (start + step * np.arange(int(years * 365.25 / step))).astype(np.int64)
    optical = bands - 1 if thermal else bands
    cube = optical_cube(dates, optical, H, W, rng)
    raw = np.round(cube * 10000.0).astype(np.int16)
    if thermal:
        phase = 2 * np.pi * (dates - dates[0]) / 365.25
        celsius = 18.0 + 12.0 * np.sin(phase) + rng.normal(0, 1.0, (H, W, dates.size))
        raw = np.concatenate([raw, np.round(celsius * 100.0).astype(np.int16)[:, :, None]], axis=2)
    fmask = rng.choice(FMASK_VALUES, size=(H, W, dates.size), p=FMASK_P)
    return dates, raw * 1e-4, fmask, raw


def s2_inputs(H: int = 2, W: int = 2, seed: int = 9, breaks: bool = True, years: float = 8
              ) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.uint8], NDArray[np.uint16]]:
    """(dates, spectral, scl, dn): a Sentinel-2 L2A series, 5-day, from
    2015-07, as the preset takes it: offset-corrected reflectance, the DN
    stored with BOA_ADD_OFFSET -1000 from baseline 04.00 and 0 before."""
    rng = np.random.default_rng(seed)
    dates = (S2_START + 5 * np.arange(int(years * 365.25 / 5))).astype(np.int64)
    cube = optical_cube(dates, 10, H, W, rng, breaks=breaks)
    offset = np.where(dates >= S2_OFFSET_DATE, -1000, 0)
    dn = (np.round(cube * 10000.0) - offset).astype(np.uint16)
    scl = rng.choice(SCL_ALL, size=(H, W, dates.size), p=SCL_P)
    return dates, (dn + offset) / 10000.0, scl, dn


#: Each preset's input, written out: the ccdcpp.Input() fields that differ.
#: Every unnamed field is the default.
PHYSICAL = dict(reflectance_scale=1.0)


_MSI = ["blue", "green", "red", "red_edge_1", "red_edge_2", "red_edge_3", "nir_broad", "nir",
        "swir1", "swir2"]


_HLS = ["coastal", "blue", "green", "red", "nir", "swir1", "swir2"]


EXPECTED: dict[str, dict[str, Any]] = {
    "hls_l30": dict(
        PHYSICAL, roles=_HLS + ["thermal"],
        thermal_scale=1.0, thermal_offset=0.0, thermal_at_100c=1.0),
    "hls_s30": dict(PHYSICAL, roles=["coastal"] + _MSI),
    "hls": dict(PHYSICAL, roles=_HLS),
    "sentinel2_l2a": dict(PHYSICAL, roles=_MSI),
}


#: Each preset's detection rows, as "ccdc" resolves them on its input.
DETECTION = {"hls_l30": [2, 3, 4, 5, 6], "hls_s30": [2, 3, 8, 9, 10], "hls": [2, 3, 4, 5, 6],
             "sentinel2_l2a": [1, 2, 7, 8, 9]}


#: Each preset's Tmask rows (window_screen's bands, "ccdc"'s roles resolved).
TMASK = {"hls_l30": [2, 5], "hls_s30": [2, 9], "hls": [2, 5], "sentinel2_l2a": [1, 8]}


BAND_NAMES = {
    "hls_l30": ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B10"),
    "hls_s30": ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"),
    "hls": ("coastal", "blue", "green", "red", "nir", "swir1", "swir2"),
    "sentinel2_l2a": ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"),
}


BUILDERS = {"hls_l30": "hls_fmask", "hls_s30": "hls_fmask", "hls": "hls_fmask",
            "sentinel2_l2a": "sentinel2_scl"}
#: The HLS and Sentinel-2 presets.
HLS_AND_SENTINEL2 = (presets.hls_l30, presets.hls_s30, presets.hls, presets.sentinel2_l2a)


def hand_built(name: str) -> tuple[ccdcpp.Input, ccdcpp.QaTable]:
    """(Input, QaTable) of preset ``name``, by hand."""
    h = ccdcpp.Input(**EXPECTED[name])
    return h, getattr(ccdcpp.qa_tables, BUILDERS[name])(h)


def preset_inputs(name: str) -> Series:
    """(dates, spectral, raw QA) for preset ``name``, in its units."""
    if name == "hls_l30":
        return hls_inputs(8, thermal=True)[:3]
    if name == "hls_s30":
        return hls_inputs(11, step=5, start=S2_START)[:3]
    if name == "hls":
        return hls_inputs(7, step=3)[:3]
    return s2_inputs()[:3]


def models(out: PixelResult) -> dict[str, Any]:
    """The fitted models of a detect output, without the per-date mask."""
    return without(out, "processing_mask")
