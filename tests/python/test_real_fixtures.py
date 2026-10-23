"""The HLS and Sentinel-2 presets on real pixels, pinned.

usage: python test_real_fixtures.py <module_dir> <source_dir> [--write-expected] [unittest args]

<module_dir> holds the built extension; <source_dir> is the repository root.
The package is assembled as the wheel lays it out (_harness.use_package()).
ctest runs this as `python_real_fixtures`.

The inputs are the raw values of four pixels of MGRS tile T16SGC, 2020-2023,
in HLS v2.0 L30 and S30 and Sentinel-2 L2A, from Planetary Computer; see
fixtures/t16sgc_2020_2023/README.md for their provenance. Each preset runs on
each pixel through ccdcpp.detect and is compared with the output stored in
expected_<preset>.json. --write-expected rewrites those files from the
current build instead of comparing: only for an intended output change.

Pixel `481_393` is named 1-based (column, row), as the production CCD table
names it (px=481, py=393): 0-based chunk row 392, column 480. Its tests say
so in their names.
"""
from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
import os
import sys
import unittest
from collections.abc import Callable
from typing import Any

from _harness import SOURCE_DIR, use_package

use_package()

import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

import ccdcpp  # noqa: E402
from ccdcpp import presets  # noqa: E402
from ccdcpp.results import PixelResult  # noqa: E402

WRITE = "--write-expected" in sys.argv
if WRITE:
    sys.argv.remove("--write-expected")

FIXTURES = os.path.join(SOURCE_DIR, "tests", "python", "fixtures", "t16sgc_2020_2023")

#: Fixture pixel -> file suffix (0-based chunk row and column of ARD chunk
#: h024v013/00030008) and the name its tests carry.
PIXELS = {
    "fixture_481_393": ("ard392_480", "fixture_481_393_one_based_col_row"),
    "water": ("ard301_173", "water"),
    "developed_high": ("ard161_27", "developed_high"),
    "break": ("ard0_90", "break"),
    "coastal": ("hls10_142", "coastal_hls10_142"),
}

#: Pixels held in the HLS products only: "coastal" is HLS window row 10,
#: column 142 (0-based), chosen because a run that tests the coastal B01
#: low bound differs (B01 <= 0 on 7 L30 and 2 S30 observations whose other
#: bands are in range).
HLS_ONLY = {"coastal"}

#: Preset -> the product files its input is read from.
PRODUCTS = {"hls_l30": ("hls_l30",), "hls_s30": ("hls_s30",), "hls": ("hls_l30", "hls_s30"),
            "sentinel2_l2a": ("s2_l2a",)}

#: hls() rows by role, from each product's band IDs.
HLS_ROWS = {
    "hls_l30": ("B01", "B02", "B03", "B04", "B05", "B06", "B07"),
    "hls_s30": ("B01", "B02", "B03", "B04", "B8A", "B11", "B12"),
}

S2_OFFSET = -1000

# Float tolerance against the stored output. On the machine that wrote it the
# comparison is exact; the tolerance allows another compiler's last bits.
RTOL, ATOL = 1e-7, 1e-12


def read_product(product: str, pixel: str) -> tuple[
        NDArray[np.int64], NDArray[np.int64], NDArray[np.uint8], NDArray[np.int64] | None,
        tuple[str, ...]]:
    """(dates, raw (B, T) int64, qa (T,) uint8, offset (T,) int64 or None,
    band IDs) of one fixture CSV."""
    path = os.path.join(FIXTURES, f"{product}_{pixel}_{PIXELS[pixel][0]}.csv")
    with open(path) as f:
        header = f.readline().lstrip("# ").split(" ")[0].strip().split(",")
    data = np.loadtxt(path, delimiter=",", dtype=np.int64, comments="#")
    offset = None
    if product == "s2_l2a":
        assert header[-2:] == ["SCL", "offset_to_apply"], header
        offset = data[:, -1]
        data = data[:, :-1]
    else:
        assert header[-1] == "Fmask", header
    assert header[0] == "ordinal", header
    return data[:, 0], data[:, 1:-1].T, data[:, -1].astype(np.uint8), offset, tuple(header[1:-1 if offset is None else -2])


def preset_input(preset: str, pixel: str, apply_offset: bool = True) -> tuple[
        NDArray[np.int64], NDArray[np.float64], NDArray[np.uint8], tuple[str, ...]]:
    """(dates, spectral (B, T) float64, raw QA, band names in row order): the
    fixture pixel as `preset` takes it. HLS: every int16 layer times 0.0001.
    Sentinel-2: (DN + offset_to_apply) * 0.0001, per scene. hls(): L30 and S30
    rows by role, merged by date with L30 first on a shared date."""
    if preset in ("hls_l30", "hls_s30"):
        dates, raw, qa, _, names = read_product(preset, pixel)
        return dates, raw * 1e-4, qa, names
    if preset == "sentinel2_l2a":
        dates, raw, qa, offset, names = read_product("s2_l2a", pixel)
        assert offset is not None   # Sentinel-2 holds one per scene
        dn = raw + offset[None, :] if apply_offset else raw
        return dates, dn * 1e-4, qa, names
    assert preset == "hls", preset
    parts = []
    for product in PRODUCTS["hls"]:
        dates, raw, qa, _, names = read_product(product, pixel)
        rows = [names.index(b) for b in HLS_ROWS[product]]
        parts.append((dates, raw[rows], qa))
    dates = np.concatenate([p[0] for p in parts])
    order = np.argsort(dates, kind="stable")
    spectral = np.concatenate([p[1] for p in parts], axis=1)[:, order]
    qa = np.concatenate([p[2] for p in parts])[order]
    return dates[order], spectral * 1e-4, qa, presets.hls().band_names


def run(preset: str, pixel: str, apply_offset: bool = True) -> PixelResult:
    dates, spectral, qa, names = preset_input(preset, pixel, apply_offset)
    return ccdcpp.detect(dates, np.ascontiguousarray(spectral), qa,
                         spec=getattr(presets, preset)(), band_names=names)


def as_stored(out: PixelResult) -> dict[str, Any]:
    """A detect output as expected_<preset>.json holds it."""
    return {
        "processing_mask": "".join(str(int(x)) for x in out["processing_mask"]),
        "models": [
            {**{k: m[k] for k in ("start_day", "end_day", "break_day", "observation_count",
                                  "change_probability", "curve_qa")},
             "bands": [{"intercept": b["intercept"], "coefficients": list(b["coefficients"]),
                        "rmse": b["rmse"], "magnitude": b["magnitude"]} for b in m["bands"]]}
            for m in out["models"]
        ],
    }


def expected_path(preset: str) -> str:
    return os.path.join(FIXTURES, f"expected_{preset}.json")


def load_expected(preset: str) -> dict[str, Any]:
    with open(expected_path(preset)) as f:
        return json.load(f)


def breaks(out: PixelResult) -> list[str]:
    """Break dates (ISO) of the segments that end in a change."""
    return [dt.date.fromordinal(m["break_day"]).isoformat()
            for m in out["models"] if m["change_probability"] >= 1.0]


def pixels_of(preset: str) -> list[str]:
    """The fixture pixels a preset has input for."""
    return [p for p in PIXELS if preset != "sentinel2_l2a" or p not in HLS_ONLY]


def write_expected() -> None:
    for preset in PRODUCTS:
        stored = {pixel: as_stored(run(preset, pixel)) for pixel in pixels_of(preset)}
        # One pixel per line: small, and a change shows which pixel moved.
        lines = [f"{json.dumps(pixel)}: {json.dumps(out, separators=(',', ':'))}"
                 for pixel, out in stored.items()]
        with open(expected_path(preset), "w") as f:
            f.write("{\n" + ",\n".join(lines) + "\n}\n")
        print("wrote", expected_path(preset))


class Pinned(unittest.TestCase):
    """Each preset on each pixel gives the stored output."""

    def check(self, preset: str, pixel: str) -> None:
        got = as_stored(run(preset, pixel))
        want = load_expected(preset)[pixel]
        self.assertEqual(got["processing_mask"], want["processing_mask"])
        self.assertEqual(len(got["models"]), len(want["models"]))
        for i, (g, w) in enumerate(zip(got["models"], want["models"])):
            with self.subTest(segment=i):
                for key in ("start_day", "end_day", "break_day", "observation_count",
                            "curve_qa", "change_probability"):
                    self.assertEqual(g[key], w[key], key)
                self.assertEqual(len(g["bands"]), len(w["bands"]))
                for b, (gb, wb) in enumerate(zip(g["bands"], w["bands"])):
                    for key in ("intercept", "coefficients", "rmse", "magnitude"):
                        np.testing.assert_allclose(gb[key], wb[key], rtol=RTOL, atol=ATOL,
                                                   err_msg=f"band {b} {key}")


def _pinned_test(preset: str, pixel: str) -> Callable[[Pinned], None]:
    def test(self: Pinned) -> None:
        self.check(preset, pixel)
    test.__doc__ = f"{preset} on pixel {pixel}, against expected_{preset}.json."
    return test


for _preset in PRODUCTS:
    for _pixel in pixels_of(_preset):
        _label = PIXELS[_pixel][1]
        setattr(Pinned, f"test_{_preset}_{_label}", _pinned_test(_preset, _pixel))


class Fixtures(unittest.TestCase):
    """The fixture files agree with each other and with the products."""

    def scenes(self, product: str) -> list[dict[str, str]]:
        with open(os.path.join(FIXTURES, f"scenes_{product}.csv"), newline="") as f:
            return list(csv.DictReader(f))

    def test_every_pixel_has_the_product_dates_and_scenes(self):
        for product in ("hls_l30", "hls_s30", "s2_l2a"):
            scenes = self.scenes(product)
            ordinals = np.array([int(s["ordinal"]) for s in scenes])
            for pixel in PIXELS:
                if product == "s2_l2a" and pixel in HLS_ONLY:
                    continue
                with self.subTest(product=product, pixel=pixel):
                    dates, raw, qa, offset, names = read_product(product, pixel)
                    np.testing.assert_array_equal(dates, ordinals)
                    self.assertTrue((np.diff(dates) >= 0).all())
                    self.assertEqual(dt.date.fromordinal(int(dates[0])).year, 2020)
                    self.assertEqual(dt.date.fromordinal(int(dates[-1])).year, 2023)
                    if offset is not None:
                        np.testing.assert_array_equal(
                            offset, [int(s["offset_to_apply"]) for s in scenes])

    def test_band_ids_are_the_presets(self):
        for product, preset in (("hls_l30", presets.hls_l30), ("hls_s30", presets.hls_s30),
                                ("s2_l2a", presets.sentinel2_l2a)):
            with self.subTest(product=product):
                self.assertEqual(read_product(product, "water")[4], preset().band_names)

    def test_item_ids_name_the_tile(self):
        for product in ("hls_l30", "hls_s30", "s2_l2a"):
            for s in self.scenes(product):
                ids = s.get("item_id") or s["item_ids"]
                for item in ids.split(";"):
                    self.assertIn("T16SGC", item)

    def test_s2_offset_follows_the_baseline(self):
        # BOA_ADD_OFFSET is -1000 from processing baseline 04.00, 0 before;
        # some 2021 scenes were reprocessed at 04.00, so the first -1000
        # scene comes before the 2022-01-25 cut.
        scenes = self.scenes("s2_l2a")
        for s in scenes:
            self.assertEqual(int(s["offset_to_apply"]),
                             S2_OFFSET if s["processing_baseline"] >= "04.00" else 0, s)
        first = min(int(s["ordinal"]) for s in scenes if int(s["offset_to_apply"]))
        self.assertEqual(dt.date.fromordinal(first), dt.date(2021, 11, 10))


class Findings(unittest.TestCase):
    """What the validation reported on these pixels, checked on the fixtures."""

    def test_fixture_481_393_one_based_col_row_is_one_stable_segment(self):
        for preset in PRODUCTS:
            with self.subTest(preset=preset):
                out = run(preset, "fixture_481_393")
                self.assertEqual(len(out["models"]), 1)
                self.assertEqual(breaks(out), [])

    def test_break_pixel_breaks_where_the_validation_found(self):
        with open(os.path.join(FIXTURES, "pixels.json")) as f:
            found = json.load(f)["break"]["breaks"]
        self.assertEqual(breaks(run("hls_l30", "break")), found["l30"])
        self.assertEqual(breaks(run("hls_s30", "break")), found["s30"])

    def test_fixture_481_393_one_based_col_row_s2_without_offset_breaks_at_the_cut(self):
        # Without the per-scene offset the series jumps by 0.1 at the
        # baseline change and the run breaks there.
        self.assertEqual(breaks(run("sentinel2_l2a", "fixture_481_393")), [])
        self.assertEqual(breaks(run("sentinel2_l2a", "fixture_481_393", apply_offset=False)),
                         ["2022-01-26"])

    def test_coastal_hls10_142_depends_on_the_coastal_low_bound(self):
        # Observations QA keeps with every band in range but B01 <= 0:
        # "ccdc"'s band_range does not test coastal, so they are kept; with
        # coastal in its bands they are dropped, and the result is another
        # one.
        for preset, count in (("hls_l30", 7), ("hls_s30", 2)):
            with self.subTest(preset=preset):
                dates, spectral, qa, names = preset_input(preset, "coastal")
                out = run(preset, "coastal")
                others = np.delete(spectral, names.index("B01"), axis=0)
                if preset == "hls_l30":
                    others = others[:-1]  # B10, thermal bounds
                coastal_only = ((spectral[names.index("B01")] <= 0)
                                & ((others > 0) & (others < 1)).all(axis=0)
                                & np.isin(getattr(presets, preset)().spec().qa_table.codes[qa],
                                          (ccdcpp.DataSpec().qa_clear,
                                           ccdcpp.DataSpec().qa_water)))
                self.assertEqual(int(coastal_only.sum()), count)
                self.assertTrue(out["processing_mask"][coastal_only].all())
                profile = getattr(presets, preset)()
                screened = ccdcpp.config("ccdc").replace(preprocess__band_range__bands=[
                    r for r in profile.roles if r not in ("thermal", "other")])
                other = ccdcpp.detect(dates, np.ascontiguousarray(spectral), qa,
                                      spec=profile, config=screened, band_names=names)
                self.assertFalse(other["processing_mask"][coastal_only].any())
                self.assertNotEqual(as_stored(other), as_stored(out))


if __name__ == "__main__":
    if WRITE:
        write_expected()
    else:
        unittest.main()
