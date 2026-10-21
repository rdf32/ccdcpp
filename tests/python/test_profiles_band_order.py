"""Band order: a profile's rows in any order give the same run, and band_names= checks
the stack's order against the profile's.

usage: python test_profiles_band_order.py <module_dir> <source_dir> [unittest args]

<module_dir> holds the built extension; <source_dir> is the repository root.
The package is assembled as the wheel lays it out (_harness.use_package()).
ctest runs this as `python_profiles_band_order`.
"""
from __future__ import annotations

import unittest
from collections.abc import Callable
from typing import Any

from _harness import use_package

use_package()

import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

import ccdcpp  # noqa: E402
from ccdcpp import presets  # noqa: E402

from _compare import assert_same  # noqa: E402
from _corpus import (CODES, CUBE_DN, DATES, every_entry, ten_band_inputs,  # noqa: E402
                     ten_band_profile, without)
from _profiles import NEW_PRESETS, permuted, preset_inputs, unpermute  # noqa: E402


#: The band order of a Sentinel-2 store that keeps the 10 m bands first
#: (B08 before the red edges), as the validation sample's store does.
S2_STORE_ORDER = ("B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12")


class Permutation(unittest.TestCase):
    def test_rows_in_any_order_are_the_same_run(self):
        # A profile's rows in another order, the spectral rows with them: each role
        # is at its new row, and the outputs, put back in the profile's order, are
        # the profile's.
        runs = [
            (presets.landsat_c1_dn(), (DATES, CUBE_DN, CODES), [6, 3, 0, 5, 1, 4, 2]),
            (ten_band_profile(), ten_band_inputs(), [9, 0, 8, 1, 7, 2, 6, 3, 5, 4]),
            (presets.hls_s30(), preset_inputs("hls_s30"), [10, 0, 9, 1, 8, 2, 7, 3, 6, 4, 5]),
        ]
        for base, (dates, cube, qa), order in runs:
            with self.subTest(profile=base.name):
                profile = permuted(base, order)
                h = profile.input()
                for row, role in enumerate(base.roles):
                    self.assertEqual(h.index(role), order.index(row), role)
                detection = ccdcpp.config("ccdc").resolve(base.input())["shared"]["detection_bands"]
                self.assertEqual(ccdcpp.config("ccdc").resolve(h)["shared"]["detection_bands"],
                                 [order.index(i) for i in detection])
                expected = every_entry(dates, cube, qa, threads=2, input=base)
                got = every_entry(dates, np.ascontiguousarray(cube[:, :, order]), qa, threads=2,
                                  input=profile)
                assert_same(self, unpermute(got, order), expected)


class BandNamesCheck(unittest.TestCase):
    """band_names= on the entry points: the stack's row names, which must be
    the profile's in the profile's order."""

    def entries(self, dates: NDArray[np.int64], cube: NDArray[np.float64], qa: NDArray[np.uint8],
                **kw: Any) -> dict[str, Callable[[], object]]:
        return {
            "detect": lambda: ccdcpp.detect(dates, cube[0, 0], qa[0, 0], **kw),
            "detect_cube": lambda: ccdcpp.detect_cube(dates, cube, qa, threads=2, **kw),
        }

    def test_profile_order_runs_unchanged(self):
        runs = [(p, preset_inputs(p.__name__)) for p in NEW_PRESETS]
        runs.append((presets.landsat_c1_dn, (DATES, CUBE_DN, CODES)))
        for preset, (dates, cube, qa) in runs:
            with self.subTest(preset=preset.__name__):
                profile = preset()
                expected = every_entry(dates, cube, qa, threads=2, input=profile)
                for names in (profile.band_names, list(profile.band_names),
                              np.array(profile.band_names)):
                    assert_same(self, every_entry(dates, cube, qa, threads=2, input=profile,
                                                  band_names=names), expected)

    def test_refuses_a_misordered_sentinel2_stack(self):
        profile = presets.sentinel2_l2a()
        dates, cube, scl = preset_inputs("sentinel2_l2a")
        store = [profile.band_names.index(b) for b in S2_STORE_ORDER]
        stacked = np.ascontiguousarray(cube[:, :, store])
        for name, call in self.entries(dates, stacked, scl, input=profile,
                                       band_names=S2_STORE_ORDER).items():
            with self.subTest(entry=name):
                with self.assertRaises(ValueError) as caught:
                    call()
                message = str(caught.exception)
                self.assertIn("SensorProfile 'sentinel2_l2a'", message)
                self.assertIn("in another order: row 3 is 'B08', where the profile has 'B05'",
                              message)
                self.assertIn("reorder the band axis of spectral to B02, B03, B04, B05", message)
        # What the check prevents: without band_names the same stack runs,
        # B08 read as red edge 1, and gives another result.
        silent = ccdcpp.detect_cube(dates, stacked, scl, threads=2, input=profile)
        right = ccdcpp.detect_cube(dates, cube, scl, threads=2, input=profile)
        self.assertEqual(silent["failures"], [])
        self.assertNotEqual(str(without(silent, "threads")), str(without(right, "threads")))
        # Reordered by the names, it is the right run.
        order = [S2_STORE_ORDER.index(b) for b in profile.band_names]
        assert_same(self, ccdcpp.detect_cube(
            dates, np.ascontiguousarray(stacked[:, :, order]), scl, threads=2,
            input=profile, band_names=tuple(np.array(S2_STORE_ORDER)[order])), right)

    def test_other_names_raise(self):
        profile = presets.hls_l30()
        dates, cube, fmask = preset_inputs("hls_l30")
        names = profile.band_names
        cases = [
            (names[:-1] + ("B11",), "missing B10; not in the profile: B11"),
            (names[:-1], "missing B10"),
            (names + ("B11",), "not in the profile: B11"),
            (names[:-1] + ("B07",), "missing B10; repeated: B07"),
            (tuple(n.lower() for n in names), "not in the profile: b01"),
        ]
        for given, text in cases:
            with self.subTest(given=given):
                with self.assertRaises(ValueError) as caught:
                    ccdcpp.detect(dates, cube[0, 0], fmask[0, 0], input=profile,
                                  band_names=given)
                self.assertIn(text, str(caught.exception))
                self.assertIn("the profile has B01, B02", str(caught.exception))
        for given in ("B01B02B03B04B05B06B07B10", b"B01", ("B01",) * 7 + (10,)):
            with self.subTest(given=given):
                with self.assertRaises(TypeError):
                    ccdcpp.detect(dates, cube[0, 0], fmask[0, 0], input=profile,
                                  band_names=given)  # pyright: ignore[reportArgumentType]  # not names, refused

    def test_needs_a_profile(self):
        h = ccdcpp.Input()
        names = presets.landsat_c1_dn().band_names
        for name, call in self.entries(DATES, CUBE_DN, CODES, input=h,
                                       band_names=names).items():
            with self.subTest(entry=name):
                with self.assertRaises(TypeError) as caught:
                    call()
                self.assertIn("pass input= a ccdcpp.SensorProfile", str(caught.exception))

    def test_check_takes_band_names(self):
        profile = presets.sentinel2_l2a()
        profile.check(np.zeros((10, 3)), profile.band_names)
        with self.assertRaises(ValueError):
            profile.check(np.zeros((10, 3)), S2_STORE_ORDER)
        # The count is checked first.
        with self.assertRaises(ValueError) as caught:
            profile.check(np.zeros((9, 3)), profile.band_names)
        self.assertIn("has 9 bands", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
