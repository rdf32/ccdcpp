"""Profiles other than Landsat's: a ten-band profile, and the HLS and Sentinel-2
presets on synthetic series in each preset's units.

usage: python test_profiles_sensors.py <module_dir> <source_dir> [unittest args]

<module_dir> holds the built extension; <source_dir> is the repository root.
The package is assembled as the wheel lays it out (_harness.use_package()).
ctest runs this as `python_profiles_sensors`.
"""
from __future__ import annotations

import inspect
import unittest

from _harness import use_package

use_package()

import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

import ccdcpp  # noqa: E402
from ccdcpp import presets  # noqa: E402
from ccdcpp.results import PixelResult  # noqa: E402

from _compare import assert_same  # noqa: E402
from _corpus import (TEN_BAND_CONFIG, every_entry, strip_band_names,  # noqa: E402
                     ten_band_inputs, ten_band_profile)
from _profiles import (BAND_NAMES, DETECTION, HLS_AND_SENTINEL2, S2_OFFSET_DATE,  # noqa: E402
                       TMASK, band_range_rows, change_threshold, doc_of, fields, hand_built,
                       hls_inputs, models, option_params, outlier_threshold, preset_inputs,
                       qa_table_of, s2_inputs, with_table)


class TenBand(unittest.TestCase):
    def test_both_entry_points_run_ten_bands(self):
        profile = ten_band_profile()
        dates, cube, scl = ten_band_inputs()
        out = ccdcpp.detect_cube(dates, cube, scl, threads=2, input=profile,
                                 config=TEN_BAND_CONFIG)
        self.assertEqual(out["failures"], [])
        self.assertEqual(out["num_bands"], 10)
        assert "band_names" in out
        self.assertEqual(out["band_names"], profile.band_names)
        n = len(out["row"])
        self.assertGreater(n, 4)  # every pixel breaks
        self.assertEqual(set(out["curve_qa"].tolist()), {8})  # Standard, every segment
        self.assertEqual(out["coefficients"].shape, (n, 10, 7))
        self.assertTrue(np.isfinite(out["rmse"]).all())

        one = ccdcpp.detect(dates, cube[1, 0], scl[1, 0], input=profile, config=TEN_BAND_CONFIG)
        assert "band_names" in one
        self.assertEqual(one["band_names"], profile.band_names)
        self.assertTrue(all(len(m["bands"]) == 10 for m in one["models"]))

    def test_is_the_hand_built_run(self):
        profile = ten_band_profile()
        dates, cube, scl = ten_band_inputs()
        h = ccdcpp.Input(roles=["blue", "green", "red", "red_edge_1", "red_edge_2", "red_edge_3",
                                "nir_broad", "nir", "swir1", "swir2"],
                         reflectance_scale=1.0)
        cfg = ccdcpp.config("ccdc").replace(
            change_test__peek_reference_days=10.0,
            shared__detection_bands=["green", "red", "red_edge_2", "nir", "swir1", "swir2"],
            shared__statistics_end="none",
        )
        self.assertEqual(TEN_BAND_CONFIG, cfg)
        self.assertEqual(profile.input(), with_table(h, ccdcpp.qa_tables.sentinel2_scl(h)))
        r = cfg.resolve(h)
        self.assertEqual(r["shared"]["detection_bands"], [1, 2, 4, 7, 8, 9])
        self.assertEqual(r["window_screen"]["bands"], [1, 8])
        self.assertEqual(set(option_params(cfg, "change_threshold", h).values()),
                         {ccdcpp.chi2_quantile(0.99, 6)})
        table = ccdcpp.qa_tables.sentinel2_scl(h)
        assert_same(self,
                    strip_band_names(every_entry(dates, cube, scl, input=profile, config=cfg)),
                    every_entry(dates, cube, scl, input=with_table(h, table), config=cfg))

    def test_documented_example_runs(self):
        # The profile in ccdcpp.profiles' docstring, which is not shipped.
        text = doc_of(ccdcpp.profiles)
        self.assertIn("NOT SHIPPED", text)
        code = text.split("::\n", 1)[1].split("\n\nThe configuration names its detection", 1)[0]
        namespace = {}
        exec(compile(inspect.cleandoc("\n" + code), "<profiles docstring>", "exec"), namespace)
        profile = namespace["landsat_oli_c2"]()
        self.assertFalse(hasattr(presets, "landsat_oli_c2"))
        h = profile.input()
        self.assertEqual((h.bands(), h.index("green"), h.index("thermal")), (8, 2, 7))
        shared = ccdcpp.config("ccdc").resolve(h)["shared"]
        self.assertEqual(shared["detection_bands"], [2, 3, 4, 5, 6])
        self.assertEqual(shared["statistics_end"], "none")
        np.testing.assert_array_equal(profile.input().qa_table.codes,
                                      ccdcpp.qa_tables.landsat_c2_qa_pixel(h).codes)

    def test_statistics_end_matters_after_2017(self):
        dates, cube, scl = ten_band_inputs()
        landsat = TEN_BAND_CONFIG.replace(shared__statistics_end=736694)
        out = ccdcpp.detect_cube(dates, cube, scl, threads=2, input=ten_band_profile(),
                                 config=landsat)
        self.assertNotEqual(set(out["curve_qa"].tolist()), {8})


class SensorPresets(unittest.TestCase):
    def test_options_are_the_written_out_options(self):
        defaults = ccdcpp.config("ccdc")
        for preset in HLS_AND_SENTINEL2:
            name = preset.__name__
            with self.subTest(preset=name):
                h, cfg = preset().input(), defaults
                expected, table = hand_built(name)
                self.assertEqual(fields(h), fields(with_table(expected, table)))
                r = cfg.resolve(h)
                self.assertEqual(r["window_screen"]["bands"], TMASK[name])
                self.assertEqual(r["shared"]["detection_bands"], DETECTION[name])
                self.assertEqual(set(option_params(cfg, "alpha", h).values()), {1e-4})
                for threshold in (change_threshold, outlier_threshold):
                    self.assertEqual(threshold(cfg, h).hex(), threshold(defaults).hex())
                self.assertEqual(r["shared"]["statistics_end"], "none")
                self.assertEqual(preset().band_names, BAND_NAMES[name])
                self.assertEqual(preset().qa_table_options,
                                 dict(sentinel2_l2a=()).get(
                                     name, (("exclude_high_aerosol", False),)))
                np.testing.assert_array_equal(qa_table_of(preset().input()).codes, table.codes)

    def test_band_range_tests_blue_to_swir2(self):
        # "ccdc"'s band_range tests the rows of blue, green, red, nir, swir1
        # and swir2, in every procedure: not coastal, the red edges or B08.
        listed = {"landsat_c1_dn": list(range(6)), "landsat_c1_physical": list(range(6)),
                  "landsat_c2": list(range(6)), "hls_l30": list(range(1, 7)),
                  "hls_s30": [1, 2, 3, 8, 9, 10], "hls": list(range(1, 7)),
                  "sentinel2_l2a": [0, 1, 2, 7, 8, 9]}
        self.assertEqual(sorted(listed), sorted(presets.__all__))
        for name, rows in listed.items():
            with self.subTest(preset=name):
                plan = ccdcpp.plan_of("ccdc", getattr(presets, name)())
                for procedure in ccdcpp.PROCEDURES:
                    steps = [s for s in getattr(plan, procedure).chain
                             if s.kind == ccdcpp.StepKind.BandRange]
                    self.assertEqual([s.bands for s in steps], [rows])

    def test_red_edge_and_broad_nir_are_fitted_not_tested(self):
        for preset in (presets.hls_s30, presets.sentinel2_l2a):
            with self.subTest(preset=preset.__name__):
                p = preset()
                h = p.input()
                detection = ccdcpp.config("ccdc").resolve(h)["shared"]["detection_bands"]
                for role in ("red_edge_1", "red_edge_2", "red_edge_3", "nir_broad"):
                    self.assertNotIn(p.index(role), detection)
                    self.assertNotIn(p.index(role), band_range_rows(h))
                self.assertEqual(p.band_names[p.index("nir")], "B8A")
                self.assertEqual(p.band_names[p.index("nir_broad")], "B08")

    def test_is_the_hand_built_run(self):
        for preset in HLS_AND_SENTINEL2:
            name = preset.__name__
            with self.subTest(preset=name):
                dates, cube, qa = preset_inputs(name)
                h, table = hand_built(name)
                profiled = every_entry(dates, cube, qa, threads=2, input=preset())
                assert_same(self, strip_band_names(profiled),
                            every_entry(dates, cube, qa, threads=2,
                                        input=with_table(h, table), config="ccdc"))
                cols = profiled["detect_cube"]
                self.assertEqual(cols["failures"], [])
                self.assertEqual(cols["band_names"], BAND_NAMES[name])
                self.assertEqual(set(cols["curve_qa"].tolist()), {8})  # Standard
                self.assertGreater(len(cols["row"]), 4)  # every pixel breaks


class HlsInput(unittest.TestCase):
    """The HLS presets take every int16 layer times 0.0001."""

    def run_pixel(self, profile: ccdcpp.SensorProfile, dates: NDArray[np.int64],
                  cube: NDArray[np.float64], fmask: NDArray[np.uint8],
                  keep: NDArray[np.intp] | None = None) -> PixelResult:
        if keep is not None:
            dates, cube, fmask = dates[keep], cube[..., keep], fmask[..., keep]
        return ccdcpp.detect(dates, np.ascontiguousarray(cube), fmask, input=profile)

    def test_every_layer_is_the_stored_int16_times_0_0001(self):
        dates, cube, _, raw = hls_inputs(8, thermal=True)
        self.assertEqual(raw.dtype, np.int16)
        np.testing.assert_array_equal(cube, raw * 1e-4)
        # Thermal B10 is degrees Celsius x 100; times 0.0001 it is degrees
        # Celsius / 100, inside "ccdc"'s -93.2..70.7 degrees as the input resolves it.
        therm = ccdcpp.plan_of("ccdc", presets.hls_l30()).standard.chain[1]
        self.assertTrue(((cube[:, :, 7] > therm.min) & (cube[:, :, 7] < therm.max)).all())

    def test_fill_is_dropped(self):
        dates, cube, fmask, raw = hls_inputs(8, thermal=True, H=1, W=1)
        clear = np.flatnonzero(fmask[0, 0] == 0)
        optical, thermal = clear[5::40], clear[25::40]
        raw = raw[0, 0].copy()
        raw[3, optical] = -9999  # B04
        raw[7, thermal] = -9999  # B10
        filled = raw * 1e-4
        profile = presets.hls_l30()
        out = self.run_pixel(profile, dates, filled, fmask[0, 0])
        self.assertFalse(out["processing_mask"][optical].any())
        self.assertFalse(out["processing_mask"][thermal].any())
        self.assertTrue(out["processing_mask"][clear[1::40]].all())
        keep = np.ones(dates.size, dtype=bool)
        keep[optical] = keep[thermal] = False
        assert_same(self, models(out),
                    models(self.run_pixel(profile, dates, filled, fmask[0, 0], keep)))

    def test_coastal_is_screened_where_listed(self):
        # "ccdc"'s band_range does not test coastal; listed, 10000 (1.0) is
        # out of the exclusive reflectance range, 9999 in.
        dates, cube, fmask, raw = hls_inputs(8, thermal=True, H=1, W=1)
        clear = np.flatnonzero(fmask[0, 0] == 0)
        high, edge = clear[7::40], clear[27::40]
        raw = raw[0, 0].copy()
        raw[0, high] = 10000
        raw[0, edge] = 9999
        out = self.run_pixel(presets.hls_l30(), dates, raw * 1e-4, fmask[0, 0])
        self.assertTrue(out["processing_mask"][high].all())
        listed = ccdcpp.config("ccdc").replace(preprocess__band_range__bands=[
            "coastal", "blue", "green", "red", "nir", "swir1", "swir2"])
        out = ccdcpp.detect(dates, np.ascontiguousarray(raw * 1e-4), fmask[0, 0],
                            input=presets.hls_l30(), config=listed)
        self.assertFalse(out["processing_mask"][high].any())
        self.assertTrue(out["processing_mask"][edge].all())

    def test_hls_same_date_keeps_the_first(self):
        # An S30 observation on the date of an L30 one, right after it: the
        # L30 observation is kept and the run is the run without the S30 one.
        dates, cube, fmask, _ = hls_inputs(7, step=4, H=1, W=1)
        cube, fmask = cube[0, 0], fmask[0, 0]
        clear = np.flatnonzero(fmask == 0)
        twins = clear[3::9]
        order = np.sort(np.concatenate([np.arange(dates.size), twins]))
        second = np.flatnonzero(np.diff(order) == 0) + 1
        pair_dates = dates[order]
        pair_fmask = fmask[order]
        pair_cube = cube[:, order].copy()
        pair_cube[:, second] += 0.05  # the S30 observation differs
        profile = presets.hls()
        out = self.run_pixel(profile, pair_dates, pair_cube, pair_fmask)
        self.assertTrue(out["processing_mask"][second - 1].all())
        self.assertFalse(out["processing_mask"][second].any())
        keep = np.ones(order.size, dtype=bool)
        keep[second] = False
        assert_same(self, models(out),
                    models(self.run_pixel(profile, pair_dates, pair_cube, pair_fmask, keep)))
        # The other order keeps the other observation.
        swapped = np.arange(order.size)
        swapped[second], swapped[second - 1] = second - 1, second
        other = self.run_pixel(profile, pair_dates, pair_cube[:, swapped], pair_fmask)
        self.assertNotEqual(str(models(other)), str(models(out)))


class Sentinel2Input(unittest.TestCase):
    """sentinel2_l2a takes offset-corrected BOA reflectance."""

    def test_offset_corrected_input(self):
        dates, cube, scl, dn = s2_inputs()
        offset = np.where(dates >= S2_OFFSET_DATE, -1000, 0)
        np.testing.assert_array_equal(cube, (dn + offset) / 10000.0)
        out = ccdcpp.detect_cube(dates, cube, scl, threads=2,
                                         input=presets.sentinel2_l2a())
        self.assertEqual(out["failures"], [])
        self.assertGreater(len(out["row"]), 4)

    def test_uncorrected_input_differs(self):
        # dn / 10000 without the offset: 0.1 too bright from 2022-01-25.
        dates, cube, scl, dn = s2_inputs(breaks=False)
        profile = presets.sentinel2_l2a()
        corrected = ccdcpp.detect_cube(dates, cube, scl, threads=2, input=profile)
        raw = ccdcpp.detect_cube(dates, dn / 10000.0, scl, threads=2, input=profile)
        # One segment per pixel without the jump; a break at the baseline
        # change with it.
        self.assertEqual(len(corrected["row"]), 4)
        self.assertGreater(len(raw["row"]), len(corrected["row"]))
        breaks = raw["break_day"][raw["break_day"] > 0]
        self.assertTrue((np.abs(breaks - S2_OFFSET_DATE) < 60).any(), breaks)

    def test_no_data_is_dropped(self):
        dates, cube, scl, dn = s2_inputs(H=1, W=1)
        clear = np.flatnonzero(np.isin(scl[0, 0], (4, 5)))
        before = clear[(dates[clear] < S2_OFFSET_DATE)][3::30]
        after = clear[(dates[clear] >= S2_OFFSET_DATE)][3::30]
        dn = dn[0, 0].copy()
        dn[2, before] = 0
        dn[2, after] = 0
        offset = np.where(dates >= S2_OFFSET_DATE, -1000, 0)
        out = ccdcpp.detect(dates, (dn + offset) / 10000.0, scl[0, 0],
                            input=presets.sentinel2_l2a())
        self.assertFalse(out["processing_mask"][before].any())
        self.assertFalse(out["processing_mask"][after].any())
        self.assertTrue(out["processing_mask"][clear[1::30]].all())


if __name__ == "__main__":
    unittest.main()
