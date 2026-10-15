"""Python entry points: input validation, the band layout, the cadence options
and __version__.

usage: python test_bindings.py <module_dir> <source_dir> [unittest args]

<module_dir> holds the built extension (ccdcpp.*.so / .pyd); <source_dir> is
the repository root. ctest runs this as `python_bindings`.
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest

MODULE_DIR, SOURCE_DIR = sys.argv[1], sys.argv[2]
del sys.argv[1:3]
sys.path.insert(0, MODULE_DIR)

import numpy as np  # noqa: E402

import ccdcpp  # noqa: E402

START = 724276  # 1984-01-01


def inputs(bands, timesteps, dates=None, qas=None):
    """(dates, spectral (B, T), qas) of in-range DN, clear QA."""
    d = START + 16 * np.arange(timesteps if dates is None else dates, dtype=np.int64)
    s = np.empty((bands, timesteps), dtype=np.float64)
    s[:] = (1000.0 + 100.0 * np.arange(bands))[:, None]
    if bands > 6:
        s[6] = 2900.0
    q = np.zeros(timesteps if qas is None else qas, dtype=np.uint8)
    return d, s, q


def cube(bands, timesteps, dtype=np.float64, dates=None, qas=None):
    """(dates, spectral (2, 3, B, T), qas (2, 3, T))."""
    d, s, _ = inputs(bands, timesteps, dates=dates)
    spectral = np.broadcast_to(s, (2, 3, bands, timesteps)).astype(dtype)
    q = np.zeros((2, 3, timesteps if qas is None else qas), dtype=np.uint8)
    return d, spectral, q


def options():
    return ccdcpp.HarmonicOptions(), ccdcpp.LassoOptions()


class BandCount(unittest.TestCase):
    def assert_band_error(self, call, bands):
        with self.assertRaises(ValueError) as caught:
            call()
        message = str(caught.exception)
        self.assertIn(f"has {bands} bands", message)
        self.assertIn("expects 7", message)

    def test_detect(self):
        for bands in (6, 8):
            with self.subTest(bands=bands):
                self.assert_band_error(
                    lambda: ccdcpp.detect(*inputs(bands, 40), *options()), bands)
        ccdcpp.detect(*inputs(7, 40), *options())

    def test_detect_cube(self):
        for dtype in (np.float64, np.float32, np.uint16):
            for bands in (6, 8):
                with self.subTest(dtype=dtype, bands=bands):
                    self.assert_band_error(
                        lambda: ccdcpp.detect_cube(
                            *cube(bands, 40, dtype), *options(), threads=1),
                        bands)
        self.assertEqual(
            len(ccdcpp.detect_cube(*cube(7, 40), *options(), threads=1)), 6)

    def test_detect_cube_columns(self):
        for dtype in (np.float64, np.float32, np.uint16):
            for bands in (6, 8):
                with self.subTest(dtype=dtype, bands=bands):
                    self.assert_band_error(
                        lambda: ccdcpp.detect_cube_columns(
                            *cube(bands, 40, dtype), *options(), threads=1),
                        bands)
        out = ccdcpp.detect_cube_columns(*cube(7, 40), *options(), threads=1)
        self.assertEqual(out["pixels_total"], 6)
        self.assertEqual(out["failures"], [])


def ten_band_layout():
    """A 10-band layout without thermal, which passes validation."""
    h = ccdcpp.HarmonicOptions()
    h.NUM_BANDS = 10
    h.DETECTION_BANDS = [1, 2, 6, 8, 9]
    h.TMASK_BANDS = [1, 8]
    h.SATURATION_BANDS = list(range(10))
    h.GREEN_IDX = 1
    h.THERMAL_IDX = ccdcpp.NO_BAND
    return h


def noisy(bands, timesteps, seed=1):
    """inputs() with noise, so every band gets a non-trivial fit."""
    d, s, q = inputs(bands, timesteps)
    s = s + np.random.default_rng(seed).normal(0.0, 20.0, s.shape).round()
    return d, s, q


class BandLayout(unittest.TestCase):
    def test_defaults(self):
        h = ccdcpp.HarmonicOptions()
        self.assertEqual(h.NUM_BANDS, 7)
        self.assertEqual(h.DETECTION_BANDS, [1, 2, 3, 4, 5])
        self.assertEqual(h.TMASK_BANDS, [1, 4])
        self.assertEqual(h.SATURATION_BANDS, [0, 1, 2, 3, 4, 5])
        # The millworks nlcd plugin checks these.
        self.assertEqual(
            [h.BLUE_IDX, h.GREEN_IDX, h.RED_IDX, h.NIR_IDX, h.SWIR1_IDX,
             h.SWIR2_IDX, h.THERMAL_IDX, h.QA_IDX],
            [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(
            [h.QA_FILL, h.QA_CLEAR, h.QA_WATER, h.QA_SHADOW, h.QA_SNOW,
             h.QA_CLOUD, h.QA_CIRRUS1, h.QA_CIRRUS2, h.QA_OCCLUSION],
            [255, 0, 1, 2, 3, 4, 8, 9, 10])
        self.assertEqual(ccdcpp.NO_BAND, 2**64 - 1)
        self.assertEqual(ccdcpp.CCD_MAX_BANDS, 32)

    def test_informational_indices_say_so(self):
        for name in ("BLUE_IDX", "RED_IDX", "NIR_IDX", "SWIR1_IDX", "SWIR2_IDX", "QA_IDX",
                     "QA_CIRRUS1", "QA_CIRRUS2", "QA_OCCLUSION"):
            with self.subTest(name=name):
                self.assertIn("Informational", getattr(ccdcpp.HarmonicOptions, name).__doc__)
        for name in ("GREEN_IDX", "THERMAL_IDX", "QA_FILL", "QA_CLEAR", "QA_WATER",
                     "QA_SHADOW", "QA_SNOW", "QA_CLOUD"):
            with self.subTest(name=name):
                self.assertNotIn("Informational", getattr(ccdcpp.HarmonicOptions, name).__doc__)

    def test_lists_keep_their_order(self):
        h = ccdcpp.HarmonicOptions()
        h.DETECTION_BANDS = [5, 1, 3]
        h.TMASK_BANDS = (4, 1)
        h.SATURATION_BANDS = []
        self.assertEqual(h.DETECTION_BANDS, [5, 1, 3])
        self.assertEqual(h.TMASK_BANDS, [4, 1])
        self.assertEqual(h.SATURATION_BANDS, [])

    def test_list_setter_rejects_what_it_cannot_hold(self):
        h = ccdcpp.HarmonicOptions()
        with self.assertRaises(ValueError):
            h.DETECTION_BANDS = [1, 32]
        with self.assertRaises(ValueError):
            h.DETECTION_BANDS = [0] * 33
        with self.assertRaises(TypeError):
            h.DETECTION_BANDS = [1, -1]
        self.assertEqual(h.DETECTION_BANDS, [1, 2, 3, 4, 5])

    def test_explicit_landsat_layout_equals_default(self):
        h = ccdcpp.HarmonicOptions()
        h.NUM_BANDS = 7
        h.DETECTION_BANDS = [1, 2, 3, 4, 5]
        h.TMASK_BANDS = [1, 4]
        h.SATURATION_BANDS = [0, 1, 2, 3, 4, 5]
        h.GREEN_IDX = 1
        h.THERMAL_IDX = 6
        args = noisy(7, 120)
        self.assertEqual(
            repr(ccdcpp.detect(*args, h, ccdcpp.LassoOptions())),
            repr(ccdcpp.detect(*args, *options())))

    def test_outputs_follow_the_band_count(self):
        h, lo = ten_band_layout(), ccdcpp.LassoOptions()
        d, s, q = noisy(10, 120)
        out = ccdcpp.detect(d, s, q, h, lo)
        self.assertEqual(out["num_bands"], 10)
        self.assertGreater(len(out["models"]), 0)
        for m in out["models"]:
            self.assertEqual(len(m["bands"]), 10)

        self.assertEqual(ccdcpp.detect(*noisy(7, 120), *options())["num_bands"], 7)

        cube_s = np.broadcast_to(s, (2, 3, 10, 120)).copy()
        cube_q = np.zeros((2, 3, 120), dtype=np.uint8)
        pixels = ccdcpp.detect_cube(d, cube_s, cube_q, h, lo, threads=1)
        for p in pixels:
            for m in p["models"]:
                self.assertEqual(len(m["bands"]), 10)

        cols = ccdcpp.detect_cube_columns(d, cube_s, cube_q, h, lo, threads=1)
        n = len(cols["row"])
        self.assertGreater(n, 0)
        self.assertEqual(cols["num_bands"], 10)
        for key in ("intercept", "rmse", "magnitude"):
            self.assertEqual(cols[key].shape, (n, 10))
        self.assertEqual(cols["coefficients"].shape, (n, 10, 7))
        # Each pixel is the detect() pixel.
        first = out["models"][0]["bands"]
        np.testing.assert_array_equal(cols["rmse"][0], [b["rmse"] for b in first])


class LayoutValidation(unittest.TestCase):
    """Each rule of ccd::validate(), as ValueError, before any pixel runs."""

    def broken(self, **fields):
        h = ten_band_layout()
        for k, v in fields.items():
            setattr(h, k, v)
        return h

    def assert_rejected(self, h, bands, expected):
        lo = ccdcpp.LassoOptions()
        d, s, q = inputs(bands, 40)
        cube_s = np.broadcast_to(s, (2, 3, bands, 40)).copy()
        cube_q = np.zeros((2, 3, 40), dtype=np.uint8)
        calls = {
            "detect": lambda: ccdcpp.detect(d, s, q, h, lo),
            "detect_cube": lambda: ccdcpp.detect_cube(d, cube_s, cube_q, h, lo, threads=1),
            "detect_cube_columns":
                lambda: ccdcpp.detect_cube_columns(d, cube_s, cube_q, h, lo, threads=1),
        }
        for name, call in calls.items():
            with self.subTest(entry=name, expected=expected):
                with self.assertRaises(ValueError) as caught:
                    call()
                self.assertIn(expected, str(caught.exception))

    def test_rules(self):
        cases = [
            (self.broken(), 7, "spectral has 7 bands; HarmonicOptions.NUM_BANDS expects 10"),
            (self.broken(NUM_BANDS=33), 33, "ccdcpp accepts at most 32"),
            (self.broken(DETECTION_BANDS=[1, 10]), 10, "DETECTION_BANDS holds band 10"),
            (self.broken(TMASK_BANDS=[1, 12]), 10, "TMASK_BANDS holds band 12"),
            (self.broken(SATURATION_BANDS=[0, 31]), 10, "SATURATION_BANDS holds band 31"),
            (self.broken(DETECTION_BANDS=[1, 2, 1]), 10, "DETECTION_BANDS holds band 1 more than once"),
            (self.broken(TMASK_BANDS=[8, 8]), 10, "TMASK_BANDS holds band 8 more than once"),
            (self.broken(SATURATION_BANDS=[3, 3]), 10, "SATURATION_BANDS holds band 3 more than once"),
            (self.broken(DETECTION_BANDS=[]), 10, "DETECTION_BANDS is empty"),
            (self.broken(TMASK_BANDS=[]), 10, "TMASK_BANDS is empty"),
            (self.broken(GREEN_IDX=ccdcpp.NO_BAND), 10, "GREEN_IDX is NO_BAND"),
            (self.broken(GREEN_IDX=10), 10, "GREEN_IDX is 10"),
            (self.broken(THERMAL_IDX=10), 10, "THERMAL_IDX is 10"),
            (self.broken(THERMAL_IDX=9), 10, "SATURATION_BANDS holds THERMAL_IDX (9)"),
        ]
        for h, bands, expected in cases:
            self.assert_rejected(h, bands, expected)

    def test_cadence_rules(self):
        def broken(**fields):
            h = ccdcpp.HarmonicOptions()
            for k, v in fields.items():
                setattr(h, k, v)
            return h

        nan = float("nan")
        cases = [
            (broken(PEEK_REFERENCE_DAYS=0.0), "PEEK_REFERENCE_DAYS is 0.000000; it must be greater than 0 days"),
            (broken(PEEK_REFERENCE_DAYS=-16.0), "PEEK_REFERENCE_DAYS is -16.000000"),
            (broken(PEEK_REFERENCE_DAYS=nan), "PEEK_REFERENCE_DAYS is nan"),
            (broken(VARIOGRAM_MIN_LAG_DAYS=0), "VARIOGRAM_MIN_LAG_DAYS is 0; it must be greater than 0 days"),
            (broken(VARIOGRAM_MIN_LAG_DAYS=-30), "VARIOGRAM_MIN_LAG_DAYS is -30"),
            (broken(RETRAIN_MIN_OBS=0), "RETRAIN_MIN_OBS is 0; it must be at least 1"),
            (broken(RETRAIN_SPAN_FACTOR=0.0), "RETRAIN_SPAN_FACTOR is 0.000000; it must be greater than 0"),
            (broken(RETRAIN_SPAN_FACTOR=-1.33), "RETRAIN_SPAN_FACTOR is -1.330000"),
            (broken(SEASONAL_RMSE_COUNT=0), "SEASONAL_RMSE_COUNT is 0; it must be at least 1"),
            (broken(SEASONAL_RMSE_DIVISOR=0.0), "SEASONAL_RMSE_DIVISOR is 0.000000; it must be greater than 0"),
            (broken(SEASONAL_RMSE_DIVISOR=-4.0), "SEASONAL_RMSE_DIVISOR is -4.000000"),
            (broken(CHANGE_PROBABILITY=0.0), "CHANGE_PROBABILITY is 0.000000; it must be in (0, 1)"),
            (broken(CHANGE_PROBABILITY=1.0), "CHANGE_PROBABILITY is 1.000000; it must be in (0, 1)"),
            (broken(CHANGE_PROBABILITY=1.5), "CHANGE_PROBABILITY is 1.500000"),
            (broken(CHANGE_PROBABILITY=nan), "CHANGE_PROBABILITY is nan"),
            (broken(PEEK_SIZE=0), "PEEK_SIZE is 0; it must be at least 2"),
            (broken(PEEK_SIZE=1), "PEEK_SIZE is 1; it must be at least 2"),
        ]
        for h, expected in cases:
            self.assert_rejected(h, 7, expected)

    def test_empty_saturation_list_is_allowed(self):
        h = self.broken(SATURATION_BANDS=[])
        ccdcpp.detect(*inputs(10, 40), h, ccdcpp.LassoOptions())


class CadenceOptions(unittest.TestCase):
    """The cadence and change-probability fields and ccdcpp.chi2_quantile."""

    DEFAULTS = {
        "PEEK_REFERENCE_DAYS": 16.0,
        "VARIOGRAM_MIN_LAG_DAYS": 30,
        "RETRAIN_MIN_OBS": 24,
        "RETRAIN_SPAN_FACTOR": 1.33,
        "SEASONAL_RMSE_COUNT": 24,
        "SEASONAL_RMSE_DIVISOR": 4.0,
        "CHANGE_PROBABILITY": 0.99,
    }

    UNITS = {
        "PEEK_REFERENCE_DAYS": "Days",
        "VARIOGRAM_MIN_LAG_DAYS": "Days",
        "RETRAIN_MIN_OBS": "Observations",
        "RETRAIN_SPAN_FACTOR": "Unitless",
        "SEASONAL_RMSE_COUNT": "Observations",
        "SEASONAL_RMSE_DIVISOR": "Unitless",
        "CHANGE_PROBABILITY": "Probability",
    }

    def test_defaults(self):
        h = ccdcpp.HarmonicOptions()
        for name, value in self.DEFAULTS.items():
            with self.subTest(name=name):
                self.assertEqual(getattr(h, name), value)
                self.assertIs(type(getattr(h, name)), type(value))

    def test_docstrings_state_unit_and_reader(self):
        for name, unit in self.UNITS.items():
            with self.subTest(name=name):
                doc = getattr(ccdcpp.HarmonicOptions, name).__doc__
                self.assertTrue(doc.startswith(unit), doc)
                self.assertIn("Standard procedure", doc)
                self.assertIn("default", doc)

    def test_fields_are_writable(self):
        h = ccdcpp.HarmonicOptions()
        h.PEEK_REFERENCE_DAYS = 2.5
        h.VARIOGRAM_MIN_LAG_DAYS = 6
        h.RETRAIN_MIN_OBS = 48
        h.RETRAIN_SPAN_FACTOR = 1.5
        h.SEASONAL_RMSE_COUNT = 30
        h.SEASONAL_RMSE_DIVISOR = 4.69
        h.CHANGE_PROBABILITY = 0.995
        self.assertEqual(
            [h.PEEK_REFERENCE_DAYS, h.VARIOGRAM_MIN_LAG_DAYS, h.RETRAIN_MIN_OBS,
             h.RETRAIN_SPAN_FACTOR, h.SEASONAL_RMSE_COUNT, h.SEASONAL_RMSE_DIVISOR,
             h.CHANGE_PROBABILITY],
            [2.5, 6, 48, 1.5, 30, 4.69, 0.995])
        # Counts are unsigned.
        with self.assertRaises(TypeError):
            h.RETRAIN_MIN_OBS = -1

    def test_chi2_quantile_is_the_change_threshold(self):
        h = ccdcpp.HarmonicOptions()
        # To the last bit.
        self.assertEqual(
            ccdcpp.chi2_quantile(0.99, 5).hex(), h.CHANGE_THRESHOLD.hex())
        self.assertEqual(
            ccdcpp.chi2_quantile(p=h.CHANGE_PROBABILITY, df=len(h.DETECTION_BANDS)),
            h.CHANGE_THRESHOLD)
        self.assertGreater(ccdcpp.chi2_quantile(0.99, 8), ccdcpp.chi2_quantile(0.99, 5))
        self.assertGreater(ccdcpp.chi2_quantile(0.999, 5), ccdcpp.chi2_quantile(0.99, 5))
        self.assertIn("same boost call", ccdcpp.chi2_quantile.__doc__)

    def test_defaults_spelled_out_equal_default(self):
        h = ccdcpp.HarmonicOptions()
        for name, value in self.DEFAULTS.items():
            setattr(h, name, value)
        for args in (noisy(7, 120), noisy(7, 400, seed=3)):
            self.assertEqual(
                repr(ccdcpp.detect(*args, h, ccdcpp.LassoOptions())),
                repr(ccdcpp.detect(*args, *options())))

    def test_peek_reference_days_reaches_the_procedure(self):
        # Eight disturbed observations fill the default peek window of 6 at a
        # 16-day cadence, but not the window of 12 that 32 reference days make.
        rng = np.random.default_rng(1)
        d = START + 16 * np.arange(500, dtype=np.int64)
        years = (d - d[0]) / 365.25
        s = (1000.0 + 100.0 * np.arange(7))[:, None] + 300.0 * np.sin(2 * np.pi * years)
        s = s + rng.normal(0.0, 30.0, s.shape)
        s[6] = 2900.0 + 50.0 * np.sin(2 * np.pi * years)
        s[1:6, 250:258] += 600.0
        q = np.zeros(500, dtype=np.uint8)

        def breaks(h):
            out = ccdcpp.detect(d, s, q, h, ccdcpp.LassoOptions())
            return sum(m["change_probability"] for m in out["models"])

        h = ccdcpp.HarmonicOptions()
        self.assertEqual(breaks(h), 1)
        h.PEEK_REFERENCE_DAYS = 32.0
        self.assertEqual(breaks(h), 0)


class BoundFlags(unittest.TestCase):
    FLAGS = {
        "REFL_MIN_INCLUSIVE": False,
        "REFL_MAX_INCLUSIVE": False,
        "THERM_MIN_INCLUSIVE": False,
        "THERM_MAX_INCLUSIVE": False,
    }

    def test_defaults_and_docstrings(self):
        h = ccdcpp.HarmonicOptions()
        for name, value in self.FLAGS.items():
            with self.subTest(name=name):
                self.assertIs(getattr(h, name), value)
                doc = getattr(ccdcpp.HarmonicOptions, name).__doc__
                self.assertIn(f"default {value}", doc)
                setattr(h, name, not value)
                self.assertIs(getattr(h, name), not value)

    def mask(self, values, band, **flags):
        """processing_mask of detect on a pixel whose `band` holds `values` at
        its first observations."""
        d, s, q = inputs(7, 40)
        s[band, :len(values)] = values
        h, lo = options()
        for name, value in flags.items():
            setattr(h, name, value)
        return ccdcpp.detect(d, s, q, h, lo)["processing_mask"][:len(values)].tolist()

    def test_reflectance_bounds(self):
        values = [0.0, 10000.0, -1.0, 10001.0]
        for flags, expected in (
            (dict(), [0, 0, 0, 0]),
            (dict(REFL_MIN_INCLUSIVE=True, REFL_MAX_INCLUSIVE=True), [1, 1, 0, 0]),
            (dict(REFL_MIN_INCLUSIVE=True), [1, 0, 0, 0]),
            (dict(REFL_MAX_INCLUSIVE=True), [0, 1, 0, 0]),
        ):
            with self.subTest(**flags):
                self.assertEqual([int(v) for v in self.mask(values, 3, **flags)], expected)

    def test_thermal_bounds(self):
        # detect transforms thermal K x 10 -> Celsius x 100 first: k10 * 10 - 27315.
        at = lambda celsius100: (celsius100 + 27315.0) / 10.0  # noqa: E731
        values = [at(-9320.0), at(7070.0), at(0.0)]
        for flags, expected in (
            (dict(), [0, 0, 1]),
            (dict(THERM_MIN_INCLUSIVE=True, THERM_MAX_INCLUSIVE=True), [1, 1, 1]),
            (dict(THERM_MIN_INCLUSIVE=True), [1, 0, 1]),
        ):
            with self.subTest(**flags):
                self.assertEqual([int(v) for v in self.mask(values, 6, **flags)], expected)

    def test_exclusive_bounds_reject_nan(self):
        # An exclusive end rejects NaN, so the pixel fits without it. (Both ends
        # inclusive let NaN through to the fit; see the README.)
        for band, flags in ((3, dict()),
                            (3, dict(REFL_MIN_INCLUSIVE=True)),
                            (3, dict(REFL_MAX_INCLUSIVE=True)),
                            (6, dict())):
            with self.subTest(band=band, **flags):
                self.assertEqual(self.mask([np.nan, 1000.0 if band != 6 else 2900.0], band, **flags),
                                 [0, 1])


class GreenFilterScope(unittest.TestCase):
    STAT_ORD = 731000

    def test_default_and_docstring(self):
        self.assertEqual(ccdcpp.HarmonicOptions().GREEN_FILTER_SCOPE, ccdcpp.GreenFilterScope.StatWindow)
        self.assertEqual(sorted(ccdcpp.GreenFilterScope.__members__), ["All", "Off", "StatWindow"])
        self.assertIn("default StatWindow", ccdcpp.HarmonicOptions.GREEN_FILTER_SCOPE.__doc__)
        self.assertIn("STAT_ORD", ccdcpp.GreenFilterScope.__doc__)

    def pixel(self, first):
        """An insufficient-clear pixel: one clear observation in five, green
        spikes at t = 10 and t = 150 among the clear ones."""
        T = 200
        d = first + 16 * np.arange(T, dtype=np.int64)
        s = np.empty((7, T))
        s[:] = (1000.0 + 100.0 * np.arange(7))[:, None]
        s[6] = 2900.0
        q = np.full(T, 4, dtype=np.uint8)
        q[::5] = 0
        s[1, 10] = s[1, 150] = 3000.0
        return d, s, q

    def detect_mask(self, pixel, scope):
        h, lo = options()
        h.STAT_ORD = self.STAT_ORD
        h.GREEN_FILTER_SCOPE = scope
        out = ccdcpp.detect(*pixel, h, lo)
        self.assertEqual({m["curve_qa"] for m in out["models"]}, {44})
        return out["processing_mask"]

    def test_spikes_before_and_after_stat_ord(self):
        pixel = self.pixel(self.STAT_ORD - 16 * 99)   # t = 99 is STAT_ORD
        for scope, kept in ((ccdcpp.GreenFilterScope.All, (0, 0)),
                            (ccdcpp.GreenFilterScope.StatWindow, (0, 1)),
                            (ccdcpp.GreenFilterScope.Off, (1, 1))):
            with self.subTest(scope=scope):
                mask = self.detect_mask(pixel, scope)
                self.assertEqual((mask[10], mask[150]), kept)
                self.assertEqual(int(mask.sum()), 38 + sum(kept))

    def test_every_date_after_stat_ord(self):
        pixel = self.pixel(self.STAT_ORD + 1)
        masks = [self.detect_mask(pixel, scope) for scope in (
            ccdcpp.GreenFilterScope.All, ccdcpp.GreenFilterScope.StatWindow,
            ccdcpp.GreenFilterScope.Off)]
        for mask in masks:
            np.testing.assert_array_equal(mask, masks[2])
        self.assertEqual((masks[0][10], masks[0][150]), (1, 1))


class TimestepMismatch(unittest.TestCase):
    def test_detect(self):
        for dates, qas in ((39, None), (41, None), (None, 39), (None, 41)):
            with self.subTest(dates=dates, qas=qas):
                with self.assertRaises(RuntimeError):
                    ccdcpp.detect(*inputs(7, 40, dates=dates, qas=qas), *options())

    def test_cubes(self):
        for entry in (ccdcpp.detect_cube, ccdcpp.detect_cube_columns):
            for dates, qas in ((39, None), (41, None), (None, 39), (None, 41)):
                with self.subTest(entry=entry.__name__, dates=dates, qas=qas):
                    with self.assertRaises(RuntimeError):
                        entry(*cube(7, 40, dates=dates, qas=qas), *options(), threads=1)


class DateOrder(unittest.TestCase):
    def swapped(self, d, at):
        d = d.copy()
        d[at], d[at + 1] = d[at + 1], d[at]
        return d

    def test_detect(self):
        d, s, q = inputs(7, 40)
        for at in (0, 20, 38):
            with self.subTest(at=at):
                with self.assertRaises(ValueError) as caught:
                    ccdcpp.detect(self.swapped(d, at), s, q, *options())
                self.assertIn("dates must be non-decreasing", str(caught.exception))
                self.assertIn(f"dates[{at + 1}] = {d[at]} follows dates[{at}] = {d[at + 1]}",
                              str(caught.exception))

    def test_raw_qa(self):
        d, s, _ = inputs(7, 40)
        raw = np.zeros(40, dtype=np.uint16)
        table = ccdcpp.qa_tables.landsat_c2_qa_pixel()
        with self.assertRaises(ValueError):
            ccdcpp.detect(self.swapped(d, 5), s, raw, *options(), qa_table=table)

    def test_cubes(self):
        for entry in (ccdcpp.detect_cube, ccdcpp.detect_cube_columns):
            for dtype in (np.float64, np.float32, np.uint16):
                with self.subTest(entry=entry.__name__, dtype=dtype.__name__):
                    d, spectral, q = cube(7, 40, dtype=dtype)
                    with self.assertRaises(ValueError) as caught:
                        entry(self.swapped(d, 7), spectral, q, *options(), threads=1)
                    self.assertIn("dates must be non-decreasing", str(caught.exception))

    def test_equal_dates_are_deduplicated(self):
        d, s, q = inputs(7, 40)
        d[11] = d[12] = d[10]
        mask = ccdcpp.detect(d, s, q, *options())["processing_mask"]
        self.assertEqual(mask[10:13].tolist(), [1, 0, 0])
        _, spectral, qc = cube(7, 40)
        out = ccdcpp.detect_cube_columns(d, spectral, qc, *options(), threads=1)
        self.assertEqual(out["failures"], [])


class Version(unittest.TestCase):
    def pyproject_version(self):
        with open(os.path.join(SOURCE_DIR, "pyproject.toml"), "rb") as f:
            return tomllib.load(f)["project"]["version"]

    def test_extension_matches_pyproject(self):
        self.assertEqual(ccdcpp.__version__, self.pyproject_version())

    def test_package_exposes_it(self):
        # The wheel layout: python/ccdcpp/*.py next to the extension.
        # Run in a fresh interpreter so the extension imported above is not
        # the one found.
        (extension,) = glob.glob(os.path.join(MODULE_DIR, "ccdcpp.*"))
        with tempfile.TemporaryDirectory() as root:
            package = os.path.join(root, "ccdcpp")
            os.mkdir(package)
            for source in glob.glob(os.path.join(SOURCE_DIR, "python", "ccdcpp", "*.py")):
                shutil.copy(source, package)
            shutil.copy(extension, package)
            out = subprocess.run(
                [sys.executable, "-I", "-c",
                 "import sys; sys.path.insert(0, sys.argv[1]); import ccdcpp; "
                 "print(ccdcpp.__file__); print(ccdcpp.__version__)", root],
                check=True, capture_output=True, text=True,
            ).stdout.split()
        self.assertEqual(out[0], os.path.join(package, "__init__.py"))
        self.assertEqual(out[1], self.pyproject_version())


if __name__ == "__main__":
    unittest.main()
