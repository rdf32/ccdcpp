"""Python entry points: input validation, the band layout, the cadence parameters
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
    """The default input description, as the entry points' positional tail."""
    return (ccdcpp.HarmonicOptions(),)


def preset(**changes):
    """The built-in ccdc preset as the extension takes it (a dict of stages),
    with stage__param=value changes."""
    tree = ccdcpp._config_preset("ccdc")
    for key, value in changes.items():
        stage, param = key.split("__")
        assert param in tree[stage]["params"], key
        tree[stage]["params"][param] = value
    return tree


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
    h.SATURATION_BANDS = list(range(10))
    h.GREEN_IDX = 1
    h.THERMAL_IDX = ccdcpp.NO_BAND
    return h


def ten_band_config(**changes):
    """The configuration for ten_band_layout(): Tmask on bands 1 and 8."""
    return preset(**dict({"window_screen__bands": [1, 8]}, **changes))


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
        self.assertEqual(preset()["window_screen"]["params"]["bands"], [1, 4])
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
        h.SATURATION_BANDS = (4, 1)
        self.assertEqual(h.DETECTION_BANDS, [5, 1, 3])
        self.assertEqual(h.SATURATION_BANDS, [4, 1])
        h.SATURATION_BANDS = []
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
        h.SATURATION_BANDS = [0, 1, 2, 3, 4, 5]
        h.GREEN_IDX = 1
        h.THERMAL_IDX = 6
        args = noisy(7, 120)
        self.assertEqual(
            repr(ccdcpp.detect(*args, h, config=preset(window_screen__bands=[1, 4]))),
            repr(ccdcpp.detect(*args, *options())))

    def test_outputs_follow_the_band_count(self):
        h, cfg = ten_band_layout(), ten_band_config()
        d, s, q = noisy(10, 120)
        out = ccdcpp.detect(d, s, q, h, config=cfg)
        self.assertEqual(out["num_bands"], 10)
        self.assertGreater(len(out["models"]), 0)
        for m in out["models"]:
            self.assertEqual(len(m["bands"]), 10)

        self.assertEqual(ccdcpp.detect(*noisy(7, 120), *options())["num_bands"], 7)

        cube_s = np.broadcast_to(s, (2, 3, 10, 120)).copy()
        cube_q = np.zeros((2, 3, 120), dtype=np.uint8)
        pixels = ccdcpp.detect_cube(d, cube_s, cube_q, h, threads=1, config=cfg)
        for p in pixels:
            for m in p["models"]:
                self.assertEqual(len(m["bands"]), 10)

        cols = ccdcpp.detect_cube_columns(d, cube_s, cube_q, h, threads=1, config=cfg)
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
    """Each rule of ccd::validate() and of the configuration's schemas, as
    ValueError, before any pixel runs."""

    def broken(self, **fields):
        h = ten_band_layout()
        for k, v in fields.items():
            setattr(h, k, v)
        return h

    def assert_rejected(self, h, bands, expected, config=None):
        d, s, q = inputs(bands, 40)
        cube_s = np.broadcast_to(s, (2, 3, bands, 40)).copy()
        cube_q = np.zeros((2, 3, 40), dtype=np.uint8)
        calls = {
            "detect": lambda: ccdcpp.detect(d, s, q, h, config=config),
            "detect_cube":
                lambda: ccdcpp.detect_cube(d, cube_s, cube_q, h, threads=1, config=config),
            "detect_cube_columns":
                lambda: ccdcpp.detect_cube_columns(d, cube_s, cube_q, h, threads=1,
                                                   config=config),
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
            (self.broken(SATURATION_BANDS=[0, 31]), 10, "SATURATION_BANDS holds band 31"),
            (self.broken(DETECTION_BANDS=[1, 2, 1]), 10, "DETECTION_BANDS holds band 1 more than once"),
            (self.broken(SATURATION_BANDS=[3, 3]), 10, "SATURATION_BANDS holds band 3 more than once"),
            (self.broken(DETECTION_BANDS=[]), 10, "DETECTION_BANDS is empty"),
            (self.broken(GREEN_IDX=ccdcpp.NO_BAND), 10, "GREEN_IDX is NO_BAND"),
            (self.broken(GREEN_IDX=10), 10, "GREEN_IDX is 10"),
            (self.broken(THERMAL_IDX=10), 10, "THERMAL_IDX is 10"),
            (self.broken(THERMAL_IDX=9), 10, "SATURATION_BANDS holds THERMAL_IDX (9)"),
        ]
        for h, bands, expected in cases:
            self.assert_rejected(h, bands, expected)

        # The Tmask bands are window_screen's parameter, checked against the
        # input's bands.
        tmask = "window_screen 'tmask' parameter 'bands' "
        for bands, expected in (([1, 12], "holds band 12; the input has 10 bands"),
                                ([8, 8], "holds band 8 twice"),
                                ([], "is empty")):
            self.assert_rejected(self.broken(), 10, tmask + expected,
                                 config=ten_band_config(window_screen__bands=bands))

    def test_cadence_rules(self):
        # The cadence parameters are the configuration's: the schemas refuse
        # them by stage, strategy and parameter.
        nan = float("nan")
        chi2 = "change_test 'chi2' parameter "
        cases = [
            (dict(change_test__peek_reference_days=0.0), chi2 + "'peek_reference_days' is 0; it must be greater than 0"),
            (dict(change_test__peek_reference_days=-16.0), chi2 + "'peek_reference_days' is -16"),
            (dict(change_test__peek_reference_days=nan), chi2 + "'peek_reference_days' is nan"),
            (dict(noise_floor__min_lag_days=0), "noise_floor 'variogram' parameter 'min_lag_days' is 0; it must be greater than 0"),
            (dict(noise_floor__min_lag_days=-30), "noise_floor 'variogram' parameter 'min_lag_days' is -30"),
            (dict(refit__min_obs=0), "refit 'span-factor' parameter 'min_obs' is 0; it must be at least 1"),
            (dict(refit__min_obs=-1), "refit 'span-factor' parameter 'min_obs' is -1"),
            (dict(refit__factor=0.0), "refit 'span-factor' parameter 'factor' is 0; it must be greater than 0"),
            (dict(refit__factor=-1.33), "refit 'span-factor' parameter 'factor' is -1.33"),
            (dict(change_test__seasonal_count=0), chi2 + "'seasonal_count' is 0; it must be at least 1"),
            (dict(change_test__seasonal_divisor=0.0), chi2 + "'seasonal_divisor' is 0; it must be greater than 0"),
            (dict(change_test__seasonal_divisor=-4.0), chi2 + "'seasonal_divisor' is -4"),
            (dict(change_test__change_probability=0.0), chi2 + "'change_probability' is 0; it must be in (0, 1)"),
            (dict(change_test__change_probability=1.0), chi2 + "'change_probability' is 1; it must be in (0, 1)"),
            (dict(change_test__change_probability=1.5), chi2 + "'change_probability' is 1.5"),
            (dict(change_test__change_probability=nan), chi2 + "'change_probability' is nan"),
            (dict(change_test__peek_size=0), chi2 + "'peek_size' is 0; it must be at least 2"),
            (dict(change_test__peek_size=1), chi2 + "'peek_size' is 1; it must be at least 2"),
        ]
        for changes, expected in cases:
            self.assert_rejected(ccdcpp.HarmonicOptions(), 7, expected, config=preset(**changes))

    def test_empty_saturation_list_is_allowed(self):
        h = self.broken(SATURATION_BANDS=[])
        ccdcpp.detect(*inputs(10, 40), h, config=ten_band_config())


class CadenceParameters(unittest.TestCase):
    """The configuration's cadence and change-probability parameters, and
    ccdcpp.chi2_quantile."""

    # (stage, parameter): the ccdc preset's value.
    DEFAULTS = {
        ("change_test", "peek_reference_days"): 16.0,
        ("noise_floor", "min_lag_days"): 30,
        ("refit", "min_obs"): 24,
        ("refit", "factor"): 1.33,
        ("change_test", "seasonal_count"): 24,
        ("change_test", "seasonal_divisor"): 4.0,
        ("change_test", "change_probability"): 0.99,
    }

    UNITS = {
        ("change_test", "peek_reference_days"): "days",
        ("noise_floor", "min_lag_days"): "days",
        ("refit", "min_obs"): "observations",
        ("refit", "factor"): "none",
        ("change_test", "seasonal_count"): "observations",
        ("change_test", "seasonal_divisor"): "none",
        ("change_test", "change_probability"): "none",
    }

    # HarmonicOptions' fields: the input's band layout, QA codes and screens.
    INPUT_FIELDS = {
        "NUM_BANDS", "BLUE_IDX", "GREEN_IDX", "RED_IDX", "NIR_IDX", "SWIR1_IDX",
        "SWIR2_IDX", "THERMAL_IDX", "QA_IDX", "DETECTION_BANDS", "SATURATION_BANDS",
        "QA_FILL", "QA_CLEAR", "QA_WATER", "QA_SHADOW", "QA_SNOW", "QA_CLOUD",
        "QA_CIRRUS1", "QA_CIRRUS2", "QA_OCCLUSION", "CLEAR_OBSERVATION_THRESHOLD",
        "THERMAL_SCALE", "THERMAL_OFFSET", "REFL_MIN", "REFL_MAX", "REFL_MIN_INCLUSIVE",
        "REFL_MAX_INCLUSIVE", "THERM_MIN", "THERM_MAX", "THERM_MIN_INCLUSIVE",
        "THERM_MAX_INCLUSIVE", "MEDIAN_GREEN_FILTER", "GREEN_FILTER_SCOPE", "STAT_ORD",
    }

    def test_defaults(self):
        tree = preset()
        for (stage, name), value in self.DEFAULTS.items():
            with self.subTest(stage=stage, name=name):
                self.assertEqual(tree[stage]["params"][name], value)
                self.assertIs(type(tree[stage]["params"][name]), type(value))

    def test_the_registry_states_unit_and_default(self):
        table = ccdcpp._strategies()
        tree = preset()
        for (stage, name), unit in self.UNITS.items():
            with self.subTest(stage=stage, name=name):
                (param,) = [p for p in table[stage][tree[stage]["strategy"]]
                            if p["name"] == name]
                self.assertEqual(param["unit"], unit)
                self.assertTrue(param["doc"])
                self.assertEqual(param["default"], self.DEFAULTS[(stage, name)])

    def test_harmonic_options_describe_the_input_only(self):
        # The algorithm's parameters are the configuration's; HarmonicOptions
        # holds the input's fields and takes no others.
        h = ccdcpp.HarmonicOptions()
        self.assertEqual({n for n in dir(h) if not n.startswith("_")}, self.INPUT_FIELDS)
        for stage, entry in preset().items():
            for name in entry["params"]:
                with self.subTest(stage=stage, name=name):
                    with self.assertRaises(AttributeError):
                        setattr(h, name.upper(), 1)


    def test_chi2_quantile_is_the_change_threshold(self):
        h = ccdcpp.HarmonicOptions()
        tree = preset()
        for stage in ("change_test", "stability_rule", "lookback"):
            with self.subTest(stage=stage):
                params = tree[stage]["params"]
                # To the last bit.
                self.assertEqual(
                    ccdcpp.chi2_quantile(0.99, 5).hex(), params["change_threshold"].hex())
                self.assertEqual(
                    ccdcpp.chi2_quantile(p=params["change_probability"],
                                         df=len(h.DETECTION_BANDS)),
                    params["change_threshold"])
        self.assertGreater(ccdcpp.chi2_quantile(0.99, 8), ccdcpp.chi2_quantile(0.99, 5))
        self.assertGreater(ccdcpp.chi2_quantile(0.999, 5), ccdcpp.chi2_quantile(0.99, 5))
        self.assertIn("same boost call", ccdcpp.chi2_quantile.__doc__)

    def test_defaults_spelled_out_equal_default(self):
        cfg = preset(**{f"{stage}__{name}": value
                        for (stage, name), value in self.DEFAULTS.items()})
        for args in (noisy(7, 120), noisy(7, 400, seed=3)):
            self.assertEqual(
                repr(ccdcpp.detect(*args, ccdcpp.HarmonicOptions(), config=cfg)),
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

        def breaks(config):
            out = ccdcpp.detect(d, s, q, ccdcpp.HarmonicOptions(), config=config)
            return sum(m["change_probability"] for m in out["models"])

        self.assertEqual(breaks(None), 1)
        self.assertEqual(breaks(preset(change_test__peek_reference_days=32.0)), 0)


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
        h = ccdcpp.HarmonicOptions()
        for name, value in flags.items():
            setattr(h, name, value)
        return ccdcpp.detect(d, s, q, h)["processing_mask"][:len(values)].tolist()

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
        h = ccdcpp.HarmonicOptions()
        h.STAT_ORD = self.STAT_ORD
        h.GREEN_FILTER_SCOPE = scope
        out = ccdcpp.detect(*pixel, h)
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
