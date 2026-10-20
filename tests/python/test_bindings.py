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
    """The default input, as the entry points' keyword."""
    return {"encoding": ccdcpp.Input()}


def preset(**changes):
    """The built-in ccdc preset as the extension takes it (a dict {"shared",
    "stages", "preprocess"}), with shared__name=value, stage__param=value and
    preprocess__step__param=value (every step of that type) changes."""
    tree = ccdcpp._config_preset("ccdc")
    for key, value in changes.items():
        stage, param = key.split("__", 1)
        if stage == "preprocess":
            step, name = param.split("__")
            found = [s for chain in tree["preprocess"].values() for s in chain["steps"]
                     if s["step"] == step]
            assert found and all(name in s for s in found), key
            for s in found:
                s[name] = value
            continue
        if stage == "shared":
            assert param in tree["shared"], key
            tree["shared"][param] = value
            continue
        assert param in tree["stages"][stage]["params"], key
        tree["stages"][stage]["params"][param] = value
    return tree


def resolved(config=None, encoding=None):
    """The configuration resolved against the encoding (default Input())."""
    return ccdcpp._config_resolve(preset() if config is None else config,
                                  ccdcpp.Input() if encoding is None else encoding)


class BandCount(unittest.TestCase):
    def assert_band_error(self, call, bands):
        with self.assertRaises(ValueError) as caught:
            call()
        message = str(caught.exception)
        self.assertIn(f"has {bands} bands", message)
        self.assertIn("the input has 7", message)

    def test_detect(self):
        for bands in (6, 8):
            with self.subTest(bands=bands):
                self.assert_band_error(
                    lambda: ccdcpp.detect(*inputs(bands, 40), **options()), bands)
        ccdcpp.detect(*inputs(7, 40), **options())

    def test_detect_cube(self):
        for dtype in (np.float64, np.float32, np.uint16):
            for bands in (6, 8):
                with self.subTest(dtype=dtype, bands=bands):
                    self.assert_band_error(
                        lambda: ccdcpp.detect_cube(
                            *cube(bands, 40, dtype), **options(), threads=1),
                        bands)
        out = ccdcpp.detect_cube(*cube(7, 40), **options(), threads=1)
        self.assertEqual(out["pixels_total"], 6)
        self.assertEqual(out["failures"], [])
        self.assertNotIn("processing_mask", out)

    def test_detect_cube_masks_are_detects(self):
        # A 2 x 3 cube of noisy pixels with clouds, fill and snow scattered
        # through them, and one pixel all fill (no segment).
        H, W, T = 2, 3, 120
        rng = np.random.default_rng(5)
        spectral = np.empty((H, W, 7, T))
        qas = np.zeros((H, W, T), dtype=np.uint8)
        for r in range(H):
            for c in range(W):
                d, spectral[r, c], _ = noisy(7, T, seed=10 * r + c)
                qas[r, c] = rng.choice([0, 0, 0, 0, 1, 3, 4, 255], T)
        qas[1, 2] = 255
        for threads in (1, 3):
            with self.subTest(threads=threads):
                out = ccdcpp.detect_cube(d, spectral, qas, **options(), threads=threads,
                                         masks=True)
                mask = out["processing_mask"]
                self.assertEqual((mask.dtype, mask.shape), (np.uint8, (H, W, T)))
                self.assertTrue(mask.flags.c_contiguous and mask.flags.owndata)
                self.assertFalse(mask[1, 2].any())
                for r in range(H):
                    for c in range(W):
                        one = ccdcpp.detect(d, spectral[r, c], qas[r, c], **options())
                        np.testing.assert_array_equal(mask[r, c], one["processing_mask"])
                # The columns are those without masks.
                plain = ccdcpp.detect_cube(d, spectral, qas, **options(), threads=threads)
                self.assertEqual(set(out) - set(plain), {"processing_mask"})
                for key in plain:
                    if hasattr(plain[key], "dtype"):
                        np.testing.assert_array_equal(out[key], plain[key])


TEN_ROLES = ["blue", "green", "red", "red_edge_1", "red_edge_2", "red_edge_3", "nir",
             "other", "swir1", "swir2"]


def ten_band_layout():
    """A 10-band layout without thermal, which passes validation: the
    configuration's detection bands are rows 1, 2, 6, 8, 9 of it, its Tmask
    bands 1 and 8."""
    return ccdcpp.Input(roles=TEN_ROLES)


def ten_band_config(**changes):
    """The configuration for ten_band_layout(): the preset."""
    return preset(**changes)


def noisy(bands, timesteps, seed=1):
    """inputs() with noise, so every band gets a non-trivial fit."""
    d, s, q = inputs(bands, timesteps)
    s = s + np.random.default_rng(seed).normal(0.0, 20.0, s.shape).round()
    return d, s, q


class BandLayout(unittest.TestCase):
    def test_defaults(self):
        h = ccdcpp.Input()
        self.assertEqual(h.bands(), 7)
        self.assertEqual(h.roles, ["blue", "green", "red", "nir", "swir1", "swir2", "thermal"])
        self.assertEqual(preset()["shared"]["detection_bands"],
                         ["green", "red", "nir", "swir1", "swir2"])
        self.assertEqual(preset()["stages"]["window_screen"]["params"]["bands"],
                         ["green", "swir1"])
        self.assertEqual(resolved()["shared"]["detection_bands"], [1, 2, 3, 4, 5])
        self.assertEqual(resolved()["stages"]["window_screen"]["params"]["bands"], [1, 4])
        band_range = [s for s in preset()["preprocess"]["standard"]["steps"]
                      if s["step"] == "band_range"]
        self.assertEqual([s["bands"] for s in band_range], ["reflective"])
        # The millworks nlcd plugin checks these.
        self.assertEqual([h.index(r) for r in h.roles], [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(h.index("coastal"), ccdcpp.NO_BAND)
        self.assertEqual(
            [h.qa_fill, h.qa_clear, h.qa_water, h.qa_shadow, h.qa_snow,
             h.qa_cloud, h.qa_cirrus1, h.qa_cirrus2, h.qa_occlusion],
            [255, 0, 1, 2, 3, 4, 8, 9, 10])
        self.assertEqual(h.reflectance_scale, 10000.0)
        self.assertEqual(ccdcpp.NO_BAND, 2**64 - 1)
        self.assertEqual(ccdcpp.CCD_MAX_BANDS, 32)

    def test_immutable_by_keyword(self):
        h = ccdcpp.Input(reflectance_scale=1.0, roles=["blue", "green", "red"], saturation_code=None)
        self.assertEqual((h.reflectance_scale, h.roles, h.saturation_code), (1.0, ["blue", "green", "red"],
                                                                      None))
        for name in ("roles", "reflectance_scale", "qa_clear", "qa_table", "saturation_code"):
            with self.subTest(name=name):
                with self.assertRaises(AttributeError):
                    setattr(h, name, getattr(h, name))
        h.roles.append("nir")   # a copy
        self.assertEqual(h.roles, ["blue", "green", "red"])
        # replace() is a new input; the old one is unchanged.
        r = h.replace(reflectance_scale=10000.0, qa_snow=7)
        self.assertEqual((r.reflectance_scale, r.qa_snow, r.roles), (10000.0, 7, h.roles))
        self.assertEqual((h.reflectance_scale, h.qa_snow), (1.0, 3))
        # Equal and hashed by value.
        self.assertEqual(r.replace(reflectance_scale=1.0, qa_snow=3), h)
        self.assertEqual(hash(r.replace(reflectance_scale=1.0, qa_snow=3)), hash(h))
        self.assertNotEqual(r, h)
        self.assertEqual(ccdcpp.Input(), ccdcpp.Input().replace())
        # Keywords only, each a field, each of its type.
        with self.assertRaises(TypeError):
            ccdcpp.Input(1.0)
        with self.assertRaisesRegex(TypeError, "Input\\(\\) has no field 'alpha'; the fields are qa_fill"):
            ccdcpp.Input(alpha=1.0)
        with self.assertRaisesRegex(TypeError, "Input.replace\\(\\) has no field 'saturation_bands'"):
            h.replace(saturation_bands=[0])
        with self.assertRaisesRegex(TypeError, "qa_clear is 300, which it cannot hold"):
            ccdcpp.Input(qa_clear=300)
        with self.assertRaisesRegex(TypeError, "roles is 'green'"):
            ccdcpp.Input(roles="green")

    def test_informational_codes_say_so(self):
        for name in ("qa_cirrus1", "qa_cirrus2", "qa_occlusion"):
            with self.subTest(name=name):
                self.assertIn("Informational", getattr(ccdcpp.Input, name).__doc__)
        for name in ("qa_fill", "qa_clear", "qa_water", "qa_shadow", "qa_snow", "qa_cloud"):
            with self.subTest(name=name):
                self.assertNotIn("Informational", getattr(ccdcpp.Input, name).__doc__)

    def test_lists_keep_their_order(self):
        cfg = preset(shared__detection_bands=[5, 1, 3])
        self.assertEqual(resolved(cfg)["shared"]["detection_bands"], [5, 1, 3])
        cfg = preset(shared__detection_bands=["swir2", "green", "nir"])
        self.assertEqual(resolved(cfg)["shared"]["detection_bands"], [5, 1, 3])

    def test_band_lists_refuse_what_they_cannot_hold(self):
        for bands in ([1, 32], [0] * 33, [1, -1]):
            with self.subTest(bands=bands):
                with self.assertRaises(ValueError):
                    resolved(preset(preprocess__band_range__bands=bands))

    def test_explicit_landsat_layout_equals_default(self):
        h = ccdcpp.Input(roles=["blue", "green", "red", "nir", "swir1", "swir2", "thermal"])
        args = noisy(7, 120)
        by_rows = preset(window_screen__bands=[1, 4], shared__detection_bands=[1, 2, 3, 4, 5],
                         preprocess__band_range__bands=[0, 1, 2, 3, 4, 5])
        self.assertEqual(
            repr(ccdcpp.detect(*args, encoding=h, config=by_rows)),
            repr(ccdcpp.detect(*args, **options())))

    def test_outputs_follow_the_band_count(self):
        h, cfg = ten_band_layout(), ten_band_config()
        d, s, q = noisy(10, 120)
        out = ccdcpp.detect(d, s, q, encoding=h, config=cfg)
        self.assertEqual(out["num_bands"], 10)
        self.assertGreater(len(out["models"]), 0)
        for m in out["models"]:
            self.assertEqual(len(m["bands"]), 10)

        self.assertEqual(ccdcpp.detect(*noisy(7, 120), **options())["num_bands"], 7)

        cube_s = np.broadcast_to(s, (2, 3, 10, 120)).copy()
        cube_q = np.zeros((2, 3, 120), dtype=np.uint8)
        cols = ccdcpp.detect_cube(d, cube_s, cube_q, encoding=h, threads=1, config=cfg)
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
        return ten_band_layout().replace(**fields)

    def assert_rejected(self, h, bands, expected, config=None):
        d, s, q = inputs(bands, 40)
        cube_s = np.broadcast_to(s, (2, 3, bands, 40)).copy()
        cube_q = np.zeros((2, 3, 40), dtype=np.uint8)
        calls = {
            "detect": lambda: ccdcpp.detect(d, s, q, encoding=h, config=config),
            "detect_cube":
                lambda: ccdcpp.detect_cube(d, cube_s, cube_q, encoding=h, threads=1, config=config),
        }
        for name, call in calls.items():
            with self.subTest(entry=name, expected=expected):
                with self.assertRaises(ValueError) as caught:
                    call()
                self.assertIn(expected, str(caught.exception))

    def test_rules(self):
        def roles(**at):
            out = list(TEN_ROLES)
            for row, role in at.items():
                out[int(row[1:])] = role
            return out
        cases = [
            (self.broken(), 7, "spectral has 7 bands; the input has 10 (a row per role)"),
            (self.broken(roles=TEN_ROLES + ["other"] * 23), 33, "ccdcpp accepts at most 32"),
            (self.broken(roles=roles(r1="other")), 10, "roles has no 'green'"),
            (self.broken(roles=roles(r7="nir8a")), 10, "roles holds 'nir8a', which is not a role"),
            (self.broken(roles=roles(r7="red")), 10, "roles holds 'red' more than once"),
            (self.broken(reflectance_scale=0.0), 10, "reflectance_scale is 0; it must be a positive number"),
        ]
        for h, bands, expected in cases:
            self.assert_rejected(h, bands, expected)

        # The detection bands are the configuration's shared value, checked
        # against the input's bands.
        detection = "shared value 'detection_bands' "
        for bands, expected in (([1, 10], "holds band 10; the input has 10 bands"),
                                ([1, 2, 1], "holds band 1 twice"),
                                ([], "is empty"),
                                (["green", "coastal"], "holds 'coastal', which the input does "
                                                       "not have")):
            self.assert_rejected(self.broken(), 10, detection + expected,
                                 config=ten_band_config(shared__detection_bands=bands))

        # The Tmask bands are window_screen's parameter, checked against the
        # input's bands.
        tmask = "window_screen 'tmask' parameter 'bands' "
        for bands, expected in (([1, 12], "holds band 12; the input has 10 bands"),
                                ([8, 8], "holds band 8 twice"),
                                ([], "is empty")):
            self.assert_rejected(self.broken(), 10, tmask + expected,
                                 config=ten_band_config(window_screen__bands=bands))

        # band_range's bands are its parameter, checked against the input's
        # bands.
        band = "preprocess standard step 2 'band_range' parameter 'bands' "
        for bands, expected in (([0, 31], "holds band 31; the input has 10 bands"),
                                ([3, 3], "holds band 3 twice"),
                                ([], "is empty"),
                                ("optical", "is 'optical'; a band list is roles, rows or "
                                            "\"reflective\"")):
            self.assert_rejected(self.broken(), 10, band + expected,
                                 config=ten_band_config(preprocess__band_range__bands=bands))

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
            (dict(shared__change_probability=0.0), "shared parameter 'change_probability' is 0; it must be in (0, 1)"),
            (dict(shared__change_probability=1.0), "shared parameter 'change_probability' is 1; it must be in (0, 1)"),
            (dict(shared__change_probability=1.5), "shared parameter 'change_probability' is 1.5"),
            (dict(shared__change_probability=nan), "shared parameter 'change_probability' is nan"),
            (dict(change_test__peek_size=0), chi2 + "'peek_size' is 0; it must be at least 2"),
            (dict(change_test__peek_size=1), chi2 + "'peek_size' is 1; it must be at least 2"),
        ]
        for changes, expected in cases:
            self.assert_rejected(ccdcpp.Input(), 7, expected, config=preset(**changes))


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
        ("shared", "change_probability"): 0.99,
    }

    UNITS = {
        ("change_test", "peek_reference_days"): "days",
        ("noise_floor", "min_lag_days"): "days",
        ("refit", "min_obs"): "observations",
        ("refit", "factor"): "none",
        ("change_test", "seasonal_count"): "observations",
        ("change_test", "seasonal_divisor"): "none",
        ("shared", "change_probability"): "none",
    }

    # The Input's fields: the data's band layout, QA codes and units.
    INPUT_FIELDS = {
        "roles", "bands", "index", "digest", "replace", "reflectance_scale",
        "reflectance_offset", "saturation_code", "thermal_saturation_code",
        "qa_fill", "qa_clear", "qa_water", "qa_shadow", "qa_snow", "qa_cloud",
        "qa_cirrus1", "qa_cirrus2", "qa_occlusion", "qa_table",
        "thermal_scale", "thermal_offset", "thermal_at_100c",
    }

    def value(self, tree, stage, name):
        return tree["shared"][name] if stage == "shared" else tree["stages"][stage]["params"][name]

    def test_defaults(self):
        tree = preset()
        for (stage, name), value in self.DEFAULTS.items():
            with self.subTest(stage=stage, name=name):
                self.assertEqual(self.value(tree, stage, name), value)
                self.assertIs(type(self.value(tree, stage, name)), type(value))

    def test_the_registry_states_unit_and_default(self):
        table = ccdcpp._strategies()
        shared = {v["name"]: v for v in ccdcpp._shared()}
        tree = preset()
        for (stage, name), unit in self.UNITS.items():
            with self.subTest(stage=stage, name=name):
                if stage == "shared":
                    param = shared[name]
                else:
                    (param,) = [p for p in table[stage][tree["stages"][stage]["strategy"]]
                                if p["name"] == name]
                self.assertEqual(param["unit"], unit)
                self.assertTrue(param["doc"])
                self.assertEqual(param["default"], self.DEFAULTS[(stage, name)])

    def test_the_input_describes_the_data_only(self):
        # The algorithm's parameters are the configuration's; the Input holds
        # the data's fields and takes no others.
        h = ccdcpp.Input()
        self.assertEqual({n for n in dir(h) if not n.startswith("_")}, self.INPUT_FIELDS)
        tree = preset()
        names = set(tree["shared"])
        for entry in tree["stages"].values():
            names |= set(entry["params"])
        for name in names:
            with self.subTest(name=name):
                with self.assertRaises(AttributeError):
                    setattr(h, name, 1)

    def test_chi2_quantile_is_the_change_threshold(self):
        tree = resolved()
        for stage in ("change_test", "stability_rule", "lookback"):
            with self.subTest(stage=stage):
                params = tree["stages"][stage]["params"]
                # To the last bit.
                self.assertEqual(
                    ccdcpp.chi2_quantile(0.99, 5).hex(), params["change_threshold"].hex())
                self.assertEqual(
                    ccdcpp.chi2_quantile(p=params["change_probability"],
                                         df=len(tree["shared"]["detection_bands"])),
                    params["change_threshold"])
        self.assertGreater(ccdcpp.chi2_quantile(0.99, 8), ccdcpp.chi2_quantile(0.99, 5))
        self.assertGreater(ccdcpp.chi2_quantile(0.999, 5), ccdcpp.chi2_quantile(0.99, 5))
        self.assertIn("same boost call", ccdcpp.chi2_quantile.__doc__)

    def test_defaults_spelled_out_equal_default(self):
        cfg = preset(**{f"{stage}__{name}": value
                        for (stage, name), value in self.DEFAULTS.items()})
        for args in (noisy(7, 120), noisy(7, 400, seed=3)):
            self.assertEqual(
                repr(ccdcpp.detect(*args, encoding=ccdcpp.Input(), config=cfg)),
                repr(ccdcpp.detect(*args, **options())))

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
            out = ccdcpp.detect(d, s, q, encoding=ccdcpp.Input(), config=config)
            return sum(m["change_probability"] for m in out["models"])

        self.assertEqual(breaks(None), 1)
        self.assertEqual(breaks(preset(change_test__peek_reference_days=32.0)), 0)


class BoundFlags(unittest.TestCase):
    """The bounds' inclusive flags: band_range's and thermal_range's parameters."""

    FLAGS = {
        ("band_range", "min_inclusive"): False,
        ("band_range", "max_inclusive"): False,
        ("thermal_range", "min_inclusive"): False,
        ("thermal_range", "max_inclusive"): False,
    }

    def test_defaults_and_docstrings(self):
        steps = ccdcpp._steps()
        for (step, name), value in self.FLAGS.items():
            with self.subTest(step=step, name=name):
                (param,) = [p for p in steps[step] if p["name"] == name]
                self.assertIs(param["default"], value)
                self.assertTrue(param["doc"])
                for chain in preset()["preprocess"].values():
                    for s in chain["steps"]:
                        if s["step"] == step:
                            self.assertIs(s[name], value)

    def mask(self, values, band, **flags):
        """processing_mask of detect on a pixel whose `band` holds `values` at
        its first observations, with the steps' flags (step__name=value)."""
        d, s, q = inputs(7, 40)
        s[band, :len(values)] = values
        config = preset(**{"preprocess__" + k: v for k, v in flags.items()})
        return ccdcpp.detect(d, s, q, encoding=ccdcpp.Input(),
                             config=config)["processing_mask"][:len(values)].tolist()

    def test_reflectance_bounds(self):
        values = [0.0, 10000.0, -1.0, 10001.0]
        for flags, expected in (
            (dict(), [0, 0, 0, 0]),
            (dict(band_range__min_inclusive=True, band_range__max_inclusive=True),
             [1, 1, 0, 0]),
            (dict(band_range__min_inclusive=True), [1, 0, 0, 0]),
            (dict(band_range__max_inclusive=True), [0, 1, 0, 0]),
        ):
            with self.subTest(**flags):
                self.assertEqual([int(v) for v in self.mask(values, 3, **flags)], expected)

    def test_thermal_bounds(self):
        # detect transforms thermal K x 10 -> Celsius x 100 first: k10 * 10 - 27315.
        at = lambda celsius100: (celsius100 + 27315.0) / 10.0  # noqa: E731
        values = [at(-9320.0), at(7070.0), at(0.0)]
        for flags, expected in (
            (dict(), [0, 0, 1]),
            (dict(thermal_range__min_inclusive=True, thermal_range__max_inclusive=True),
             [1, 1, 1]),
            (dict(thermal_range__min_inclusive=True), [1, 0, 1]),
        ):
            with self.subTest(**flags):
                self.assertEqual([int(v) for v in self.mask(values, 6, **flags)], expected)

    def test_exclusive_bounds_reject_nan(self):
        # An exclusive end rejects NaN, so the pixel fits without it. (Both ends
        # inclusive let NaN through to the fit; see the README.)
        for band, flags in ((3, dict()),
                            (3, dict(band_range__min_inclusive=True)),
                            (3, dict(band_range__max_inclusive=True)),
                            (6, dict())):
            with self.subTest(band=band, **flags):
                self.assertEqual(self.mask([np.nan, 1000.0 if band != 6 else 2900.0], band, **flags),
                                 [0, 1])


class GreenFilterScope(unittest.TestCase):
    STATISTICS_END = 731000

    NAMES = {"All": "all", "StatWindow": "stat_window", "Off": "off"}

    def test_default_and_docstring(self):
        green = preset()["preprocess"]["insufficient_clear"]["steps"][3]
        self.assertEqual((green["step"], green["scope"]), ("green_median", "stat_window"))
        self.assertEqual(sorted(ccdcpp.GreenFilterScope.__members__), ["All", "Off", "StatWindow"])
        self.assertIn("statistics_end", ccdcpp.GreenFilterScope.__doc__)
        self.assertIn('"stat_window"', ccdcpp.GreenFilterScope.__doc__)

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
        config = preset(shared__statistics_end=self.STATISTICS_END,
                        preprocess__green_median__scope=self.NAMES[scope.name])
        out = ccdcpp.detect(*pixel, encoding=ccdcpp.Input(), config=config)
        self.assertEqual({m["curve_qa"] for m in out["models"]}, {44})
        return out["processing_mask"]

    def test_spikes_before_and_after_statistics_end(self):
        pixel = self.pixel(self.STATISTICS_END - 16 * 99)   # t = 99 is statistics_end
        for scope, kept in ((ccdcpp.GreenFilterScope.All, (0, 0)),
                            (ccdcpp.GreenFilterScope.StatWindow, (0, 1)),
                            (ccdcpp.GreenFilterScope.Off, (1, 1))):
            with self.subTest(scope=scope):
                mask = self.detect_mask(pixel, scope)
                self.assertEqual((mask[10], mask[150]), kept)
                self.assertEqual(int(mask.sum()), 38 + sum(kept))

    def test_every_date_after_statistics_end(self):
        pixel = self.pixel(self.STATISTICS_END + 1)
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
                    ccdcpp.detect(*inputs(7, 40, dates=dates, qas=qas), **options())

    def test_cubes(self):
        for dates, qas in ((39, None), (41, None), (None, 39), (None, 41)):
            with self.subTest(dates=dates, qas=qas):
                with self.assertRaises(RuntimeError):
                    ccdcpp.detect_cube(*cube(7, 40, dates=dates, qas=qas), **options(), threads=1)


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
                    ccdcpp.detect(self.swapped(d, at), s, q, **options())
                self.assertIn("dates must be non-decreasing", str(caught.exception))
                self.assertIn(f"dates[{at + 1}] = {d[at]} follows dates[{at}] = {d[at + 1]}",
                              str(caught.exception))

    def test_raw_qa(self):
        d, s, _ = inputs(7, 40)
        raw = np.zeros(40, dtype=np.uint16)
        h = ccdcpp.Input(qa_table=ccdcpp.qa_tables.landsat_c2_qa_pixel())
        with self.assertRaises(ValueError):
            ccdcpp.detect(self.swapped(d, 5), s, raw, encoding=h)

    def test_cubes(self):
        for dtype in (np.float64, np.float32, np.uint16):
            with self.subTest(dtype=dtype.__name__):
                d, spectral, q = cube(7, 40, dtype=dtype)
                with self.assertRaises(ValueError) as caught:
                    ccdcpp.detect_cube(self.swapped(d, 7), spectral, q, **options(), threads=1)
                self.assertIn("dates must be non-decreasing", str(caught.exception))

    def test_equal_dates_are_deduplicated(self):
        d, s, q = inputs(7, 40)
        d[11] = d[12] = d[10]
        mask = ccdcpp.detect(d, s, q, **options())["processing_mask"]
        self.assertEqual(mask[10:13].tolist(), [1, 0, 0])
        _, spectral, qc = cube(7, 40)
        out = ccdcpp.detect_cube(d, spectral, qc, **options(), threads=1)
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
