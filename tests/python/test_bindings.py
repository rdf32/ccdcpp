"""Python entry points: input validation, the band layout, the cadence parameters
and __version__.

usage: python test_bindings.py <module_dir> <source_dir> [unittest args]

<module_dir> holds the built extension (ccdcpp.*.so / .pyd); <source_dir> is
the repository root. ctest runs this as `python_bindings`.
"""
from __future__ import annotations

import os
import tomllib
import unittest
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypedDict

from _harness import SOURCE_DIR, use_package

use_package()

import numpy as np  # noqa: E402
from numpy.typing import DTypeLike, NDArray  # noqa: E402

# The extension module, which ccdcpp/ccdcpp.pyi types.
from ccdcpp import ccdcpp as ext  # noqa: E402

START = 724276  # 1984-01-01


Series = tuple[NDArray[np.int64], NDArray[Any], NDArray[np.uint8]]


def inputs(bands: int, timesteps: int, dates: int | None = None,
           qas: int | None = None) -> Series:
    """(dates, spectral (B, T), qas) of in-range DN, clear QA."""
    d = START + 16 * np.arange(timesteps if dates is None else dates, dtype=np.int64)
    s = np.empty((bands, timesteps), dtype=np.float64)
    s[:] = (1000.0 + 100.0 * np.arange(bands))[:, None]
    if bands > 6:
        s[6] = 2900.0
    q = np.zeros(timesteps if qas is None else qas, dtype=np.uint8)
    return d, s, q


def cube(bands: int, timesteps: int, dtype: DTypeLike = np.float64, dates: int | None = None,
         qas: int | None = None) -> Series:
    """(dates, spectral (2, 3, B, T), qas (2, 3, T))."""
    d, s, _ = inputs(bands, timesteps, dates=dates)
    spectral = np.broadcast_to(s, (2, 3, bands, timesteps)).astype(dtype)
    q = np.zeros((2, 3, timesteps if qas is None else qas), dtype=np.uint8)
    return d, spectral, q


class Options(TypedDict):
    input: ext.Input


def options() -> Options:
    """The default input, as the entry points' keyword."""
    return {"input": ext.Input()}


def preset(**changes: object) -> dict[str, Any]:
    """The built-in ccdc preset as the extension takes it (a configuration's dict, laid out
    as its TOML file), with shared__name=value, stage__param=value and
    preprocess__step__param=value (every step of that type) changes."""
    tree = ext._config_preset("ccdc")
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
        assert param in tree[stage], key
        tree[stage][param] = value
    return tree


def algorithm(**changes: object) -> ext.Algorithm:
    """preset(**changes) as the extension's config= takes it."""
    return ext.Algorithm(preset(**changes))


def resolved(config: Mapping[str, Any] | None = None,
             input: ext.Input | None = None) -> dict[str, Any]:
    """The configuration resolved against the input (default Input())."""
    return ext._config_resolve(preset() if config is None else config,
                               ext.Input() if input is None else input)


class BandCount(unittest.TestCase):
    def assert_band_error(self, call: Callable[[], object], bands: int) -> None:
        with self.assertRaises(ValueError) as caught:
            call()
        message = str(caught.exception)
        self.assertIn(f"has {bands} bands", message)
        self.assertIn("the data spec has 7", message)

    def test_detect(self):
        for bands in (6, 8):
            with self.subTest(bands=bands):
                self.assert_band_error(
                    lambda: ext.detect(*inputs(bands, 40), **options()), bands)
        ext.detect(*inputs(7, 40), **options())

    def test_detect_cube(self):
        for dtype in (np.float64, np.float32, np.uint16):
            for bands in (6, 8):
                with self.subTest(dtype=dtype, bands=bands):
                    self.assert_band_error(
                        lambda: ext.detect_cube(
                            *cube(bands, 40, dtype), **options(), threads=1),
                        bands)
        out = ext.detect_cube(*cube(7, 40), **options(), threads=1)
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
                out = ext.detect_cube(d, spectral, qas, **options(), threads=threads,
                                      masks=True)
                assert "processing_mask" in out
                mask = out["processing_mask"]
                self.assertEqual((mask.dtype, mask.shape), (np.uint8, (H, W, T)))
                # The run's own (H, W, T) array, not a view of another ndarray.
                self.assertTrue(mask.flags.c_contiguous and mask.flags.writeable)
                self.assertNotIsInstance(mask.base, np.ndarray)
                self.assertFalse(mask[1, 2].any())
                for r in range(H):
                    for c in range(W):
                        one = ext.detect(d, spectral[r, c], qas[r, c], **options())
                        np.testing.assert_array_equal(mask[r, c], one["processing_mask"])
                # The columns are those without masks.
                plain = ext.detect_cube(d, spectral, qas, **options(), threads=threads)
                self.assertEqual(set(out) - set(plain), {"processing_mask"})
                for key in plain:
                    if hasattr(plain[key], "dtype"):
                        np.testing.assert_array_equal(out[key], plain[key])


class PixelLayouts(unittest.TestCase):
    def test_every_dtype_and_layout_is_the_float64_run(self):
        d, s, q = noisy(7, 120, seed=3)
        q = q.copy()
        q[::7] = 4
        base = ext.detect(d, s, q, **options())
        self.assertTrue(base["models"])
        wide = np.zeros((7, 240))
        wide[:, ::2] = s
        qs = np.zeros(240, dtype=np.uint8)
        qs[::2] = q
        forms = {
            "float32": (s.astype(np.float32), q),
            "uint16": (s.astype(np.uint16), q),
            "transposed": (np.asfortranarray(s), q),
            "strided": (wide[:, ::2], qs[::2]),
            # Converted before the run: negative strides, another byte order, int64.
            "reversed": (np.ascontiguousarray(s[:, ::-1])[:, ::-1],
                         np.ascontiguousarray(q[::-1])[::-1]),
            "byte-swapped": (s.astype(s.dtype.newbyteorder()), q),
            "int64": (s.astype(np.int64), q),
        }
        for name, (spectral, qas) in forms.items():
            with self.subTest(form=name):
                out = ext.detect(d, spectral, qas, **options())
                self.assertEqual(repr(out["models"]), repr(base["models"]))
                np.testing.assert_array_equal(out["processing_mask"], base["processing_mask"])


TEN_ROLES = ["blue","green", "red", "red_edge_1", "red_edge_2", "red_edge_3", "nir",
             "other", "swir1", "swir2"]


def ten_band_layout() -> ext.Input:
    """A 10-band layout without thermal, which passes validation: the
    configuration's detection bands are rows 1, 2, 6, 8, 9 of it, its Tmask
    bands 1 and 8."""
    return ext.Input(roles=TEN_ROLES)


def noisy(bands: int, timesteps: int, seed: int = 1) -> Series:
    """inputs() with noise, so every band gets a non-trivial fit."""
    d, s, q = inputs(bands, timesteps)
    s = s + np.random.default_rng(seed).normal(0.0, 20.0, s.shape).round()
    return d, s, q


class BandLayout(unittest.TestCase):
    def test_the_default_input_is_landsats_layout(self):
        h = ext.Input()
        self.assertEqual(h.bands(), 7)
        self.assertEqual(h.roles, ["blue", "green", "red", "nir", "swir1", "swir2", "thermal"])
        self.assertEqual(preset()["shared"]["detection_bands"],
                         ["green", "red", "nir", "swir1", "swir2"])
        self.assertEqual(preset()["window_screen"]["bands"],
                         ["green", "swir1"])
        self.assertEqual(resolved()["shared"]["detection_bands"], [1, 2, 3, 4, 5])
        self.assertEqual(resolved()["window_screen"]["bands"], [1, 4])
        band_range = [s for s in preset()["preprocess"]["standard"]["steps"]
                      if s["step"] == "band_range"]
        self.assertEqual([s["bands"] for s in band_range],
                         [["blue", "green", "red", "nir", "swir1", "swir2"]])
        # The millworks nlcd plugin checks these.
        self.assertEqual([h.index(r) for r in h.roles], [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(h.index("coastal"), ext.NO_BAND)
        self.assertEqual(
            [h.qa_fill, h.qa_clear, h.qa_water, h.qa_shadow, h.qa_snow,
             h.qa_cloud, h.qa_cirrus1, h.qa_cirrus2, h.qa_occlusion],
            [255, 0, 1, 2, 3, 4, 8, 9, 10])
        self.assertEqual(h.reflectance_scale, 10000.0)
        self.assertEqual(ext.NO_BAND, 2**64 - 1)
        self.assertEqual(ext.CCD_MAX_BANDS, 32)

    def test_immutable_by_keyword(self):
        h = ext.Input(reflectance_scale=1.0, roles=["blue", "green", "red"], saturation_code=None)
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
        self.assertEqual(ext.Input(), ext.Input().replace())
        # Keywords only, each a field, each of its type.
        with self.assertRaises(TypeError):
            ext.Input(1.0)  # pyright: ignore[reportCallIssue]  # positional, refused
        with self.assertRaisesRegex(TypeError, "Input\\(\\) has no field 'alpha'; the fields are qa_fill"):
            ext.Input(alpha=1.0)  # pyright: ignore[reportCallIssue]  # no such field
        with self.assertRaisesRegex(TypeError, "Input.replace\\(\\) has no field 'saturation_bands'"):
            h.replace(saturation_bands=[0])  # pyright: ignore[reportCallIssue]  # no such field
        with self.assertRaisesRegex(TypeError, "qa_clear is 300, which it cannot hold"):
            ext.Input(qa_clear=300)
        with self.assertRaisesRegex(TypeError, "roles is 'green'"):
            ext.Input(roles="green")

    def test_informational_codes_say_so(self):
        for name in ("qa_cirrus1", "qa_cirrus2", "qa_occlusion"):
            with self.subTest(name=name):
                self.assertIn("Informational", getattr(ext.Input, name).__doc__)
        for name in ("qa_fill", "qa_clear", "qa_water", "qa_shadow", "qa_snow", "qa_cloud"):
            with self.subTest(name=name):
                self.assertNotIn("Informational", getattr(ext.Input, name).__doc__)

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
        h = ext.Input(roles=["blue", "green", "red", "nir", "swir1", "swir2", "thermal"])
        args = noisy(7, 120)
        by_rows = algorithm(window_screen__bands=[1, 4], shared__detection_bands=[1, 2, 3, 4, 5],
                            preprocess__band_range__bands=[0, 1, 2, 3, 4, 5])
        self.assertEqual(
            repr(ext.detect(*args, input=h, config=by_rows)),
            repr(ext.detect(*args, **options())))

    def test_outputs_follow_the_band_count(self):
        h, cfg = ten_band_layout(), ext.Algorithm(preset())
        d, s, q = noisy(10, 120)
        out = ext.detect(d, s, q, input=h, config=cfg)
        self.assertEqual(out["num_bands"], 10)
        self.assertGreater(len(out["models"]), 0)
        for m in out["models"]:
            self.assertEqual(len(m["bands"]), 10)

        self.assertEqual(ext.detect(*noisy(7, 120), **options())["num_bands"], 7)

        cube_s = np.broadcast_to(s, (2, 3, 10, 120)).copy()
        cube_q = np.zeros((2, 3, 120), dtype=np.uint8)
        cols = ext.detect_cube(d, cube_s, cube_q, input=h, threads=1, config=cfg)
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

    def broken(self, **fields: Any) -> ext.Input:
        return ten_band_layout().replace(**fields)

    def assert_rejected(self, h: ext.Input, bands: int, expected: str,
                        config: Mapping[str, Any] | None = None) -> None:
        d, s, q = inputs(bands, 40)
        cube_s = np.broadcast_to(s, (2, 3, bands, 40)).copy()
        cube_q = np.zeros((2, 3, 40), dtype=np.uint8)
        # The configuration is built in the call: some refusals are its own.
        def compiled() -> ext.Algorithm | None:
            return None if config is None else ext.Algorithm(config)
        calls = {
            "detect": lambda: ext.detect(d, s, q, input=h, config=compiled()),
            "detect_cube":
                lambda: ext.detect_cube(d, cube_s, cube_q, input=h, threads=1,
                                        config=compiled()),
        }
        for name, call in calls.items():
            with self.subTest(entry=name, expected=expected):
                with self.assertRaises(ValueError) as caught:
                    call()
                self.assertIn(expected, str(caught.exception))

    def test_the_layout_and_band_lists_are_refused_rule_by_rule(self):
        def roles(**at: str) -> list[str]:
            out = list(TEN_ROLES)
            for row, role in at.items():
                out[int(row[1:])] = role
            return out
        cases = [
            (self.broken(), 7, "spectral has 7 bands; the data spec has 10 (a row per role)"),
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
        for bands, expected in (([1, 10], "holds band 10; the data spec has 10 bands"),
                                ([1, 2, 1], "holds band 1 twice"),
                                ([], "is empty"),
                                (["green", "coastal"], "holds 'coastal', which the data spec does "
                                                       "not have")):
            self.assert_rejected(self.broken(), 10, detection + expected,
                                 config=preset(shared__detection_bands=bands))

        # The Tmask bands are window_screen's parameter, checked against the
        # input's bands.
        tmask = "window_screen 'tmask' parameter 'bands' "
        for bands, expected in (([1, 12], "holds band 12; the data spec has 10 bands"),
                                ([8, 8], "holds band 8 twice"),
                                ([], "is empty")):
            self.assert_rejected(self.broken(), 10, tmask + expected,
                                 config=preset(window_screen__bands=bands))

        # band_range's bands are its parameter, checked against the input's
        # bands.
        band = "preprocess standard step 2 'band_range' parameter 'bands' "
        for bands, expected in (([0, 31], "holds band 31; the data spec has 10 bands"),
                                ([3, 3], "holds band 3 twice"),
                                ([], "is empty"),
                                ("optical", "is 'optical'; a band list is roles or rows")):
            self.assert_rejected(self.broken(), 10, band + expected,
                                 config=preset(preprocess__band_range__bands=bands))

    def test_cadence_parameters_are_refused_by_stage_and_parameter(self):
        # The cadence parameters are the configuration's: the schemas refuse
        # them by stage, strategy and parameter.
        nan = float("nan")
        chi2 = "change_test 'chi2' parameter "
        cases = [
            (dict(change_test__peek_reference_days=0.0), chi2 + "'peek_reference_days' is 0; it must be greater than 0"),
            (dict(change_test__peek_reference_days=-16.0), chi2 + "'peek_reference_days' is -16"),
            (dict(change_test__peek_reference_days=nan), chi2 + "'peek_reference_days' is nan"),
            (dict(noise__min_lag_days=0), "noise 'variogram' parameter 'min_lag_days' is 0; it must be greater than 0"),
            (dict(noise__min_lag_days=-30), "noise 'variogram' parameter 'min_lag_days' is -30"),
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
            self.assert_rejected(ext.Input(), 7, expected, config=preset(**changes))


class CadenceParameters(unittest.TestCase):
    """The configuration's cadence and change-probability parameters, and
    ccdcpp.chi2_quantile."""

    # (stage, parameter): the ccdc preset's value.
    DEFAULTS = {
        ("change_test", "peek_reference_days"): 16.0,
        ("noise", "min_lag_days"): 30,
        ("refit", "min_obs"): 24,
        ("refit", "factor"): 1.33,
        ("change_test", "seasonal_count"): 24,
        ("change_test", "seasonal_divisor"): 4.0,
        ("shared", "change_probability"): 0.99,
    }

    UNITS = {
        ("change_test", "peek_reference_days"): "days",
        ("noise", "min_lag_days"): "days",
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

    def value(self, tree: Mapping[str, Any], stage: str, name: str) -> Any:
        return tree["shared"][name] if stage == "shared" else tree[stage][name]

    def test_the_preset_holds_pyccds_values(self):
        tree = preset()
        for (stage, name), value in self.DEFAULTS.items():
            with self.subTest(stage=stage, name=name):
                self.assertEqual(self.value(tree, stage, name), value)
                self.assertIs(type(self.value(tree, stage, name)), type(value))

    def test_the_registry_states_unit_and_default(self):
        table = ext._strategies()
        shared = {v["name"]: v for v in ext._shared()}
        tree = preset()
        for (stage, name), unit in self.UNITS.items():
            with self.subTest(stage=stage, name=name):
                if stage == "shared":
                    param = shared[name]
                else:
                    (param,) = [p for p in table[stage][tree[stage]["strategy"]]
                                if p["name"] == name]
                self.assertEqual(param["unit"], unit)
                self.assertTrue(param["doc"])
                self.assertEqual(param["default"], self.DEFAULTS[(stage, name)])

    def test_the_input_describes_the_data_only(self):
        # The algorithm's parameters are the configuration's; the Input holds
        # the data's fields and takes no others.
        h = ext.Input()
        self.assertEqual({n for n in dir(h) if not n.startswith("_")}, self.INPUT_FIELDS)
        tree = preset()
        names = set(tree["shared"])
        for stage, entry in tree.items():
            if stage not in ("shared", "preprocess"):
                names |= set(entry) - {"strategy"}
        for name in names:
            with self.subTest(name=name):
                with self.assertRaises(AttributeError):
                    setattr(h, name, 1)

    def test_input_equality_and_hash_agree(self):
        # -0 and +0 are one value: equal Inputs hash alike.
        zero = ext.Input(reflectance_offset=0.0, saturation_code=0.0)
        negative = ext.Input(reflectance_offset=-0.0, saturation_code=-0.0)
        self.assertEqual(zero, negative)
        self.assertEqual(hash(zero), hash(negative))
        self.assertEqual(len({zero, negative}), 1)
        self.assertNotEqual(hash(zero), hash(ext.Input(reflectance_offset=1.0)))

    def test_a_nan_saturation_code_is_refused(self):
        nan = float("nan")
        for field, h in (("saturation_code", ext.Input(saturation_code=nan)),
                         ("thermal_saturation_code", ext.Input(thermal_saturation_code=nan))):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, f"^{field} is nan; it is a number"):
                    ext.detect(*noisy(7, 120), input=h)

    def test_chi2_quantile_is_the_change_threshold(self):
        tree = resolved()
        for stage in ("change_test", "stability_rule", "lookback"):
            with self.subTest(stage=stage):
                params = tree[stage]
                # To the last bit.
                self.assertEqual(
                    ext.chi2_quantile(0.99, 5).hex(), params["change_threshold"].hex())
                self.assertEqual(
                    ext.chi2_quantile(p=params["change_probability"],
                                      df=len(tree["shared"]["detection_bands"])),
                    params["change_threshold"])
        self.assertGreater(ext.chi2_quantile(0.99, 8), ext.chi2_quantile(0.99, 5))
        self.assertGreater(ext.chi2_quantile(0.999, 5), ext.chi2_quantile(0.99, 5))
        self.assertIn("same boost call", ext.chi2_quantile.__doc__ or "")

    def test_defaults_spelled_out_equal_default(self):
        cfg = algorithm(**{f"{stage}__{name}": value
                           for (stage, name), value in self.DEFAULTS.items()})
        for args in (noisy(7, 120), noisy(7, 400, seed=3)):
            self.assertEqual(
                repr(ext.detect(*args, input=ext.Input(), config=cfg)),
                repr(ext.detect(*args, **options())))

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

        def breaks(config: ext.Algorithm | None) -> float:
            out = ext.detect(d, s, q, input=ext.Input(), config=config)
            return sum(m["change_probability"] for m in out["models"])

        self.assertEqual(breaks(None), 1)
        self.assertEqual(breaks(algorithm(change_test__peek_reference_days=32.0)), 0)


class BoundFlags(unittest.TestCase):
    """The bounds' inclusive flags: band_range's and thermal_range's parameters."""

    FLAGS = {
        ("band_range", "inclusive"): [False, False],
        ("thermal_range", "inclusive"): [False, False],
    }

    def test_defaults_and_docstrings(self):
        steps = ext._steps()
        for (step, name), value in self.FLAGS.items():
            with self.subTest(step=step, name=name):
                (param,) = [p for p in steps[step] if p["name"] == name]
                self.assertEqual(param["default"], value)
                self.assertTrue(param["doc"])
                for chain in preset()["preprocess"].values():
                    for s in chain["steps"]:
                        if s["step"] == step:
                            self.assertEqual(s[name], value)

    def mask(self, values: Sequence[float], band: int, **flags: object) -> list[int]:
        """processing_mask of detect on a pixel whose `band` holds `values` at
        its first observations, with the steps' flags (step__name=value)."""
        d, s, q = inputs(7, 40)
        s[band, :len(values)] = values
        config = algorithm(**{"preprocess__" + k: v for k, v in flags.items()})
        return ext.detect(d, s, q, input=ext.Input(),
                          config=config)["processing_mask"][:len(values)].tolist()

    def test_reflectance_bounds(self):
        values = [0.0, 10000.0, -1.0, 10001.0]
        for flags, expected in (
            (dict(), [0, 0, 0, 0]),
            (dict(band_range__inclusive=[True, True]), [1, 1, 0, 0]),
            (dict(band_range__inclusive=[True, False]), [1, 0, 0, 0]),
            (dict(band_range__inclusive=[False, True]), [0, 1, 0, 0]),
        ):
            with self.subTest(flags=flags):
                self.assertEqual([int(v) for v in self.mask(values, 3, **flags)], expected)

    def test_thermal_bounds(self):
        # detect transforms thermal K x 10 -> Celsius x 100 first: k10 * 10 - 27315.
        at = lambda celsius100: (celsius100 + 27315.0) / 10.0  # noqa: E731
        values = [at(-9320.0), at(7070.0), at(0.0)]
        for flags, expected in (
            (dict(), [0, 0, 1]),
            (dict(thermal_range__inclusive=[True, True]), [1, 1, 1]),
            (dict(thermal_range__inclusive=[True, False]), [1, 0, 1]),
        ):
            with self.subTest(flags=flags):
                self.assertEqual([int(v) for v in self.mask(values, 6, **flags)], expected)

    def test_exclusive_bounds_reject_nan(self):
        # An exclusive end rejects NaN, so the pixel fits without it. (Both ends
        # inclusive let NaN through to the fit; see the README.)
        for band, flags in ((3, dict()),
                            (3, dict(band_range__inclusive=[True, False])),
                            (3, dict(band_range__inclusive=[False, True])),
                            (6, dict())):
            with self.subTest(band=band, flags=flags):
                self.assertEqual(self.mask([np.nan, 1000.0 if band != 6 else 2900.0], band, **flags),
                                 [0, 1])


class GreenFilterScope(unittest.TestCase):
    STATISTICS_END = 731000

    NAMES = {"All": "all", "StatWindow": "stat_window", "Off": "off"}

    def test_default_and_docstring(self):
        green = preset()["preprocess"]["insufficient_clear"]["steps"][4]
        self.assertEqual((green["step"], green["scope"]), ("green_median", "stat_window"))
        self.assertEqual(sorted(ext.GreenFilterScope.__members__), ["All", "Off", "StatWindow"])
        self.assertIn("statistics_end", ext.GreenFilterScope.__doc__ or "")
        self.assertIn('"stat_window"', ext.GreenFilterScope.__doc__ or "")

    def pixel(self, first: int) -> Series:
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

    def detect_mask(self, pixel: Series, scope: ext.GreenFilterScope) -> NDArray[np.uint8]:
        config = algorithm(shared__statistics_end=self.STATISTICS_END,
                           preprocess__green_median__scope=self.NAMES[scope.name])
        out = ext.detect(*pixel, input=ext.Input(), config=config)
        self.assertEqual({m["curve_qa"] for m in out["models"]}, {44})
        return out["processing_mask"]

    def test_spikes_before_and_after_statistics_end(self):
        pixel = self.pixel(self.STATISTICS_END - 16 * 99)   # t = 99 is statistics_end
        for scope, kept in ((ext.GreenFilterScope.All, (0, 0)),
                            (ext.GreenFilterScope.StatWindow, (0, 1)),
                            (ext.GreenFilterScope.Off, (1, 1))):
            with self.subTest(scope=scope):
                mask = self.detect_mask(pixel, scope)
                self.assertEqual((mask[10], mask[150]), kept)
                self.assertEqual(int(mask.sum()), 38 + sum(kept))

    def test_every_date_after_statistics_end(self):
        pixel = self.pixel(self.STATISTICS_END + 1)
        masks = [self.detect_mask(pixel, scope) for scope in (
            ext.GreenFilterScope.All, ext.GreenFilterScope.StatWindow,
            ext.GreenFilterScope.Off)]
        for mask in masks:
            np.testing.assert_array_equal(mask, masks[2])
        self.assertEqual((masks[0][10], masks[0][150]), (1, 1))


class TimestepMismatch(unittest.TestCase):
    def test_detect_refuses_dates_or_qas_of_another_length(self):
        for dates, qas in ((39, None), (41, None), (None, 39), (None, 41)):
            with self.subTest(dates=dates, qas=qas):
                with self.assertRaises(ValueError):
                    ext.detect(*inputs(7, 40, dates=dates, qas=qas), **options())

    def test_detect_cube_refuses_dates_or_qas_of_another_length(self):
        for dates, qas in ((39, None), (41, None), (None, 39), (None, 41)):
            with self.subTest(dates=dates, qas=qas):
                with self.assertRaises(ValueError):
                    ext.detect_cube(*cube(7, 40, dates=dates, qas=qas), **options(), threads=1)


class DateOrder(unittest.TestCase):
    def swapped(self, d: NDArray[np.int64], at: int) -> NDArray[np.int64]:
        d = d.copy()
        d[at], d[at + 1] = d[at + 1], d[at]
        return d

    def test_detect(self):
        d, s, q = inputs(7, 40)
        for at in (0, 20, 38):
            with self.subTest(at=at):
                with self.assertRaises(ValueError) as caught:
                    ext.detect(self.swapped(d, at), s, q, **options())
                self.assertIn("dates must be non-decreasing", str(caught.exception))
                self.assertIn(f"dates[{at + 1}] = {d[at]} follows dates[{at}] = {d[at + 1]}",
                              str(caught.exception))

    def test_raw_qa(self):
        d, s, _ = inputs(7, 40)
        raw = np.zeros(40, dtype=np.uint16)
        h = ext.Input(qa_table=ext.qa_tables.landsat_c2_qa_pixel())
        with self.assertRaises(ValueError):
            ext.detect(self.swapped(d, 5), s, raw, input=h)

    def test_cubes(self):
        for dtype in (np.float64, np.float32, np.uint16):
            with self.subTest(dtype=dtype.__name__):
                d, spectral, q = cube(7, 40, dtype=dtype)
                with self.assertRaises(ValueError) as caught:
                    ext.detect_cube(self.swapped(d, 7), spectral, q, **options(), threads=1)
                self.assertIn("dates must be non-decreasing", str(caught.exception))

    def test_equal_dates_are_deduplicated(self):
        d, s, q = inputs(7, 40)
        d[11] = d[12] = d[10]
        mask = ext.detect(d, s, q, **options())["processing_mask"]
        self.assertEqual(mask[10:13].tolist(), [1, 0, 0])
        _, spectral, qc = cube(7, 40)
        out = ext.detect_cube(d, spectral, qc, **options(), threads=1)
        self.assertEqual(out["failures"], [])


class Version(unittest.TestCase):
    def test_extension_matches_pyproject(self):
        with open(os.path.join(SOURCE_DIR, "pyproject.toml"), "rb") as f:
            self.assertEqual(ext.__version__, tomllib.load(f)["project"]["version"])


if __name__ == "__main__":
    unittest.main()
