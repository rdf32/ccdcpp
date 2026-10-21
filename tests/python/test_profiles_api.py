"""SensorProfile: its checks, the roles, its immutability, the inputs it builds, and
the entry points without one.

usage: python test_profiles_api.py <module_dir> <source_dir> [unittest args]

<module_dir> holds the built extension; <source_dir> is the repository root.
The package is assembled as the wheel lays it out (_harness.use_package()).
ctest runs this as `python_profiles_api`.
"""
from __future__ import annotations

import copy
import dataclasses
import inspect
import pickle
import pydoc
import unittest
from typing import Any

from _harness import use_package

use_package()

import numpy as np  # noqa: E402

import ccdcpp  # noqa: E402
# The extension module, which ccdcpp/ccdcpp.pyi types.
from ccdcpp import ccdcpp as ext  # noqa: E402
from ccdcpp import presets  # noqa: E402

from _compare import assert_same  # noqa: E402
from _corpus import (CODES, CUBE_DN, CUBE_PHYS, DATES, TEN_BAND_CONFIG, every_entry,  # noqa: E402
                     strip_band_names, ten_band_profile, without)
from _profiles import (C2_RAW, FMASK_CODES, FMASK_VALUES, SCL_ALL, SCL_CODES,  # noqa: E402
                       band_range_rows, change_threshold, fields, option_params,
                       outlier_threshold, preset_inputs, qa_table_of, resolved, with_table)


class Check(unittest.TestCase):
    def test_band_count(self):
        profile = presets.landsat_c1_dn()
        profile.check(np.zeros((7, 10)))
        profile.check(np.zeros((2, 3, 7, 10)))
        for shape in ((6, 10), (8, 10), (2, 3, 6, 10), (2, 3, 10, 7)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError) as caught:
                    profile.check(np.zeros(shape))
                message = str(caught.exception)
                self.assertIn("'landsat_c1_dn'", message)
                self.assertIn(f"has {shape[-2]} bands", message)
                self.assertIn("the profile has 7: blue, green", message)
        with self.assertRaises(ValueError) as caught:
            profile.check(np.zeros(10))
        self.assertIn("no band axis", str(caught.exception))

    def test_entry_points_check_first(self):
        d, s, q = DATES, CUBE_DN[:, :, :6], CODES
        profile = presets.landsat_c1_dn()
        calls = {
            "detect": lambda: ccdcpp.detect(d, s[0, 0], q[0, 0], input=profile),
            "detect_cube": lambda: ccdcpp.detect_cube(d, s, q, input=profile),
        }
        for name, call in calls.items():
            with self.subTest(entry=name):
                with self.assertRaises(ValueError) as caught:
                    call()
                self.assertIn("SensorProfile 'landsat_c1_dn'", str(caught.exception))

    def test_not_a_profile(self):
        with self.assertRaises(TypeError):
            ccdcpp.detect(DATES, CUBE_DN[0, 0], CODES[0, 0],
                          input="landsat_c1_dn")  # pyright: ignore[reportArgumentType]  # a name, refused

    def test_an_inconsistent_profile_is_refused(self):
        base = presets.landsat_c1_dn()
        cases = [
            (dict(roles=base.roles[:6]), "7 band_names and 6 roles"),
            (dict(band_names=("a",) * 7), "band_names repeats"),
            (dict(roles=("blue",) * 7), "repeats a role other than 'other'"),
            (dict(roles=base.roles[:6] + ("infrared",)), "unknown roles ['infrared']"),
            (dict(roles=base.roles[:6] + ("nir_narrow",)), "unknown roles ['nir_narrow']"),
            (dict(roles=("blue", "other") + base.roles[2:]), "no 'green' band"),
            (dict(thermal=None), "a 'thermal' band and thermal= go together"),
            (dict(qa_table_builder="landsat_c3"), "no builder 'landsat_c3'"),
            (dict(band_names=tuple(map(str, range(33))), roles=("green",) + ("other",) * 32),
             "has 33 bands"),
        ]
        for change, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaises(ValueError) as caught:
                    dataclasses.replace(base, **change)
                self.assertIn(expected, str(caught.exception))
        # An 'other' band is fitted, and no list can name it.
        other = dataclasses.replace(base, roles=base.roles[:6] + ("other",), thermal=None)
        self.assertEqual(other.input().bands(), 7)
        self.assertEqual(band_range_rows(other), [0, 1, 2, 3, 4, 5])
        with self.assertRaises(ValueError) as caught:
            ccdcpp.plan_of(ccdcpp.config("ccdc").replace(
                preprocess__band_range__bands=["green", "other"]), other)
        self.assertIn("holds 'other', which no band list names", str(caught.exception))


class Roles(unittest.TestCase):
    def test_roles(self):
        self.assertEqual(ccdcpp.ROLES, (
            "blue", "green", "red", "nir", "swir1", "swir2", "thermal", "coastal",
            "red_edge_1", "red_edge_2", "red_edge_3", "nir_broad", "other"))

    def test_nir_broad_is_a_band_role(self):
        # B08 as nir_broad next to B8A as nir: it is fitted, reported and may
        # be listed ("ccdc"'s band_range does not), and it is not the "nir"
        # row.
        p = ten_band_profile()
        h = p.input()
        self.assertEqual(p.index("nir_broad"), 6)
        self.assertEqual(h.index("nir"), 7)
        self.assertNotIn(6, band_range_rows(h))
        listed = ccdcpp.config("ccdc").replace(preprocess__band_range__bands=["blue", "nir_broad"])
        self.assertEqual(band_range_rows(h, listed), [0, 6])
        cfg = ccdcpp.config("ccdc").replace(
            shared__detection_bands=["green", "red", "nir_broad", "nir", "swir1", "swir2"])
        self.assertEqual(cfg.resolve(h)["shared"]["detection_bands"], [1, 2, 6, 7, 8, 9])
        with self.assertRaises(ValueError) as caught:
            dataclasses.replace(p, roles=p.roles[:7] + ("nir_broad",) + p.roles[8:])
        self.assertIn("repeats a role other than 'other'", str(caught.exception))


class Unprofiled(unittest.TestCase):
    """With an Input the package entry points are the extension's."""

    def test_same_output_every_call_form(self):
        h = ccdcpp.Input()
        d, s, q = DATES, CUBE_DN, CODES
        expected = without(ext.detect_cube(d, s, q, input=h, threads=1), "threads")
        forms = {
            # The millworks nlcd plugin's call.
            "keywords": lambda: ccdcpp.detect_cube(
                dates=d, spectral=s, qas=q, input=h, threads=1),
            "positional": lambda: ccdcpp.detect_cube(d, s, q, input=h, threads=1),
            # "ccdc" is the run without config=.
            "config='ccdc'": lambda: ccdcpp.detect_cube(d, s, q, input=h, threads=1,
                                                        config="ccdc"),
            "config=Config": lambda: ccdcpp.detect_cube(d, s, q, input=h, threads=1,
                                                        config=ccdcpp.config("ccdc")),
        }
        for name, call in forms.items():
            with self.subTest(form=name):
                assert_same(self, without(call(), "threads"), expected)
        assert_same(self, ccdcpp.detect(d, s[0, 0], q[0, 0], input=h),
                    ext.detect(d, s[0, 0], q[0, 0], input=h))
        raw = np.vectorize(C2_RAW.get, otypes=[np.uint16])(q)
        decoding = with_table(h, ccdcpp.qa_tables.landsat_c2_qa_pixel())
        assert_same(self, ccdcpp.detect(d, s[0, 0], raw[0, 0], input=decoding),
                    ext.detect(d, s[0, 0], raw[0, 0], input=decoding))

    def test_signatures(self):
        positional = ["dates", "spectral", "qas"]
        for name, keywords in (
            ("detect", ["input", "config", "record", "resume", "band_names", "trace",
                        "reasons", "diagnostics"]),
            ("detect_cube", ["input", "config", "threads", "record", "resume", "band_names",
                             "trace_pixels", "masks", "reasons", "diagnostics"]),
        ):
            with self.subTest(entry=name):
                params = inspect.signature(getattr(ccdcpp, name)).parameters
                self.assertEqual(list(params), positional + keywords)
                kinds = [p.kind for p in params.values()]
                self.assertEqual(kinds, [inspect.Parameter.POSITIONAL_OR_KEYWORD] * len(positional)
                                 + [inspect.Parameter.KEYWORD_ONLY] * len(keywords))
                self.assertIs(params["input"].default, inspect.Parameter.empty)
                self.assertEqual(params["config"].default, "ccdc")
                self.assertIs(params["record"].default, False)
                self.assertIsNone(params["resume"].default)
                self.assertIsNone(params["band_names"].default)
                if "threads" in params:
                    self.assertEqual(params["threads"].default, 0)
        for gone in ("detect_cube_columns", "resume", "resume_cube_columns"):
            self.assertFalse(hasattr(ccdcpp, gone), gone)

    def test_config_is_keyword_only_and_defaults_to_ccdc(self):
        h = ccdcpp.Input()
        d, s, q = DATES, CUBE_DN, CODES
        with self.assertRaises(TypeError):
            ccdcpp.detect(d, s[0, 0], q[0, 0], h)  # pyright: ignore[reportCallIssue]  # input positional, refused
        with self.assertRaisesRegex(ValueError, "no configuration 'cdcc'"):
            ccdcpp.detect(d, s[0, 0], q[0, 0], input=h, config="cdcc")
        with self.assertRaises(TypeError):
            ccdcpp.detect(d, s[0, 0], q[0, 0], input=h,
                          config=ccdcpp.config("ccdc").to_dict())  # pyright: ignore[reportArgumentType]  # a dict, refused

    def test_input_required(self):
        with self.assertRaises(TypeError):
            ccdcpp.detect(DATES, CUBE_DN[0, 0], CODES[0, 0])  # pyright: ignore[reportCallIssue]  # no input, refused
        with self.assertRaises(TypeError):
            ccdcpp.detect_cube(DATES, CUBE_DN, CODES, config="ccdc")  # pyright: ignore[reportCallIssue]  # no input, refused
        with self.assertRaises(TypeError):
            ccdcpp.detect_cube(DATES, CUBE_DN, CODES, input=None)  # pyright: ignore[reportArgumentType]  # None, refused
        # input= alone is a whole call: the configuration defaults to "ccdc".
        self.assertEqual(ccdcpp.detect_cube(DATES, CUBE_DN, CODES, input=ccdcpp.Input(),
                                            threads=1)["failures"], [])

    def test_cube_masks_per_pixel(self):
        H, W = CUBE_DN.shape[:2]
        out = ccdcpp.detect_cube(DATES, CUBE_DN, CODES, input=presets.landsat_c1_dn(),
                                 threads=2, masks=True)
        assert "processing_mask" in out
        self.assertEqual(out["processing_mask"].shape, (H, W, len(DATES)))
        pixels = ccdcpp.to_pixels(out)
        self.assertGreater(len(pixels), 0)
        for p in pixels:
            one = ccdcpp.detect(DATES, CUBE_DN[p["row"], p["col"]], CODES[p["row"], p["col"]],
                                input=presets.landsat_c1_dn())
            assert "processing_mask" in p
            np.testing.assert_array_equal(p["processing_mask"], one["processing_mask"])
            self.assertEqual(p["models"], one["models"])
        self.assertNotIn("processing_mask", ccdcpp.to_pixels(
            ccdcpp.detect_cube(DATES, CUBE_DN, CODES, input=ccdcpp.Input(), threads=2))[0])

    def test_extension_errors_pass_through(self):
        h = ccdcpp.Input()
        with self.assertRaises(ValueError) as caught:
            ccdcpp.detect(DATES, CUBE_DN[0, 0, :6], CODES[0, 0], input=h)
        self.assertIn("the input has 7", str(caught.exception))
        # The configuration's schemas, checked when the configuration is built.
        with self.assertRaisesRegex(ValueError, "'peek_size' is 1; it must be at least 2"):
            ccdcpp.config("ccdc").replace(change_test__peek_size=1)

    def test_help_and_pickle(self):
        for name in ("detect", "detect_cube", "to_pixels"):
            with self.subTest(entry=name):
                entry = getattr(ccdcpp, name)
                self.assertEqual((entry.__module__, entry.__name__), ("ccdcpp", name))
                text = pydoc.render_doc(entry)
                words = ("input", "config", "record", "resume", "band_names") \
                    if name != "to_pixels" else ("detect_cube", "models")
                for word in words:
                    self.assertIn(word, text)
                self.assertIs(pickle.loads(pickle.dumps(entry)), entry)
        self.assertIn("SensorProfile", pydoc.render_doc(ccdcpp.SensorProfile))


class Profiles(unittest.TestCase):
    def test_immutable_hashable_picklable(self):
        for preset in [getattr(presets, name) for name in presets.__all__] + [ten_band_profile]:
            with self.subTest(preset=preset.__name__):
                p = preset()
                self.assertEqual(p, preset())
                self.assertEqual(hash(p), hash(preset()))
                q = pickle.loads(pickle.dumps(p))
                self.assertEqual(q, p)
                self.assertEqual(fields(q.input()), fields(p.input()))
                with self.assertRaises(dataclasses.FrozenInstanceError):
                    p.saturation_code = 2.0  # pyright: ignore[reportAttributeAccessIssue]  # frozen, refused

    def test_raw_qa_is_the_decoded_codes(self):
        # A preset with a QA table: raw QA runs as the codes the table decodes it to,
        # given without the table.
        c2 = np.vectorize(C2_RAW.get, otypes=[np.uint16])(CODES)
        fmask = dict(zip(FMASK_VALUES.tolist(), FMASK_CODES.tolist()))
        runs = {"landsat_c2": ((DATES, CUBE_PHYS, c2), {raw: code for code, raw in C2_RAW.items()}),
                "hls_l30": (preset_inputs("hls_l30"), fmask),
                "hls_s30": (preset_inputs("hls_s30"), fmask),
                "hls": (preset_inputs("hls"), fmask),
                "sentinel2_l2a": (preset_inputs("sentinel2_l2a"),
                                  dict(zip(SCL_ALL.tolist(), SCL_CODES.tolist())))}
        for name, ((dates, cube, qa), lookup) in runs.items():
            with self.subTest(preset=name):
                preset = getattr(presets, name)
                values = np.array(list(lookup))
                np.testing.assert_array_equal(qa_table_of(preset().input()).codes[values],
                                              [lookup[v] for v in values.tolist()])
                decoded = np.vectorize(lookup.get, otypes=[np.uint8])(qa)
                assert_same(self,
                            strip_band_names(every_entry(dates, cube, qa, threads=2,
                                                         input=preset())),
                            every_entry(dates, cube, decoded, threads=2,
                                        input=with_table(preset().input(), None)))

    def test_input_is_built_once_and_immutable(self):
        p = presets.landsat_c1_dn()
        self.assertIs(p.input(), p.input())
        self.assertEqual(p.input(), presets.landsat_c1_dn().input())
        with self.assertRaises(AttributeError):
            p.input().reflectance_scale = 99.0  # pyright: ignore[reportAttributeAccessIssue]  # immutable, refused
        h = p.input().replace(reflectance_scale=99.0)
        self.assertEqual(h.reflectance_scale, 99.0)
        self.assertEqual(p.input().reflectance_scale, 10000.0)
        self.assertEqual(copy.copy(p.input()), p.input())

    def test_thermal_at_100c_is_the_transform_of_100_celsius(self):
        # Each preset's raw thermal value of 100 degrees Celsius, as its
        # docstring states the input: Kelvin x 10 (373.15 K) on DN, degrees
        # Celsius / 100 on the physical inputs. Through the thermal transform
        # it is the input's thermal_at_100c, exactly; the presets without a
        # thermal band have none to transform.
        raw_100c = {"landsat_c1_dn": 3731.5, "landsat_c1_physical": 1.0,
                    "landsat_c2": 1.0, "hls_l30": 1.0}
        for preset in presets.__all__:
            with self.subTest(preset=preset):
                h = getattr(presets, preset)().input()
                if preset not in raw_100c:
                    self.assertEqual(h.index("thermal"), ccdcpp.NO_BAND)
                    continue
                self.assertEqual(raw_100c[preset] * h.thermal_scale + h.thermal_offset,
                                 h.thermal_at_100c)
        # The bounds it resolves, bit for bit: pyccd's -93.2 and 70.7 degrees.
        for preset, (lo, hi) in {"landsat_c1_dn": (-9320.0, 7070.0),
                                 "landsat_c2": (-0.932, 0.707)}.items():
            with self.subTest(preset=preset):
                step = ccdcpp.plan_of("ccdc", getattr(presets, preset)()).standard.chain[1]
                self.assertEqual((step.min.hex(), step.max.hex()), (lo.hex(), hi.hex()))

    def test_saturated_reaches_the_input(self):
        for preset in presets.__all__:
            with self.subTest(preset=preset):
                p = getattr(presets, preset)()
                expected = 20000.0 if preset == "landsat_c1_dn" else None
                for name in ("saturation_code", "thermal_saturation_code"):
                    self.assertEqual(getattr(p, name), expected)
                    self.assertEqual(getattr(p.input(), name), expected)
        self.assertIsNone(ccdcpp.Input().saturation_code)
        self.assertIsNone(ccdcpp.Input().thermal_saturation_code)
        p = dataclasses.replace(presets.landsat_c1_physical(), saturation_code=2, thermal_saturation_code=3)
        self.assertEqual((p.saturation_code, p.thermal_saturation_code), (2.0, 3.0))
        self.assertEqual((p.input().saturation_code, p.input().thermal_saturation_code), (2.0, 3.0))
        # In the input's equality and digest, each on its own.
        h = ccdcpp.Input()
        for name in ("saturation_code", "thermal_saturation_code"):
            for value in (20000.0, 65535.0):
                with self.subTest(name=name, value=value):
                    changed: dict[str, Any] = {name: value}
                    other = ccdcpp.Input(**changed)
                    self.assertEqual(getattr(other, name), value)
                    self.assertNotEqual(other, h)
                    self.assertNotEqual(other.digest(), h.digest())
        self.assertNotEqual(ccdcpp.Input(saturation_code=1.0), ccdcpp.Input(thermal_saturation_code=1.0))
        self.assertNotEqual(ccdcpp.Input(saturation_code=1.0).digest(),
                            ccdcpp.Input(thermal_saturation_code=1.0).digest())
        self.assertEqual(copy.copy(h), h)

    def test_a_denser_cadence_and_no_cutoff_are_a_configuration(self):
        cfg = ccdcpp.config("ccdc").replace(change_test__peek_reference_days=5.0,
                                            shared__min_observations=18)
        h = presets.landsat_c1_dn().input()
        self.assertEqual(cfg.params("change_test")["peek_reference_days"], 5.0)
        self.assertEqual(cfg.shared()["min_observations"], 18)
        stages = cfg.resolve(h)
        self.assertEqual(stages["start_search"]["min_observations"], 18)
        self.assertEqual(stages["edge_segments"]["min_observations"], 18)
        self.assertEqual(cfg.shared()["statistics_end"], "none")
        self.assertEqual(cfg.resolve(h)["shared"]["statistics_end"], "none")
        out = every_entry(DATES, CUBE_DN, CODES, input=presets.landsat_c1_dn(), config=cfg)
        self.assertEqual(out["detect_cube"]["failures"], [])

    def test_change_probability_moves_the_threshold(self):
        cfg = ccdcpp.config("ccdc").replace(shared__change_probability=0.995)
        self.assertEqual(set(option_params(cfg, "change_threshold").values()),
                         {ccdcpp.chi2_quantile(0.995, 5)})
        self.assertEqual(resolved(cfg)["change_test"]["change_probability"], 0.995)
        self.assertEqual(option_params(cfg, "outlier_threshold"),
                         option_params(ccdcpp.config("ccdc"), "outlier_threshold"))


class DetectionCount(unittest.TestCase):
    def test_thresholds_follow_the_count(self):
        h, cfg = ten_band_profile().input(), TEN_BAND_CONFIG
        self.assertEqual(len(cfg.resolve(h)["shared"]["detection_bands"]), 6)
        self.assertEqual(set(option_params(cfg, "change_threshold", h).values()),
                         {ccdcpp.chi2_quantile(0.99, 6)})
        self.assertEqual(set(option_params(cfg, "outlier_threshold", h).values()),
                         {ccdcpp.chi2_quantile(1.0 - 1e-6, 6)})
        self.assertEqual(cfg.shared()["outlier_probability"], 1.0 - 1e-6)
        self.assertEqual(ccdcpp.chi2_quantile(1.0 - 1e-6, 5).hex(),
                         outlier_threshold(ccdcpp.config("ccdc")).hex())
        self.assertEqual(ccdcpp.chi2_quantile(0.99, 5).hex(),
                         change_threshold(ccdcpp.config("ccdc")).hex())


if __name__ == "__main__":
    unittest.main()
