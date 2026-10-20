"""The algorithm's configuration from Python: ccdcpp.Config, the presets, config=.

usage: python test_config.py <module_dir> <source_dir> [unittest args]

<module_dir> holds the built extension; <source_dir> is the repository root.
The package is assembled as the wheel lays it out (python/ccdcpp/*.py and
python/ccdcpp/configs next to the extension) in <module_dir>/test_config_package.
ctest runs this as `python_config`.
"""

import glob
import os
import pickle
import shutil
import sys
import unittest

MODULE_DIR, SOURCE_DIR = sys.argv[1], sys.argv[2]
del sys.argv[1:3]

_ROOT = os.path.join(MODULE_DIR, "test_config_package")
shutil.rmtree(_ROOT, ignore_errors=True)
_PACKAGE = os.path.join(_ROOT, "ccdcpp")
shutil.copytree(os.path.join(SOURCE_DIR, "python", "ccdcpp"), _PACKAGE,
                ignore=shutil.ignore_patterns("__pycache__"))
(_extension,) = glob.glob(os.path.join(MODULE_DIR, "ccdcpp.*"))
shutil.copy(_extension, _PACKAGE)
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

import ccdcpp  # noqa: E402

assert ccdcpp.__file__ == os.path.join(_PACKAGE, "__init__.py"), ccdcpp.__file__


def pixel(seed=3, T=420):
    """A pixel of T observations 16 days apart: an annual cycle in every band
    with a level change two thirds of the way, and noise."""
    rng = np.random.default_rng(seed)
    dates = 730000 + 16 * np.arange(T, dtype=np.int64)
    t = np.arange(T)
    spectral = np.empty((7, T))
    for b in range(7):
        spectral[b] = (1000 + 100 * b + 200 * np.sin(2 * np.pi * 16 * t / 365.25 + b)
                       + np.where(t > 2 * T // 3, 400.0, 0.0) + rng.normal(0, 20, T))
    spectral[5] = 2900 + rng.normal(0, 5, T)   # thermal, in range
    qas = np.zeros(T, dtype=np.uint8)
    return dates, np.ascontiguousarray(spectral), qas


class Presets(unittest.TestCase):
    def test_the_ccdc_preset_is_the_built_in_one(self):
        preset = ccdcpp.config("ccdc")
        built_in = ccdcpp.Config(ccdcpp.ccdcpp._config_preset("ccdc"))
        self.assertEqual(preset, built_in)
        self.assertEqual(preset.hash(), built_in.hash())
        self.assertEqual(ccdcpp.config(), preset)

    def test_the_preset_is_physical(self):
        shared = ccdcpp.config("ccdc").shared()
        self.assertEqual(shared["alpha"], 1e-4)
        self.assertNotIn("green_filter", shared)
        self.assertEqual(shared["change_probability"], 0.99)
        self.assertEqual(shared["outlier_probability"], 1.0 - 1e-6)
        self.assertEqual(shared["stat_ord"], ccdcpp.NO_STAT_CUTOFF)
        self.assertEqual(shared["detection_bands"], ["green", "red", "nir", "swir1", "swir2"])
        self.assertEqual(ccdcpp.config("ccdc").params("window_screen")["bands"],
                         ["green", "swir1"])
        self.assertNotIn("alpha", ccdcpp.config("ccdc").params("model"))
        # pyccd's chain, in physical terms.
        clear = ccdcpp.config("ccdc").preprocess("insufficient_clear")
        self.assertFalse(clear["thermal_transform"])
        self.assertEqual([s["step"] for s in clear["steps"]],
                         ["keep_qa", "thermal_range", "band_range", "dedupe", "green_median",
                          "dedupe"])
        self.assertEqual(clear["steps"][0]["codes"], ["clear", "water"])
        self.assertEqual((clear["steps"][1]["min"], clear["steps"][1]["max"]), (-93.2, 70.7))
        self.assertEqual((clear["steps"][2]["min"], clear["steps"][2]["max"]), (0.0, 1.0))
        self.assertEqual(clear["steps"][4], {"step": "green_median", "offset": 0.04,
                                             "scope": "stat_window"})
        self.assertTrue(ccdcpp.config("ccdc").preprocess("standard")["thermal_transform"])

    def test_a_preset_by_name_is_built_once(self):
        self.assertIs(ccdcpp.config("ccdc"), ccdcpp.config("ccdc"))
        self.assertIs(ccdcpp.config(), ccdcpp.config("ccdc"))

    def test_the_file_is_the_registrys_preset(self):
        path = os.path.join(_PACKAGE, "configs", "ccdc.toml")
        self.assertEqual(ccdcpp.config(path),
                         ccdcpp.Config(ccdcpp.ccdcpp._config_preset("ccdc")))

    def test_resolved_on_dn_it_is_pyccds_tree(self):
        r = ccdcpp.config("ccdc").resolve(ccdcpp.Input())
        self.assertEqual(r["stages"]["model"]["params"]["alpha"], 1.0)
        self.assertEqual(r["stages"]["change_test"]["params"]["change_threshold"],
                         15.086272469388987)
        self.assertEqual(r["stages"]["outlier"]["params"]["threshold"], 35.888186879610423)
        self.assertEqual(r["stages"]["window_screen"]["params"]["bands"], [1, 4])
        self.assertEqual(r["shared"]["detection_bands"], [1, 2, 3, 4, 5])
        clear = r["preprocess"]["insufficient_clear"]["steps"]
        self.assertEqual(clear[0]["codes"], [0, 1])
        self.assertEqual((clear[1]["min"], clear[1]["max"]), (-9320.0, 7070.0))
        self.assertEqual((clear[2]["min"], clear[2]["max"]), (0.0, 10000.0))
        self.assertEqual(clear[4], {"step": "green_median", "offset": 400.0,
                                    "scope": "stat_window", "max_date": ccdcpp.NO_STAT_CUTOFF})

    def test_a_file_extends_a_preset(self):
        path = os.path.join(_ROOT, "variant.toml")
        with open(path, "w") as f:
            f.write('extends = "ccdc"\n\n[refit]\nstrategy = "span-factor"\n'
                    '[refit.params]\nfactor = 2.0\n')
        variant = ccdcpp.config(path)
        base = ccdcpp.config("ccdc")
        self.assertEqual(variant.params("refit"), {"min_obs": 24, "factor": 2.0})
        for stage in base.stages():
            if stage != "refit":
                self.assertEqual(variant.params(stage), base.params(stage), stage)
        self.assertNotEqual(variant.hash(), base.hash())

    def test_to_toml_reads_back_as_the_same_configuration(self):
        path = os.path.join(_ROOT, "written.toml")
        for cfg in (ccdcpp.config(),
                    ccdcpp.config().replace(window_screen__bands=[1, 2, 4], refit__factor=1.5),
                    ccdcpp.config().replace(shared__stat_ord=None,
                                            shared__detection_bands=["red", "nir"])):
            with open(path, "w") as f:
                f.write(cfg.to_toml())
            self.assertEqual(ccdcpp.config(path), cfg)

    def test_no_statistics_cutoff_is_the_largest_ordinal(self):
        cfg = ccdcpp.config().replace(shared__stat_ord=None)
        self.assertEqual(cfg.shared()["stat_ord"], ccdcpp.NO_STAT_CUTOFF)
        self.assertEqual(ccdcpp.NO_STAT_CUTOFF, 2**63 - 1)
        self.assertIn("stat_ord = 9223372036854775807", cfg.to_toml())

    def test_an_unknown_preset_names_the_presets(self):
        with self.assertRaisesRegex(ValueError, r"the presets are (.*, )?ccdc$"):
            ccdcpp.config("nope")
        # Names are exact, whatever the file system's case rules.
        with self.assertRaises(ValueError):
            ccdcpp.config("CCDC")

    def test_the_registrys_presets_are_presets(self):
        # A preset an extension registers has no file; its name finds it.
        self.assertIn("ccdc", ccdcpp.ccdcpp._presets())
        for name in ccdcpp.ccdcpp._presets():
            self.assertEqual(ccdcpp.config(name),
                             ccdcpp.Config(ccdcpp.ccdcpp._config_preset(name)), name)


class Replace(unittest.TestCase):
    def test_a_parameter(self):
        cfg = ccdcpp.config().replace(change_test__peek_size=7)
        self.assertEqual(cfg.params("change_test")["peek_size"], 7)
        self.assertEqual(ccdcpp.config().params("change_test")["peek_size"], 6)

    def test_a_shared_value_reaches_every_stage_that_reads_it(self):
        cfg = ccdcpp.config().replace(shared__change_probability=0.95)
        self.assertEqual(cfg.shared()["change_probability"], 0.95)
        stages = cfg.resolve(ccdcpp.Input())["stages"]
        for stage in ("change_test", "stability_rule", "lookback"):
            self.assertEqual(stages[stage]["params"]["change_probability"], 0.95, stage)
            self.assertEqual(stages[stage]["params"]["change_threshold"],
                             ccdcpp.chi2_quantile(0.95, 5), stage)
        self.assertEqual(ccdcpp.config().shared()["change_probability"], 0.99)

    def test_a_stage_cannot_set_a_shared_value(self):
        with self.assertRaisesRegex(ValueError, "parameter 'alpha' is the shared value 'alpha'; "
                                                "set shared__alpha instead"):
            ccdcpp.config().replace(model__alpha=1.0)
        with self.assertRaisesRegex(ValueError, "parameter 'change_threshold' is the shared "
                                                "value 'change_probability'"):
            ccdcpp.config().replace(change_test__change_threshold=1.0)
        with self.assertRaisesRegex(ValueError, "there is no shared value 'alfa'"):
            ccdcpp.config().replace(shared__alfa=1.0)

    def test_a_strategy_swap_starts_from_its_defaults(self):
        cfg = ccdcpp.config().replace(refit__factor=3.0).replace(refit="span-factor")
        self.assertEqual(cfg.params("refit"), {"min_obs": 24, "factor": 1.33})

    def test_unknown_names_are_refused_by_name(self):
        with self.assertRaisesRegex(ValueError, "has no parameter 'fctor'; it has min_obs, factor"):
            ccdcpp.config().replace(refit__fctor=2.0)
        with self.assertRaisesRegex(ValueError, "refit has no strategy 'every-n'"):
            ccdcpp.config().replace(refit="every-n")

    def test_check_refuses_what_a_run_would(self):
        with self.assertRaisesRegex(ValueError, "parameter 'factor' is 0; it must be greater than 0"):
            ccdcpp.config().replace(refit__factor=0.0).check()
        three = ccdcpp.Input()
        three.roles = ["blue", "green", "red"]
        three.saturation_bands = [0, 1, 2]
        with self.assertRaisesRegex(ValueError, "holds 'nir', which the input does not have; "
                                                "its rows are blue, green, red"):
            ccdcpp.config().check(three)
        rows = ccdcpp.config().replace(window_screen__bands=[1, 4],
                                       shared__detection_bands=[1, 2])
        with self.assertRaisesRegex(ValueError, "holds band 4; the input has 3 bands"):
            rows.check(three)
        ccdcpp.config().check(ccdcpp.Input())

    def test_equal_values_hash_equal(self):
        import numpy as np
        a = ccdcpp.config().replace(refit__factor=2)
        b = ccdcpp.config().replace(refit__factor=2.0)
        self.assertEqual(a, b)
        self.assertEqual(a.hash(), b.hash())
        c = ccdcpp.config().replace(refit__min_obs=np.int64(30), window_screen__bands=np.array([1, 4]))
        self.assertEqual(c.params("refit")["min_obs"], 30)
        self.assertEqual(c.params("window_screen")["bands"], [1, 4])
        c.check()

    def test_bad_values_raise_value_errors(self):
        with self.assertRaisesRegex(ValueError, "does not fit a 64-bit integer"):
            ccdcpp.config().replace(refit__min_obs=2**70).check()
        with self.assertRaises(TypeError):
            ccdcpp.ccdcpp._config_digest(None)

    def test_immutable_printable_picklable(self):
        cfg = ccdcpp.config()
        with self.assertRaises(AttributeError):
            cfg.x = 1
        self.assertIn("refit: span-factor(factor=1.33, min_obs=24)", repr(cfg))
        self.assertIn("shared: alpha=0.0001", repr(cfg))
        self.assertEqual(pickle.loads(pickle.dumps(cfg)), cfg)


class Strategies(unittest.TestCase):
    def test_every_stage_lists_its_strategies_and_parameters(self):
        table = ccdcpp.strategies()
        self.assertEqual(sorted(table), ccdcpp.config().stages())
        refit = {p["name"]: p for p in table["refit"]["span-factor"]}
        self.assertEqual(refit["factor"]["default"], 1.33)
        self.assertEqual(refit["min_obs"]["unit"], "observations")
        self.assertEqual(refit["factor"]["bounds"], "greater than 0")
        self.assertEqual(refit["factor"]["source"], "own")
        lasso = {p["name"]: p for p in table["model"]["lasso"]}
        self.assertEqual((lasso["alpha"]["source"], lasso["alpha"]["from"]), ("shared", "alpha"))
        self.assertIsNone(lasso["alpha"]["default"])
        chi2 = {p["name"]: p for p in table["change_test"]["chi2"]}
        self.assertEqual((chi2["change_threshold"]["source"], chi2["change_threshold"]["from"]),
                         ("chi2", "change_probability"))
        shared = {v["name"]: v for v in ccdcpp.shared_values()}
        self.assertEqual(shared["alpha"]["unit"], "reflectance")
        self.assertEqual(shared["alpha"]["default"], 1e-4)


class Runs(unittest.TestCase):
    def test_the_preset_runs_as_the_default(self):
        d, s, q = pixel()
        h = ccdcpp.Input()
        plain = ccdcpp.detect(d, s, q, input=h)
        self.assertGreater(len(plain["models"]), 1)
        for cfg in ("ccdc", ccdcpp.config("ccdc"),
                    ccdcpp.Config(ccdcpp.ccdcpp._config_preset("ccdc"))):
            configured = ccdcpp.detect(d, s, q, input=h, config=cfg)
            self.assertEqual(plain["models"], configured["models"])
        # The extension's own default (config=None) is the same built-in preset.
        bare = ccdcpp.ccdcpp.detect(d, s, q, input=h)
        self.assertEqual(plain["models"], bare["models"])

    def test_config_is_a_config_or_a_preset_name(self):
        d, s, q = pixel()
        with self.assertRaisesRegex(TypeError, "a ccdcpp.Config or a preset's name"):
            ccdcpp.detect(d, s, q, input=ccdcpp.Input(), config=ccdcpp.config().to_dict())

    def test_a_configuration_changes_the_run(self):
        d, s, q = pixel()
        h = ccdcpp.Input()
        plain = ccdcpp.detect(d, s, q, input=h)
        sparse = ccdcpp.detect(d, s, q, input=h, config=ccdcpp.config().replace(shared__alpha=0.01))
        # A hundred times the Lasso penalty in every Lasso stage: other coefficients.
        self.assertNotEqual(sparse["models"], plain["models"])
        strict = ccdcpp.detect(
            d, s, q, input=h, config=ccdcpp.config().replace(change_test__peek_size=200))
        # No peek of 200 observations fits after the change: it is not found.
        self.assertLess(len(strict["models"]), len(plain["models"]))

    def test_the_cube_entry_points_take_it(self):
        d, s, q = pixel()
        h = ccdcpp.Input()
        cube = np.ascontiguousarray(np.broadcast_to(s, (2, 2) + s.shape))
        qc = np.ascontiguousarray(np.broadcast_to(q, (2, 2) + q.shape))
        cfg = ccdcpp.config("ccdc")
        a = ccdcpp.detect_cube(d, cube, qc, input=h, threads=1)
        b = ccdcpp.detect_cube(d, cube, qc, input=h, threads=1, config=cfg)
        for key in a:
            if key != "failures":
                np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        self.assertEqual(b["pixels_total"], 4)

    def test_a_trace_records_the_run(self):
        d, s, q = pixel()
        h = ccdcpp.Input()
        plain = ccdcpp.detect(d, s, q, input=h)
        traced = ccdcpp.detect(d, s, q, input=h, trace=True)
        self.assertEqual(plain["models"], traced["models"])
        rows = traced["trace"]
        if not rows:
            self.skipTest("built with CCD_OBSERVE=OFF")
        self.assertEqual(rows[0]["kind"], "transition")
        self.assertEqual(rows[0]["phase"], "Prepare")
        self.assertEqual(rows[-1]["kind"], "exit")
        self.assertEqual(sum(r["kind"] == "segment" for r in rows), len(plain["models"]))
        tests = [r for r in rows if r["kind"] == "test"]
        self.assertTrue(any(r["outcome"] == "change" for r in tests))
        cube = np.ascontiguousarray(np.broadcast_to(s, (2, 2) + s.shape))
        qc = np.ascontiguousarray(np.broadcast_to(q, (2, 2) + q.shape))
        out = ccdcpp.detect_cube(d, cube, qc, input=h, threads=2, trace_pixels=[(1, 0)])
        self.assertEqual(list(out["traces"]), [(1, 0)])
        self.assertEqual(out["traces"][(1, 0)], rows)

    def test_a_bad_configuration_is_refused_before_any_pixel(self):
        d, s, q = pixel()
        h = ccdcpp.Input()
        tree = ccdcpp.config().to_dict()
        tree["stages"]["refit"] = {"strategy": "nope", "params": {}}
        with self.assertRaisesRegex(ValueError, "refit has no strategy 'nope'"):
            ccdcpp.detect(d, s, q, input=h, config=ccdcpp.Config(tree))
        with self.assertRaisesRegex(ValueError, "the configuration has no 'mode' stage"):
            ccdcpp.detect(d, s, q, input=h,
                          config=ccdcpp.Config({"stages": {"refit": {"strategy": "span-factor"}}}))
        with self.assertRaisesRegex(ValueError, "refit 'span-factor' parameter 'factor' is 0"):
            ccdcpp.detect(d, s, q, input=h, config=ccdcpp.config().replace(refit__factor=0.0))


if __name__ == "__main__":
    try:
        unittest.main()
    finally:
        shutil.rmtree(_ROOT, ignore_errors=True)
