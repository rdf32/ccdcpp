"""Sensor profiles, the Landsat presets and a profile as encoding=.

usage: python test_profiles.py <module_dir> <source_dir> [unittest args]

<module_dir> holds the built extension; <source_dir> is the repository root.
The package is assembled as the wheel lays it out, python/ccdcpp/*.py next to
the extension, in a temporary directory. ctest runs this as
`python_profiles`.
"""
import copy
import dataclasses
import glob
import inspect
import os
import pickle
import pydoc
import shutil
import sys
import unittest
from unittest import mock
from datetime import date

MODULE_DIR, SOURCE_DIR = sys.argv[1], sys.argv[2]
del sys.argv[1:3]

# The package as the wheel lays it out (python/ccdcpp with its configs, the
# extension beside them), in the build directory.
# One per test file that runs this setup (test_preprocess.py and
# test_real_fixtures.py import it), so parallel runs do not share it.
_ROOT = os.path.join(MODULE_DIR, "test_package_" +
                     os.path.splitext(os.path.basename(sys.argv[0]))[0])
shutil.rmtree(_ROOT, ignore_errors=True)
_PACKAGE = os.path.join(_ROOT, "ccdcpp")
shutil.copytree(os.path.join(SOURCE_DIR, "python", "ccdcpp"), _PACKAGE,
                ignore=shutil.ignore_patterns("__pycache__"))
(_extension,) = glob.glob(os.path.join(MODULE_DIR, "ccdcpp.*"))
shutil.copy(_extension, _PACKAGE)
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

import ccdcpp  # noqa: E402
from ccdcpp import presets  # noqa: E402

assert ccdcpp.__file__ == os.path.join(_PACKAGE, "__init__.py"), ccdcpp.__file__

EXT = ccdcpp.ccdcpp
START = 724276  # 1984-01-01
SCALE = 10000.0
FILL, CLEAR, WATER, SHADOW, SNOW, CLOUD = 255, 0, 1, 2, 3, 4

#: One raw QA_PIXEL value per class (Landsat Collection 2).
C2_RAW = {FILL: 1, CLEAR: 21824, WATER: 21952, SHADOW: 23888, SNOW: 30048, CLOUD: 22280}

#: One raw HLS v2.0 Fmask value per class: fill 255, bit 5 water, bit 3
#: shadow, bit 4 snow, bit 1 cloud; 0 is clear with climatology aerosol.
FMASK_RAW = {FILL: 255, CLEAR: 0, WATER: 32, SHADOW: 8, SNOW: 16, CLOUD: 2}


LANDSAT_PRESETS = (presets.landsat_c1_dn, presets.landsat_c1_physical, presets.landsat_c2)


def fields(obj):
    return {
        name: getattr(obj, name) for name in dir(obj)
        if not name.startswith("_") and not callable(getattr(obj, name))
    }


def assert_same(test, a, b, where="out"):
    """Exact, structural equality of entry point outputs."""
    if isinstance(a, dict):
        test.assertEqual(list(a), list(b), where)
        for k in a:
            assert_same(test, a[k], b[k], f"{where}[{k!r}]")
    elif isinstance(a, (list, tuple)):
        test.assertEqual(type(a), type(b), where)
        test.assertEqual(len(a), len(b), where)
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(test, x, y, f"{where}[{i}]")
    elif isinstance(a, np.ndarray):
        test.assertEqual((a.dtype, a.shape), (b.dtype, b.shape), where)
        test.assertEqual(a.tobytes(), b.tobytes(), where)
    elif isinstance(a, float):
        test.assertEqual(a.hex(), b.hex(), where)
    else:
        test.assertEqual(a, b, where)


def without(out, *keys):
    out = dict(out)
    for k in keys:
        out.pop(k)
    return out


def reference_pixel():
    """pyccd's reference pixel: (dates, spectral (7, T) DN, qas)."""
    path = os.path.join(SOURCE_DIR, "tests", "ccd", "machine", "test_3657_3610_observations.csv")
    data = np.loadtxt(path, delimiter=",", dtype=np.int64)
    return (data[:, 0].copy(), np.ascontiguousarray(data[:, 1:8].T, dtype=np.float64),
            data[:, 8].astype(np.uint8))


def corpus_cube(H=2, W=3, seed=11):
    """A (H, W, 7, T) DN cube on the reference pixel's dates: the reference
    pixel itself, a snow-dominated pixel (PermanentSnow), a cloud-dominated
    one (InsufficientClear), and clear pixels with a break (Standard).

    Returns (dates, spectral, codes), codes free of FILL-only rows and of
    code 8, so each has a QA_PIXEL encoding."""
    dates, ref, ref_qa = reference_pixel()
    T = dates.size
    rng = np.random.default_rng(seed)
    phase = 2 * np.pi * (dates - START) / 365.25
    spectral = np.empty((H, W, 7, T))
    codes = np.empty((H, W, T), dtype=np.uint8)
    for p in range(H * W):
        r, c = divmod(p, W)
        if p == 0:
            spectral[r, c], codes[r, c] = ref, ref_qa
            continue
        base = 600 + 300 * np.arange(7, dtype=float)[:, None]
        season = (150 + 20 * p) * np.sin(phase + p)[None, :]
        level = np.where(np.arange(T) > T // 2 + 7 * p, 900.0, 0.0)[None, :]
        s = base + season + level + rng.normal(0, 40, (7, T))
        s[6] = 2900 + 80 * np.sin(phase) + rng.normal(0, 10, T)
        spectral[r, c] = np.round(s)
        probs = {1: [0.02, 0.05, 0.01, 0.05, 0.85, 0.02],
                 2: [0.02, 0.17, 0.03, 0.05, 0.02, 0.71]}.get(
                     p, [0.04, 0.74, 0.06, 0.05, 0.03, 0.08])
        codes[r, c] = rng.choice(
            np.array([FILL, CLEAR, WATER, SHADOW, SNOW, CLOUD], dtype=np.uint8), size=T, p=probs)
    codes[codes == 8] = SHADOW
    return dates, spectral, codes


def to_physical(spectral):
    """DN -> the physical preset's units: reflectance, degrees Celsius / 100."""
    s = spectral.copy()
    s[..., :6, :] = s[..., :6, :] / SCALE
    s[..., 6, :] = (s[..., 6, :] * 10.0 - 27315.0) / SCALE
    return s


#: The "ccdc" preset with pyccd's statistics cutoff, 2017-12-31: what the Landsat profiles run.
PYCCD = ccdcpp.config("ccdc").replace(shared__statistics_end=736694)


def explicit_physical():
    """The physical-unit input written out, as the golden tests write it, and the
    configuration it runs: (Input, Config)."""
    h = ccdcpp.Input(reflectance_scale=1.0, thermal_scale=1.0, thermal_offset=0.0,
                     thermal_at_100c=1.0)
    return h, PYCCD


def resolved(cfg, encoding=None):
    """``cfg``'s stages resolved against ``encoding`` (default ccdcpp.Input())."""
    return cfg.resolve(ccdcpp.Input() if encoding is None else encoding)["stages"]


def change_threshold(cfg, encoding=None):
    return resolved(cfg, encoding)["change_test"]["params"]["change_threshold"]


def outlier_threshold(cfg, encoding=None):
    return resolved(cfg, encoding)["outlier"]["params"]["threshold"]


# The parameters one shared value fills in, as a run resolves them.
GROUPS = {
    "alpha": (("model", "alpha"), ("stability_fit", "alpha"), ("edge_fit", "alpha")),
    "coef_floor": (("model", "coef_floor"), ("stability_fit", "coef_floor"),
                   ("edge_fit", "coef_floor")),
    "change_threshold": (("change_test", "change_threshold"),
                         ("stability_rule", "change_threshold"),
                         ("lookback", "change_threshold")),
    "outlier_threshold": (("outlier", "threshold"), ("lookback", "outlier_threshold")),
    "tmask": (("window_screen", "bands"),),
}


def option_params(cfg, name, encoding=None):
    """{(stage, parameter): value} of group ``name`` (GROUPS) in ``cfg`` resolved
    against ``encoding``, or of one ``stage__parameter``."""
    pairs = GROUPS.get(name) or (tuple(name.split("__", 1)),)
    stages = resolved(cfg, encoding)
    return {(stage, param): stages[stage]["params"][param]
            for stage, param in pairs
            if stage in stages and param in stages[stage]["params"]}


DATES, CUBE_DN, CODES = corpus_cube()
CUBE_PHYS = to_physical(CUBE_DN)


def every_entry(dates, spectral, qas, threads=8, **kw):
    """{entry: output} for detect on every pixel and detect_cube (its columns,
    without the thread count)."""
    H, W = spectral.shape[:2]
    return {
        "detect": [ccdcpp.detect(dates, spectral[r, c], qas[r, c], **kw)
                   for r in range(H) for c in range(W)],
        "detect_cube": without(
            ccdcpp.detect_cube(dates, spectral, qas, threads=threads, **kw), "threads"),
    }


def strip_band_names(outs):
    return {
        "detect": [without(o, "band_names") for o in outs["detect"]],
        "detect_cube": without(outs["detect_cube"], "band_names"),
    }


def band_range_rows(encoding, config="ccdc"):
    """The rows the Standard chain's band_range step tests on ``encoding``."""
    (step,) = [s for s in ccdcpp.plan_of(config, encoding).standard.chain
               if s.kind == ccdcpp.StepKind.BandRange]
    return step.bands


def with_table(encoding, table):
    """``encoding`` (a ccdcpp.Input) decoding its QA through ``table``."""
    return encoding.replace(qa_table=table)


class Corpus(unittest.TestCase):
    def test_corpus_runs_every_procedure(self):
        h = ccdcpp.Input()
        out = EXT.detect_cube(DATES, CUBE_DN, CODES, encoding=h, threads=1)
        self.assertEqual(out["failures"], [])
        # Standard, PermanentSnow and InsufficientClear.
        self.assertTrue({8, 54, 44} <= set(out["curve_qa"].tolist()), out["curve_qa"])
        self.assertGreater(len(out["row"]), 6)


class LandsatC1DN(unittest.TestCase):
    def test_options_are_the_defaults_field_by_field(self):
        h = presets.landsat_c1_dn().input()
        self.assertEqual(fields(h), fields(ccdcpp.Input(saturation_code=20000.0,
                                                        thermal_saturation_code=20000.0)))
        for threshold in (change_threshold, outlier_threshold):
            self.assertEqual(threshold(PYCCD, h).hex(), threshold(ccdcpp.config("ccdc")).hex())
        self.assertGreater(len(fields(h)), 14)
        self.assertIsNone(presets.landsat_c1_dn().input().qa_table)

    def test_output_is_the_unprofiled_output(self):
        h = ccdcpp.Input()
        profiled = every_entry(DATES, CUBE_DN, CODES, encoding=presets.landsat_c1_dn())
        plain = every_entry(DATES, CUBE_DN, CODES, encoding=h)
        assert_same(self, strip_band_names(profiled), plain)


class LandsatPhysical(unittest.TestCase):
    def test_options_are_the_explicit_physical_options(self):
        explicit, cfg = explicit_physical()
        for preset in (presets.landsat_c1_physical, presets.landsat_c2):
            with self.subTest(preset=preset.__name__):
                # landsat_c2's input decodes raw QA_PIXEL.
                h = with_table(explicit, preset().input().qa_table)
                self.assertEqual(fields(preset().input()), fields(h))
                self.assertEqual(cfg.resolve(preset().input()), cfg.resolve(h))

    def test_output_is_the_explicit_physical_output(self):
        h, cfg = explicit_physical()
        profiled = every_entry(DATES, CUBE_PHYS, CODES, encoding=presets.landsat_c1_physical(),
                               config=cfg)
        plain = every_entry(DATES, CUBE_PHYS, CODES, encoding=h, config=cfg)
        assert_same(self, strip_band_names(profiled), plain)

    def test_c2_raw_qa_pixel_is_the_decoded_codes(self):
        raw = np.vectorize(C2_RAW.get, otypes=[np.uint16])(CODES)
        profile = presets.landsat_c2()
        np.testing.assert_array_equal(profile.input().qa_table.codes[raw], CODES)
        decoded = every_entry(DATES, CUBE_PHYS, raw, encoding=profile)
        codes = every_entry(DATES, CUBE_PHYS, CODES, encoding=with_table(profile.input(), None))
        assert_same(self, strip_band_names(decoded), codes)
        h, _ = explicit_physical()
        assert_same(self, codes, every_entry(DATES, CUBE_PHYS, CODES, encoding=h))

    def test_presets_state_units_and_band_order(self):
        for preset in LANDSAT_PRESETS:
            with self.subTest(preset=preset.__name__):
                self.assertIn("Input:", preset.__doc__)
                self.assertIn("QA:", preset.__doc__)
                self.assertEqual(preset().band_names,
                                 ("blue", "green", "red", "nir", "swir1", "swir2", "thermal"))
                self.assertEqual(preset().name, preset.__name__)
        self.assertIn("blue, green, red, nir, swir1, swir2, thermal", presets.__doc__)
        self.assertEqual(presets.__all__, ["landsat_c1_dn", "landsat_c1_physical", "landsat_c2",
                                           "hls_l30", "hls_s30", "hls", "sentinel2_l2a"])


def permuted(profile, order):
    return dataclasses.replace(
        profile,
        band_names=tuple(profile.band_names[i] for i in order),
        roles=tuple(profile.roles[i] for i in order),
    )


def unpermute(outs, order):
    """Outputs of a run on rows `order` -> the outputs of the run in row order."""
    inv = np.argsort(order)

    def models(ms):
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


class Permutation(unittest.TestCase):
    def test_landsat(self):
        order = [6, 3, 0, 5, 1, 4, 2]
        base = presets.landsat_c1_dn()
        profile = permuted(base, order)
        h = profile.input()
        self.assertEqual(h.index("green"), order.index(1))
        self.assertEqual(h.index("thermal"), order.index(6))
        self.assertEqual(ccdcpp.config("ccdc").resolve(h)["shared"]["detection_bands"],
                         [order.index(i) for i in (1, 2, 3, 4, 5)])
        expected = every_entry(DATES, CUBE_DN, CODES, encoding=base)
        got = every_entry(DATES, np.ascontiguousarray(CUBE_DN[:, :, order]), CODES,
                          encoding=profile)
        assert_same(self, unpermute(got, order), expected)

    def test_ten_bands(self):
        order = [9, 0, 8, 1, 7, 2, 6, 3, 5, 4]
        base = ten_band_profile()
        dates, cube, scl = ten_band_inputs()
        expected = every_entry(dates, cube, scl, encoding=base)
        got = every_entry(dates, np.ascontiguousarray(cube[:, :, order]), scl,
                          encoding=permuted(base, order))
        assert_same(self, unpermute(got, order), expected)


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
            "detect": lambda: ccdcpp.detect(d, s[0, 0], q[0, 0], encoding=profile),
            "detect_cube": lambda: ccdcpp.detect_cube(d, s, q, encoding=profile),
        }
        for name, call in calls.items():
            with self.subTest(entry=name):
                with self.assertRaises(ValueError) as caught:
                    call()
                self.assertIn("SensorProfile 'landsat_c1_dn'", str(caught.exception))

    def test_not_a_profile(self):
        with self.assertRaises(TypeError):
            ccdcpp.detect(DATES, CUBE_DN[0, 0], CODES[0, 0], encoding="landsat_c1_dn")

    def test_construction(self):
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
        # be listed, and it is not the "nir" row.
        p = ten_band_profile()
        h = p.input()
        self.assertEqual(p.index("nir_broad"), 6)
        self.assertEqual(h.index("nir"), 7)
        self.assertIn(6, band_range_rows(h))
        cfg = ccdcpp.config("ccdc").replace(
            shared__detection_bands=["green", "red", "nir_broad", "nir", "swir1", "swir2"])
        self.assertEqual(cfg.resolve(h)["shared"]["detection_bands"], [1, 2, 6, 7, 8, 9])
        with self.assertRaises(ValueError) as caught:
            dataclasses.replace(p, roles=p.roles[:7] + ("nir_broad",) + p.roles[8:])
        self.assertIn("repeats a role other than 'other'", str(caught.exception))


#: Every raw HLS Fmask value, for comparing whole tables.
ALL_VALUES = np.arange(65536)


class QaTableOptionsField(unittest.TestCase):
    def hls(self, **kw):
        return dataclasses.replace(presets.landsat_c1_physical(), qa_table_builder="hls_fmask",
                                   **kw)

    def test_default_is_empty(self):
        for preset in (presets.landsat_c1_dn, presets.landsat_c1_physical, presets.landsat_c2):
            with self.subTest(preset=preset.__name__):
                self.assertEqual(preset().qa_table_options, ())
        np.testing.assert_array_equal(self.hls().input().qa_table.codes,
                                      ccdcpp.qa_tables.hls_fmask().codes)

    def test_builder_arguments_reach_the_builder(self):
        for aerosol in (False, True):
            with self.subTest(exclude_high_aerosol=aerosol):
                p = self.hls(qa_table_options={"exclude_high_aerosol": aerosol})
                h = p.input()
                expected = ccdcpp.qa_tables.hls_fmask(h, exclude_high_aerosol=aerosol).codes
                got = p.input().qa_table.codes
                self.assertEqual(got.shape, ALL_VALUES.shape)
                np.testing.assert_array_equal(got, expected)
        # The two tables differ exactly where the aerosol bits are 11 and no
        # higher-precedence bit (fill, cloud, adjacent, cirrus) is set.
        on = self.hls(qa_table_options={"exclude_high_aerosol": True}).input().qa_table.codes
        off = self.hls().input().qa_table.codes
        differs = np.flatnonzero(on != off)
        self.assertTrue(len(differs) > 0)
        self.assertTrue(((differs >> 6) == 3).all())
        self.assertEqual(int(on[192]), CLOUD)
        self.assertEqual(int(off[192]), CLEAR)

    def test_cached_by_value(self):
        a = self.hls(qa_table_options={"exclude_high_aerosol": True})
        b = self.hls(qa_table_options=(("exclude_high_aerosol", True),))
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertIs(a.input().qa_table, b.input().qa_table)
        self.assertIsNot(a.input().qa_table, self.hls().input().qa_table)

    def test_stored_sorted_hashable_picklable(self):
        p = self.hls(qa_table_options={"exclude_high_aerosol": True})
        self.assertEqual(p.qa_table_options, (("exclude_high_aerosol", True),))
        q = pickle.loads(pickle.dumps(p))
        self.assertEqual(q, p)
        np.testing.assert_array_equal(q.input().qa_table.codes, p.input().qa_table.codes)

    def test_entry_points_use_the_profile_table(self):
        # Raw Fmask, with high aerosol on every third clear observation:
        # screened out only with the option.
        fmask = np.vectorize(FMASK_RAW.get, otypes=[np.uint8])(CODES)
        aerosol = np.zeros(fmask.shape, dtype=bool)
        aerosol[..., ::3] = True
        aerosol &= fmask == 0
        fmask[aerosol] = 192
        base = self.hls()
        screened = self.hls(qa_table_options={"exclude_high_aerosol": True})
        table = ccdcpp.qa_tables.hls_fmask(screened.input(), True)
        got = strip_band_names(every_entry(DATES, CUBE_PHYS, fmask, encoding=screened))
        assert_same(self, got, every_entry(DATES, CUBE_PHYS, fmask,
                                           encoding=with_table(base.input(), table)))
        codes = np.where(aerosol, np.uint8(CLOUD), CODES)
        assert_same(self, got, every_entry(DATES, CUBE_PHYS, codes,
                                           encoding=with_table(base.input(), None)))
        self.assertNotEqual(
            ccdcpp.detect(DATES, CUBE_PHYS[0, 1], fmask[0, 1], encoding=base)["processing_mask"]
            .tobytes(),
            ccdcpp.detect(DATES, CUBE_PHYS[0, 1], fmask[0, 1], encoding=screened)["processing_mask"]
            .tobytes())

    def test_validation(self):
        cases = [
            (presets.landsat_c1_dn(), dict(qa_table_options={"exclude_high_aerosol": True}),
             "needs a qa_table_builder"),
            (presets.landsat_c2(), dict(qa_table_options={"exclude_high_aerosol": True}),
             "landsat_c2_qa_pixel does not take qa_table_options"),
            (self.hls(), dict(qa_table_options={"high_aerosol": True}),
             "hls_fmask does not take qa_table_options {'high_aerosol': True}"),
            (self.hls(), dict(qa_table_options=(("exclude_high_aerosol", True),
                                                ("exclude_high_aerosol", False))),
             "qa_table_options repeats a name"),
            (self.hls(), dict(qa_table_options={"exclude_high_aerosol": [True]}),
             "'exclude_high_aerosol' is [True] (list); ccdcpp.qa_tables.hls_fmask takes a "
             "bool"),
        ]
        for base, change, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaises(ValueError) as caught:
                    dataclasses.replace(base, **change)
                self.assertIn(expected, str(caught.exception))

    def test_values_are_the_declared_type(self):
        # A bool option takes True or False; nothing else is read as one.
        for value in (None, 0, 1, 2, 1.0, "True", "", np.bool_(True)):
            with self.subTest(value=value):
                with self.assertRaises(ValueError) as caught:
                    self.hls(qa_table_options={"exclude_high_aerosol": value})
                message = str(caught.exception)
                self.assertIn(f"'exclude_high_aerosol' is {value!r} "
                              f"({type(value).__name__})", message)
                self.assertIn("ccdcpp.qa_tables.hls_fmask takes a bool", message)
                self.assertNotIn("does not take", message)

    def test_declared_options_are_the_bindings(self):
        builders = sorted(name for name in dir(ccdcpp.qa_tables)
                          if not name.startswith("_")
                          and callable(getattr(ccdcpp.qa_tables, name)))
        self.assertEqual(sorted(ccdcpp.QA_TABLE_OPTIONS), builders)
        for builder, declared in ccdcpp.QA_TABLE_OPTIONS.items():
            with self.subTest(builder=builder):
                signature = getattr(ccdcpp.qa_tables, builder).__doc__.splitlines()[0]
                self.assertEqual(signature.count(": "), 1 + len(declared), signature)
                self.assertIn(f"{builder}(encoding: ", signature)
                for name, (kind, default) in declared.items():
                    self.assertIn(f"{name}: {kind.__name__} = {default!r}", signature)

    def test_malformed_input_raises(self):
        cases = [
            (None, "qa_table_options is None; it is a mapping or (name, value) pairs"),
            ("ab", "qa_table_options is 'ab'; it is a mapping or (name, value) pairs"),
            (5, "qa_table_options is 5"),
            (("exclude_high_aerosol",), "entry 'exclude_high_aerosol' is not a (name, value) "
             "pair"),
            ((("exclude_high_aerosol", True, 1),), "is not a (name, value) pair"),
            ({1: True}, "qa_table_options name 1 is not a str"),
            ((("exclude_high_aerosol", [1]), ("exclude_high_aerosol", {})),
             "qa_table_options repeats a name"),
            ((("exclude_high_aerosol", True), ("exclude_high_aerosol", True)),
             "qa_table_options repeats a name"),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                with self.assertRaises(ValueError) as caught:
                    self.hls(qa_table_options=value)
                self.assertTrue(str(caught.exception).startswith("SensorProfile "),
                                str(caught.exception))
                self.assertIn(expected, str(caught.exception))

    def test_declared_defaults_are_written_out(self):
        written = (("exclude_high_aerosol", False),)
        given = [(), {}, {"exclude_high_aerosol": False}, written]
        profiles = [self.hls(qa_table_options=options) for options in given]
        for options, p in zip(given, profiles):
            with self.subTest(options=options):
                self.assertEqual(p.qa_table_options, written)
                self.assertEqual(p, profiles[0])
                self.assertEqual(hash(p), hash(profiles[0]))
                self.assertIs(p.input().qa_table, profiles[0].input().qa_table)
                self.assertIn("qa_table_options=(('exclude_high_aerosol', False),)", repr(p))
        for preset in (presets.hls_l30, presets.hls_s30, presets.hls):
            with self.subTest(preset=preset.__name__):
                self.assertEqual(preset().qa_table_options, written)
                self.assertEqual(dataclasses.replace(preset(), qa_table_options=()), preset())
                self.assertEqual(pickle.loads(pickle.dumps(preset())).qa_table_options,
                                 written)
        # Builders that declare nothing, and no builder, store nothing.
        for preset in (presets.landsat_c1_dn, presets.landsat_c2, presets.sentinel2_l2a):
            with self.subTest(preset=preset.__name__):
                self.assertEqual(preset().qa_table_options, ())

    def test_names_are_stored_as_str(self):
        class Name(str):
            pass

        p = self.hls(qa_table_options={Name("exclude_high_aerosol"): True})
        self.assertIs(type(p.qa_table_options[0][0]), str)

    def test_numpy_scalar_hint(self):
        with self.assertRaises(ValueError) as caught:
            self.hls(qa_table_options={"exclude_high_aerosol": np.bool_(True)})
        self.assertIn("takes a bool; pass bool(value) for a numpy scalar", str(caught.exception))
        for value in (None, 1, np.int64(1)):
            with self.subTest(value=value):
                with self.assertRaises(ValueError) as caught:
                    self.hls(qa_table_options={"exclude_high_aerosol": value})
                self.assertNotIn("pass bool(value)", str(caught.exception))

    def test_builder_rejecting_a_declared_option(self):
        # QA_TABLE_OPTIONS declaring an argument the binding does not take.
        declared = dict(ccdcpp.profiles.QA_TABLE_OPTIONS)
        declared["sentinel2_scl"] = {"exclude_haze": (bool, False)}
        with mock.patch.object(ccdcpp.profiles, "QA_TABLE_OPTIONS", declared):
            with self.assertRaises(ValueError) as caught:
                presets.sentinel2_l2a()
        message = str(caught.exception)
        self.assertTrue(message.startswith("SensorProfile 'sentinel2_l2a'"), message)
        self.assertIn("ccdcpp.qa_tables.sentinel2_scl rejects qa_table_options "
                      "{'exclude_haze': False}, which QA_TABLE_OPTIONS allows", message)

    def test_builder_name_is_a_str(self):
        for builder in (1, ["hls_fmask"], b"hls_fmask"):
            for options in ((), {"exclude_high_aerosol": True}):
                with self.subTest(builder=builder, options=options):
                    with self.assertRaises(ValueError) as caught:
                        dataclasses.replace(presets.landsat_c1_physical(),
                                            qa_table_builder=builder,
                                            qa_table_options=options)
                    self.assertIn(f"qa_table_builder is {builder!r}; it is a str or None",
                                  str(caught.exception))


class Bounds(unittest.TestCase):
    """The bounds are the configuration's, in physical units; a profile gives only the
    units they resolve with."""

    def test_the_profiles_resolve_pyccds_bounds(self):
        for preset, refl, therm in ((presets.landsat_c1_dn, 10000.0, (-9320.0, 7070.0)),
                                    (presets.landsat_c1_physical, 1.0, (-0.932, 0.707)),
                                    (presets.hls_l30, 1.0, (-0.932, 0.707))):
            with self.subTest(preset=preset.__name__):
                _, t, b, _ = ccdcpp.plan_of("ccdc", preset()).standard.chain
                self.assertEqual((t.min, t.max, t.inclusive), therm + ((False, False),))
                self.assertEqual((b.min, b.max, b.inclusive), (0.0, refl, (False, False)))

    def test_the_flags_reach_the_plan(self):
        base = presets.landsat_c1_physical()
        for refl in ((True, True), (False, False), (True, False), (False, True)):
            for therm in ((False, False), (True, True), (False, True)):
                with self.subTest(refl=refl, therm=therm):
                    cfg = ccdcpp.config("ccdc").replace(
                        preprocess__band_range__min_inclusive=refl[0],
                        preprocess__band_range__max_inclusive=refl[1],
                        preprocess__thermal_range__min_inclusive=therm[0],
                        preprocess__thermal_range__max_inclusive=therm[1])
                    plan = ccdcpp.plan_of(cfg, base)
                    self.assertEqual(plan.standard.chain[1].inclusive, therm)
                    self.assertEqual(plan.standard.chain[2].inclusive, refl)
                    self.assertEqual(plan.snow.chain[1].inclusive, refl)

    def test_units_hashable_and_picklable(self):
        for u in (ccdcpp.DN, ccdcpp.REFLECTANCE, ccdcpp.KELVIN_TENTHS,
                  ccdcpp.CELSIUS_OVER_100):
            self.assertEqual(pickle.loads(pickle.dumps(u)), u)
            self.assertEqual(hash(u), hash(dataclasses.replace(u)))
        self.assertNotEqual(ccdcpp.DN, ccdcpp.REFLECTANCE)
        self.assertEqual(ccdcpp.Units(1.0), ccdcpp.REFLECTANCE)
        self.assertEqual(ccdcpp.Thermal(1.0, 0.0, 1.0), ccdcpp.CELSIUS_OVER_100)
        # Units(scale, offset), as Thermal(scale, offset, at_100c): the Input's
        # reflectance_scale and reflectance_offset.
        h = presets.landsat_c1_dn().input()
        self.assertEqual(ccdcpp.Units(scale=10000.0, offset=0.0), ccdcpp.DN)
        self.assertEqual((h.reflectance_scale, h.reflectance_offset),
                         (ccdcpp.DN.scale, ccdcpp.DN.offset))


class DateOrder(unittest.TestCase):
    def test_profile_entry_points_raise(self):
        dates = DATES.copy()
        dates[3], dates[4] = dates[4], dates[3]
        profile = presets.landsat_c1_dn()
        with self.assertRaises(ValueError) as caught:
            ccdcpp.detect(dates, CUBE_DN[0, 0], CODES[0, 0], encoding=profile)
        self.assertIn("dates must be non-decreasing", str(caught.exception))
        with self.assertRaises(ValueError):
            ccdcpp.detect_cube(dates, CUBE_DN, CODES, threads=1, encoding=profile)


class ProfileAndConfig(unittest.TestCase):
    def test_a_profile_is_its_input(self):
        # encoding=profile runs profile.input(); config= is the configuration
        # as given, the same with or without a profile.
        h_phys, cfg_phys = explicit_physical()
        s, q = CUBE_PHYS[0, 0], CODES[0, 0]
        profile = presets.landsat_c1_physical()
        for cfg in (ccdcpp.config("ccdc"), cfg_phys):
            assert_same(self, without(ccdcpp.detect(DATES, s, q, encoding=profile, config=cfg),
                                      "band_names"),
                        EXT.detect(DATES, s, q, encoding=h_phys, config=cfg))
        # The default is "ccdc", whatever the profile.
        assert_same(self, ccdcpp.detect(DATES, s, q, encoding=profile),
                    ccdcpp.detect(DATES, s, q, encoding=profile, config=ccdcpp.config("ccdc")))


class BandNames(unittest.TestCase):
    def test_only_with_a_profile(self):
        h = ccdcpp.Input()
        names = presets.landsat_c1_dn().band_names
        plain = every_entry(DATES, CUBE_DN, CODES, encoding=h)
        profiled = every_entry(DATES, CUBE_DN, CODES, encoding=presets.landsat_c1_dn())
        for out in plain["detect"] + [plain["detect_cube"]]:
            self.assertNotIn("band_names", out)
        for out in profiled["detect"] + [profiled["detect_cube"]]:
            self.assertEqual(out["band_names"], names)
            self.assertIsInstance(out["band_names"], tuple)


class Unprofiled(unittest.TestCase):
    """With an Input the package entry points are the extension's."""

    def test_same_output_every_call_form(self):
        h = ccdcpp.Input()
        d, s, q = DATES, CUBE_DN, CODES
        expected = without(EXT.detect_cube(d, s, q, encoding=h, threads=1), "threads")
        forms = {
            # The millworks nlcd plugin's call.
            "keywords": lambda: ccdcpp.detect_cube(
                dates=d, spectral=s, qas=q, encoding=h, threads=1),
            "positional": lambda: ccdcpp.detect_cube(d, s, q, encoding=h, threads=1),
            # "ccdc" is the run without config=.
            "config='ccdc'": lambda: ccdcpp.detect_cube(d, s, q, encoding=h, threads=1,
                                                        config="ccdc"),
            "config=Config": lambda: ccdcpp.detect_cube(d, s, q, encoding=h, threads=1,
                                                        config=ccdcpp.config("ccdc")),
        }
        for name, call in forms.items():
            with self.subTest(form=name):
                assert_same(self, without(call(), "threads"), expected)
        assert_same(self, ccdcpp.detect(d, s[0, 0], q[0, 0], encoding=h),
                    EXT.detect(d, s[0, 0], q[0, 0], encoding=h))
        raw = np.vectorize(C2_RAW.get, otypes=[np.uint16])(q)
        decoding = with_table(h, ccdcpp.qa_tables.landsat_c2_qa_pixel())
        assert_same(self, ccdcpp.detect(d, s[0, 0], raw[0, 0], encoding=decoding),
                    EXT.detect(d, s[0, 0], raw[0, 0], encoding=decoding))

    def test_signatures(self):
        positional = ["dates", "spectral", "qas"]
        for name, keywords in (
            ("detect", ["encoding", "config", "record", "resume", "band_names", "trace"]),
            ("detect_cube", ["encoding", "config", "threads", "record", "resume", "band_names",
                             "trace_pixels", "masks"]),
        ):
            with self.subTest(entry=name):
                params = inspect.signature(getattr(ccdcpp, name)).parameters
                self.assertEqual(list(params), positional + keywords)
                kinds = [p.kind for p in params.values()]
                self.assertEqual(kinds, [inspect.Parameter.POSITIONAL_OR_KEYWORD] * len(positional)
                                 + [inspect.Parameter.KEYWORD_ONLY] * len(keywords))
                self.assertIs(params["encoding"].default, inspect.Parameter.empty)
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
            ccdcpp.detect(d, s[0, 0], q[0, 0], h)
        with self.assertRaisesRegex(ValueError, "no configuration 'cdcc'"):
            ccdcpp.detect(d, s[0, 0], q[0, 0], encoding=h, config="cdcc")
        with self.assertRaises(TypeError):
            ccdcpp.detect(d, s[0, 0], q[0, 0], encoding=h, config=ccdcpp.config("ccdc").to_dict())

    def test_input_required(self):
        with self.assertRaises(TypeError):
            ccdcpp.detect(DATES, CUBE_DN[0, 0], CODES[0, 0])
        with self.assertRaises(TypeError):
            ccdcpp.detect_cube(DATES, CUBE_DN, CODES, config="ccdc")
        with self.assertRaises(TypeError):
            ccdcpp.detect_cube(DATES, CUBE_DN, CODES, encoding=None)
        # encoding= alone is a whole call: the configuration defaults to "ccdc".
        self.assertEqual(ccdcpp.detect_cube(DATES, CUBE_DN, CODES, encoding=ccdcpp.Input(),
                                            threads=1)["failures"], [])

    def test_cube_masks_per_pixel(self):
        H, W = CUBE_DN.shape[:2]
        out = ccdcpp.detect_cube(DATES, CUBE_DN, CODES, encoding=presets.landsat_c1_dn(),
                                 threads=2, masks=True)
        self.assertEqual(out["processing_mask"].shape, (H, W, len(DATES)))
        pixels = ccdcpp.to_pixels(out)
        self.assertGreater(len(pixels), 0)
        for p in pixels:
            one = ccdcpp.detect(DATES, CUBE_DN[p["row"], p["col"]], CODES[p["row"], p["col"]],
                                encoding=presets.landsat_c1_dn())
            np.testing.assert_array_equal(p["processing_mask"], one["processing_mask"])
            self.assertEqual(p["models"], one["models"])
        self.assertNotIn("processing_mask", ccdcpp.to_pixels(
            ccdcpp.detect_cube(DATES, CUBE_DN, CODES, encoding=ccdcpp.Input(), threads=2))[0])

    def test_extension_errors_pass_through(self):
        h = ccdcpp.Input()
        with self.assertRaises(ValueError) as caught:
            ccdcpp.detect(DATES, CUBE_DN[0, 0, :6], CODES[0, 0], encoding=h)
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
                words = ("encoding", "config", "record", "resume", "band_names") \
                    if name != "to_pixels" else ("detect_cube", "models")
                for word in words:
                    self.assertIn(word, text)
                self.assertIs(pickle.loads(pickle.dumps(entry)), entry)
        self.assertIn("SensorProfile", pydoc.render_doc(ccdcpp.SensorProfile))


class Profiles(unittest.TestCase):
    def test_immutable_hashable_picklable(self):
        for preset in (presets.landsat_c1_dn, presets.landsat_c1_physical, presets.landsat_c2,
                       ten_band_profile):
            with self.subTest(preset=preset.__name__):
                p = preset()
                self.assertEqual(p, preset())
                self.assertEqual(hash(p), hash(preset()))
                q = pickle.loads(pickle.dumps(p))
                self.assertEqual(q, p)
                self.assertEqual(fields(q.input()), fields(p.input()))
                with self.assertRaises(dataclasses.FrozenInstanceError):
                    p.saturation_code = 2.0

    def test_input_is_built_once_and_immutable(self):
        p = presets.landsat_c1_dn()
        self.assertIs(p.input(), p.input())
        self.assertIs(p.input(), presets.landsat_c1_dn().input())
        with self.assertRaises(AttributeError):
            p.input().reflectance_scale = 99.0
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
                    other = ccdcpp.Input(**{name: value})
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
        stages = cfg.resolve(h)["stages"]
        self.assertEqual(stages["start_search"]["params"]["meow_size"], 18)
        self.assertEqual(stages["edge_fit"]["params"]["min_observations"], 18)
        self.assertEqual(cfg.shared()["statistics_end"], ccdcpp.NO_STATISTICS_END)
        self.assertEqual(cfg.resolve(h)["shared"]["statistics_end"], 2**63 - 1)
        out = every_entry(DATES, CUBE_DN, CODES, encoding=presets.landsat_c1_dn(), config=cfg)
        self.assertEqual(out["detect_cube"]["failures"], [])

    def test_change_probability_moves_the_threshold(self):
        cfg = ccdcpp.config("ccdc").replace(shared__change_probability=0.995)
        self.assertEqual(set(option_params(cfg, "change_threshold").values()),
                         {ccdcpp.chi2_quantile(0.995, 5)})
        self.assertEqual(resolved(cfg)["change_test"]["params"]["change_probability"], 0.995)
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


#------------------------------------------------------------------------------
# A ten-band, thermal-less, non-Landsat profile, defined here and not shipped:
# a Sentinel-2-like layout in reflectance with SCL QA.
#------------------------------------------------------------------------------

def ten_band_profile():
    return ccdcpp.SensorProfile(
        name="test_ten_band",
        band_names=("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"),
        roles=("blue", "green", "red", "red_edge_1", "red_edge_2", "red_edge_3", "nir_broad",
               "nir", "swir1", "swir2"),
        units=ccdcpp.REFLECTANCE,
        qa_table_builder="sentinel2_scl",
    )


#: The configuration the ten-band tests run: six detection bands by role, a
#: 10-day reference cadence.
TEN_BAND_CONFIG = ccdcpp.config("ccdc").replace(
    change_test__peek_reference_days=10.0,
    shared__detection_bands=["green", "red", "red_edge_2", "nir", "swir1", "swir2"],
)


#: Raw SCL values: 4 vegetation, 5 bare (clear); 6 water; 8, 9 cloud; 3 shadow;
#: 0 no data.
SCL_VALUES = np.array([4, 5, 6, 8, 9, 3, 0], dtype=np.uint8)


def ten_band_inputs(H=2, W=2, seed=5):
    """(dates, spectral (H, W, 10, T) reflectance, raw SCL (H, W, T)): a 5-day
    series from 2018, after the Landsat statistics_end, with a break per pixel."""
    rng = np.random.default_rng(seed)
    dates = (736695 + 5 * np.arange(4 * 73)).astype(np.int64)
    T = dates.size
    phase = 2 * np.pi * (dates - dates[0]) / 365.25
    spectral = np.empty((H, W, 10, T))
    for p in range(H * W):
        r, c = divmod(p, W)
        base = (0.04 + 0.03 * np.arange(10))[:, None]
        season = (0.02 + 0.005 * p) * np.sin(phase + p)[None, :]
        level = np.where(np.arange(T) > T // 2 + 9 * p, 0.08, 0.0)[None, :]
        spectral[r, c] = base + season + level + rng.normal(0, 0.004, (10, T))
    scl = rng.choice(SCL_VALUES, size=(H, W, T), p=[0.5, 0.2, 0.05, 0.1, 0.05, 0.05, 0.05])
    return dates, spectral, scl


class TenBand(unittest.TestCase):
    def test_end_to_end(self):
        profile = ten_band_profile()
        dates, cube, scl = ten_band_inputs()
        out = ccdcpp.detect_cube(dates, cube, scl, threads=2, encoding=profile,
                                 config=TEN_BAND_CONFIG)
        self.assertEqual(out["failures"], [])
        self.assertEqual(out["num_bands"], 10)
        self.assertEqual(out["band_names"], profile.band_names)
        n = len(out["row"])
        self.assertGreater(n, 4)  # every pixel breaks
        self.assertEqual(set(out["curve_qa"].tolist()), {8})  # Standard, every segment
        self.assertEqual(out["coefficients"].shape, (n, 10, 7))
        self.assertTrue(np.isfinite(out["rmse"]).all())

        one = ccdcpp.detect(dates, cube[1, 0], scl[1, 0], encoding=profile, config=TEN_BAND_CONFIG)
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
            shared__statistics_end=None,
        )
        self.assertEqual(TEN_BAND_CONFIG, cfg)
        self.assertEqual(profile.input(), with_table(h, ccdcpp.qa_tables.sentinel2_scl(h)))
        r = cfg.resolve(h)
        self.assertEqual(r["shared"]["detection_bands"], [1, 2, 4, 7, 8, 9])
        self.assertEqual(r["stages"]["window_screen"]["params"]["bands"], [1, 8])
        self.assertEqual(set(option_params(cfg, "change_threshold", h).values()),
                         {ccdcpp.chi2_quantile(0.99, 6)})
        table = ccdcpp.qa_tables.sentinel2_scl(h)
        assert_same(self,
                    strip_band_names(every_entry(dates, cube, scl, encoding=profile, config=cfg)),
                    every_entry(dates, cube, scl, encoding=with_table(h, table), config=cfg))

    def test_documented_example_runs(self):
        # The profile in ccdcpp.profiles' docstring, which is not shipped.
        doc = ccdcpp.profiles.__doc__
        self.assertIn("NOT SHIPPED", doc)
        code = doc.split("::\n", 1)[1].split("\n\nThe configuration names its detection", 1)[0]
        namespace = {}
        exec(compile(inspect.cleandoc("\n" + code), "<profiles docstring>", "exec"), namespace)
        profile = namespace["landsat_oli_c2"]()
        self.assertFalse(hasattr(presets, "landsat_oli_c2"))
        h = profile.input()
        self.assertEqual((h.bands(), h.index("green"), h.index("thermal")), (8, 2, 7))
        shared = ccdcpp.config("ccdc").resolve(h)["shared"]
        self.assertEqual(shared["detection_bands"], [2, 3, 4, 5, 6])
        self.assertEqual(shared["statistics_end"], ccdcpp.NO_STATISTICS_END)
        np.testing.assert_array_equal(profile.input().qa_table.codes,
                                      ccdcpp.qa_tables.landsat_c2_qa_pixel(h).codes)

    def test_statistics_end_matters_after_2017(self):
        dates, cube, scl = ten_band_inputs()
        landsat = TEN_BAND_CONFIG.replace(shared__statistics_end=736694)
        out = ccdcpp.detect_cube(dates, cube, scl, threads=2, encoding=ten_band_profile(),
                                 config=landsat)
        self.assertNotEqual(set(out["curve_qa"].tolist()), {8})


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


def optical_cube(dates, bands, H, W, rng, breaks=True):
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


def hls_inputs(bands, thermal=False, step=8, start=L30_START, years=7, H=2, W=2, seed=3):
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


def s2_inputs(H=2, W=2, seed=9, breaks=True, years=8):
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
EXPECTED = {
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
NEW_PRESETS = (presets.hls_l30, presets.hls_s30, presets.hls, presets.sentinel2_l2a)


def hand_built(name):
    """(Input, QaTable) of preset ``name``, by hand."""
    h = ccdcpp.Input(**EXPECTED[name])
    return h, getattr(ccdcpp.qa_tables, BUILDERS[name])(h)


def preset_inputs(name):
    """(dates, spectral, raw QA) for preset ``name``, in its units."""
    if name == "hls_l30":
        return hls_inputs(8, thermal=True)[:3]
    if name == "hls_s30":
        return hls_inputs(11, step=5, start=S2_START)[:3]
    if name == "hls":
        return hls_inputs(7, step=3)[:3]
    return s2_inputs()[:3]


def models(out):
    """The fitted models of a detect output, without the per-date mask."""
    return without(out, "processing_mask")


class SensorPresets(unittest.TestCase):
    def test_options_are_the_written_out_options(self):
        defaults = ccdcpp.config("ccdc")
        for preset in NEW_PRESETS:
            name = preset.__name__
            with self.subTest(preset=name):
                h, cfg = preset().input(), defaults
                expected, table = hand_built(name)
                self.assertEqual(fields(h), fields(with_table(expected, table)))
                r = cfg.resolve(h)
                self.assertEqual(r["stages"]["window_screen"]["params"]["bands"], TMASK[name])
                self.assertEqual(r["shared"]["detection_bands"], DETECTION[name])
                self.assertEqual(set(option_params(cfg, "alpha", h).values()), {1e-4})
                for threshold in (change_threshold, outlier_threshold):
                    self.assertEqual(threshold(cfg, h).hex(), threshold(defaults).hex())
                self.assertEqual(r["shared"]["statistics_end"], ccdcpp.NO_STATISTICS_END)
                self.assertEqual(preset().band_names, BAND_NAMES[name])
                self.assertEqual(preset().qa_table_options,
                                 dict(sentinel2_l2a=()).get(
                                     name, (("exclude_high_aerosol", False),)))
                np.testing.assert_array_equal(preset().input().qa_table.codes, table.codes)

    def test_band_range_tests_the_rows_the_profiles_listed(self):
        # Each shipped profile listed the rows a saturated value can appear
        # in, in row order: its every row but thermal, which is what "ccdc"'s
        # band_range tests ("reflective"), in every procedure.
        listed = {"landsat_c1_dn": list(range(6)), "landsat_c1_physical": list(range(6)),
                  "landsat_c2": list(range(6)), "hls_l30": list(range(7)),
                  "hls_s30": list(range(11)), "hls": list(range(7)),
                  "sentinel2_l2a": list(range(10))}
        self.assertEqual(sorted(listed), sorted(presets.__all__))
        for name, rows in listed.items():
            with self.subTest(preset=name):
                plan = ccdcpp.plan_of("ccdc", getattr(presets, name)())
                for procedure in ccdcpp.PROCEDURES:
                    steps = [s for s in getattr(plan, procedure).chain
                             if s.kind == ccdcpp.StepKind.BandRange]
                    self.assertEqual([s.bands for s in steps], [rows])

    def test_red_edge_and_broad_nir_are_fitted_not_detected(self):
        for preset in (presets.hls_s30, presets.sentinel2_l2a):
            with self.subTest(preset=preset.__name__):
                p = preset()
                h = p.input()
                detection = ccdcpp.config("ccdc").resolve(h)["shared"]["detection_bands"]
                for role in ("red_edge_1", "red_edge_2", "red_edge_3", "nir_broad"):
                    self.assertNotIn(p.index(role), detection)
                    self.assertIn(p.index(role), band_range_rows(h))
                self.assertEqual(p.band_names[p.index("nir")], "B8A")
                self.assertEqual(p.band_names[p.index("nir_broad")], "B08")

    def test_docstrings(self):
        for preset in NEW_PRESETS:
            with self.subTest(preset=preset.__name__):
                doc = preset.__doc__
                self.assertNotIn("not validated", doc.lower())
                self.assertIn("Input:", doc)
                self.assertIn("QA:", doc)
                self.assertIn(BUILDERS[preset.__name__], doc)
                self.assertEqual(preset().name, preset.__name__)
                for band in BAND_NAMES[preset.__name__]:
                    self.assertIn(band, doc)
        self.assertNotIn("not validated", presets.__doc__.lower())
        self.assertIn("BOA_ADD_OFFSET", presets.sentinel2_l2a.__doc__)
        self.assertIn("put L30 before S30", presets.hls.__doc__)
        # S30 detects on bands HLS adjusts to OLI, but B01 and B02 are
        # adjusted too, and Sentinel-2C is part of the cadence.
        self.assertIn("HLS\n    adjusts these, and B01 and B02, to the OLI bandpasses",
                      presets.hls_s30.__doc__)
        for preset in (presets.hls_s30, presets.sentinel2_l2a):
            with self.subTest(preset=preset.__name__):
                self.assertIn("2C from 2024", preset.__doc__)

    def test_docstrings_state_the_validation(self):
        def flat(text):
            return " ".join(text.split())
        # The record every one was validated on.
        for preset in NEW_PRESETS:
            with self.subTest(preset=preset.__name__):
                doc = flat(preset.__doc__)
                self.assertIn("Validation: one cell of MGRS tile T16SGC near Atlanta", doc)
                self.assertIn("2020-2023, read from Planetary Computer", doc)
        self.assertIn("2020-2023, read from Planetary Computer, four years of one cell",
                      flat(presets.__doc__))
        # hls_l30 is promoted outright; the summary line carries no caveat.
        self.assertEqual(presets.hls_l30.__doc__.splitlines()[0],
                         "HLS v2.0 L30 (Landsat 8/9 OLI/TIRS at 30 m).")
        self.assertIn("kappa 0.44", flat(presets.hls_l30.__doc__))
        # S30 and hls: the Fmask-over-bright-surfaces caveat, with numbers.
        for preset in (presets.hls_s30, presets.hls):
            with self.subTest(preset=preset.__name__):
                doc = flat(preset.__doc__)
                self.assertIn("Validated, with a caveat: Fmask over bright surfaces.",
                              flat(doc.split("\n\n")[0]))
                self.assertIn("Caveat, Fmask over bright surfaces", doc)
        s30 = flat(presets.hls_s30.__doc__)
        self.assertIn("Over all 1,862 quarry (land cover 31) pixels on the HLS grid it flags "
                      "63% of S30 observations as cloud, and 1,145 of those pixels go to the "
                      "InsufficientClear procedure (68% cloud over them)", s30)
        self.assertIn("170 of Landsat ARD's 173 such pixels, and both of L30's 2", s30)
        self.assertNotIn("61%", s30)
        self.assertIn("85 same-date pairs (84 dates", flat(presets.hls.__doc__))
        self.assertIn("apart from 24 water pixels (0.60% of water pixels against a 0.08% "
                      "baseline)", flat(presets.sentinel2_l2a.__doc__))
        self.assertIn("1,741 of 300,288 pixels go to the InsufficientClear",
                      flat(presets.hls.__doc__))
        # Sentinel-2: median level, PC's processing, not ESA's.
        label = ("Validated at median level on Planetary Computer's Esri/MSFT/S2RP L2A, "
                 "2020–23; ESA's own processing not sampled.")
        s2 = flat(presets.sentinel2_l2a.__doc__)
        self.assertTrue(s2.startswith("Sentinel-2 L2A (ESA Sen2Cor), native bands. " + label), s2)
        self.assertIn(label, flat(presets.__doc__).replace(":func:`sentinel2_l2a` is v", "V"))
        # SCL 7 stays cloud; the water trade-off with the measured counts.
        self.assertIn("7-10 cloud (7, unclassified, included)", s2)
        self.assertIn("sends 1,355 of 3,983 water pixels, and 3,426 pixels in all", s2)
        self.assertIn("decoding 7 as fill would send 2,545, as clear 587. It stays cloud.", s2)
        # The measured hls cadence, not the 2-4 days once documented.
        hls_doc = flat(presets.hls.__doc__)
        self.assertIn("median gap of 4-5 days, and the peek window grew to 19-24", hls_doc)
        self.assertNotIn("2-4 days", hls_doc)
        self.assertNotIn("32-48", hls_doc)

    def test_docstrings_are_wrapped(self):
        for obj in [presets] + [getattr(presets, name) for name in presets.__all__]:
            with self.subTest(name=obj.__name__):
                lines = obj.__doc__.splitlines()
                for line in lines:
                    self.assertLessEqual(len(line), 79, line)
                # Prose is filled: a line is not broken where the next word,
                # or the next ``literal`` whole, fits with room to spare.
                for line, following in zip(lines, lines[1:]):
                    indent = len(line) - len(line.lstrip())
                    if (not line.strip() or not following.strip() or indent > 4
                            or len(following) - len(following.lstrip()) != indent
                            or line.endswith(":")):
                        continue
                    word = following.split()[0]
                    if word.startswith("``"):
                        rest = following.strip()
                        end = rest.find("``", 2)
                        word = rest if end < 0 else rest[:end + 2]
                    self.assertGreater(len(line) + 1 + len(word), 70, (line, following))

    def test_profiles_are_hashable_and_picklable(self):
        for preset in NEW_PRESETS:
            with self.subTest(preset=preset.__name__):
                p = preset()
                self.assertEqual(hash(p), hash(preset()))
                self.assertEqual(pickle.loads(pickle.dumps(p)), p)

    def test_is_the_hand_built_run(self):
        for preset in NEW_PRESETS:
            name = preset.__name__
            with self.subTest(preset=name):
                dates, cube, qa = preset_inputs(name)
                h, table = hand_built(name)
                profiled = every_entry(dates, cube, qa, threads=2, encoding=preset())
                assert_same(self, strip_band_names(profiled),
                            every_entry(dates, cube, qa, threads=2,
                                        encoding=with_table(h, table), config="ccdc"))
                cols = profiled["detect_cube"]
                self.assertEqual(cols["failures"], [])
                self.assertEqual(cols["band_names"], BAND_NAMES[name])
                self.assertEqual(set(cols["curve_qa"].tolist()), {8})  # Standard
                self.assertGreater(len(cols["row"]), 4)  # every pixel breaks

    def test_raw_qa_is_the_decoded_codes(self):
        for preset, values, codes in ((presets.hls_l30, FMASK_VALUES, FMASK_CODES),
                                      (presets.sentinel2_l2a, SCL_ALL, SCL_CODES)):
            with self.subTest(preset=preset.__name__):
                np.testing.assert_array_equal(preset().input().qa_table.codes[values], codes)
                dates, cube, qa = preset_inputs(preset.__name__)
                lookup = dict(zip(values.tolist(), codes.tolist()))
                decoded = np.vectorize(lookup.get, otypes=[np.uint8])(qa)
                assert_same(self,
                            strip_band_names(every_entry(dates, cube, qa, threads=2,
                                                         encoding=preset())),
                            every_entry(dates, cube, decoded, threads=2,
                                        encoding=with_table(preset().input(), None)))

    def test_permutation(self):
        order = [10, 0, 9, 1, 8, 2, 7, 3, 6, 4, 5]
        base = presets.hls_s30()
        dates, cube, fmask = preset_inputs("hls_s30")
        expected = every_entry(dates, cube, fmask, threads=2, encoding=base)
        got = every_entry(dates, np.ascontiguousarray(cube[:, :, order]), fmask, threads=2,
                          encoding=permuted(base, order))
        assert_same(self, unpermute(got, order), expected)


#: The band order of a Sentinel-2 store that keeps the 10 m bands first
#: (B08 before the red edges), as the validation sample's store does.
S2_STORE_ORDER = ("B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12")


class BandNamesCheck(unittest.TestCase):
    """band_names= on the entry points: the stack's row names, which must be
    the profile's in the profile's order."""

    def entries(self, dates, cube, qa, **kw):
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
                expected = every_entry(dates, cube, qa, threads=2, encoding=profile)
                for names in (profile.band_names, list(profile.band_names),
                              np.array(profile.band_names)):
                    assert_same(self, every_entry(dates, cube, qa, threads=2, encoding=profile,
                                                  band_names=names), expected)

    def test_refuses_a_misordered_sentinel2_stack(self):
        profile = presets.sentinel2_l2a()
        dates, cube, scl = preset_inputs("sentinel2_l2a")
        store = [profile.band_names.index(b) for b in S2_STORE_ORDER]
        stacked = np.ascontiguousarray(cube[:, :, store])
        for name, call in self.entries(dates, stacked, scl, encoding=profile,
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
        silent = ccdcpp.detect_cube(dates, stacked, scl, threads=2, encoding=profile)
        right = ccdcpp.detect_cube(dates, cube, scl, threads=2, encoding=profile)
        self.assertEqual(silent["failures"], [])
        self.assertNotEqual(str(without(silent, "threads")), str(without(right, "threads")))
        # Reordered by the names, it is the right run.
        order = [S2_STORE_ORDER.index(b) for b in profile.band_names]
        assert_same(self, ccdcpp.detect_cube(
            dates, np.ascontiguousarray(stacked[:, :, order]), scl, threads=2,
            encoding=profile, band_names=tuple(np.array(S2_STORE_ORDER)[order])), right)

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
                    ccdcpp.detect(dates, cube[0, 0], fmask[0, 0], encoding=profile,
                                  band_names=given)
                self.assertIn(text, str(caught.exception))
                self.assertIn("the profile has B01, B02", str(caught.exception))
        for given in ("B01B02B03B04B05B06B07B10", b"B01", ("B01",) * 7 + (10,)):
            with self.subTest(given=given):
                with self.assertRaises(TypeError):
                    ccdcpp.detect(dates, cube[0, 0], fmask[0, 0], encoding=profile,
                                  band_names=given)

    def test_needs_a_profile(self):
        h = ccdcpp.Input()
        names = presets.landsat_c1_dn().band_names
        for name, call in self.entries(DATES, CUBE_DN, CODES, encoding=h,
                                       band_names=names).items():
            with self.subTest(entry=name):
                with self.assertRaises(TypeError) as caught:
                    call()
                self.assertIn("pass encoding= a ccdcpp.SensorProfile", str(caught.exception))

    def test_check_takes_band_names(self):
        profile = presets.sentinel2_l2a()
        profile.check(np.zeros((10, 3)), profile.band_names)
        with self.assertRaises(ValueError):
            profile.check(np.zeros((10, 3)), S2_STORE_ORDER)
        # The count is checked first.
        with self.assertRaises(ValueError) as caught:
            profile.check(np.zeros((9, 3)), profile.band_names)
        self.assertIn("has 9 bands", str(caught.exception))


class HlsInput(unittest.TestCase):
    """The HLS presets take every int16 layer times 0.0001."""

    def run_pixel(self, profile, dates, cube, fmask, keep=None):
        if keep is not None:
            dates, cube, fmask = dates[keep], cube[..., keep], fmask[..., keep]
        return ccdcpp.detect(dates, np.ascontiguousarray(cube), fmask, encoding=profile)

    def test_scale(self):
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

    def test_coastal_is_screened(self):
        # 10000 (1.0) is out of the exclusive reflectance range, 9999 in.
        dates, cube, fmask, raw = hls_inputs(8, thermal=True, H=1, W=1)
        clear = np.flatnonzero(fmask[0, 0] == 0)
        high, edge = clear[7::40], clear[27::40]
        raw = raw[0, 0].copy()
        raw[0, high] = 10000
        raw[0, edge] = 9999
        out = self.run_pixel(presets.hls_l30(), dates, raw * 1e-4, fmask[0, 0])
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
                                         encoding=presets.sentinel2_l2a())
        self.assertEqual(out["failures"], [])
        self.assertGreater(len(out["row"]), 4)

    def test_uncorrected_input_differs(self):
        # dn / 10000 without the offset: 0.1 too bright from 2022-01-25.
        dates, cube, scl, dn = s2_inputs(breaks=False)
        profile = presets.sentinel2_l2a()
        corrected = ccdcpp.detect_cube(dates, cube, scl, threads=2, encoding=profile)
        raw = ccdcpp.detect_cube(dates, dn / 10000.0, scl, threads=2, encoding=profile)
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
                            encoding=presets.sentinel2_l2a())
        self.assertFalse(out["processing_mask"][before].any())
        self.assertFalse(out["processing_mask"][after].any())
        self.assertTrue(out["processing_mask"][clear[1::30]].all())

if __name__ == "__main__":
    unittest.main()
