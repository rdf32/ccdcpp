"""Sensor profiles, the Landsat presets and the profile= keyword.

usage: python test_profiles.py <module_dir> <source_dir> [unittest args]

<module_dir> holds the built extension; <source_dir> is the repository root.
The package is assembled as the wheel lays it out, python/ccdcpp/*.py next to
the extension, in a temporary directory. ctest runs this as
`python_profiles`.
"""
import atexit
import dataclasses
import glob
import inspect
import os
import pickle
import pydoc
import shutil
import sys
import tempfile
import unittest
import warnings

MODULE_DIR, SOURCE_DIR = sys.argv[1], sys.argv[2]
del sys.argv[1:3]

_ROOT = tempfile.mkdtemp(prefix="ccdcpp-profiles-")
atexit.register(shutil.rmtree, _ROOT, True)
_PACKAGE = os.path.join(_ROOT, "ccdcpp")
os.mkdir(_PACKAGE)
for _source in glob.glob(os.path.join(SOURCE_DIR, "python", "ccdcpp", "*.py")):
    shutil.copy(_source, _PACKAGE)
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
    path = os.path.join(SOURCE_DIR, "tests", "ccd", "procedure", "test_3657_3610_observations.csv")
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


def explicit_physical():
    """The physical-unit preset written out, as the golden tests write it."""
    h, lo = ccdcpp.HarmonicOptions(), ccdcpp.LassoOptions()
    h.THERMAL_SCALE = 1.0
    h.THERMAL_OFFSET = 0.0
    h.REFL_MIN = 0.0 / SCALE
    h.REFL_MAX = 10000.0 / SCALE
    h.THERM_MIN = -9320.0 / SCALE
    h.THERM_MAX = 7070.0 / SCALE
    h.MEDIAN_GREEN_FILTER = 400.0 / SCALE
    lo.alpha = 1.0 / SCALE
    lo.coef_floor = 1.0 / SCALE
    return h, lo


DATES, CUBE_DN, CODES = corpus_cube()
CUBE_PHYS = to_physical(CUBE_DN)


def every_entry(dates, spectral, qas, threads=8, **kw):
    """{entry: output} for detect on every pixel, detect_cube, detect_cube_columns."""
    H, W = spectral.shape[:2]
    return {
        "detect": [ccdcpp.detect(dates, spectral[r, c], qas[r, c], **kw)
                   for r in range(H) for c in range(W)],
        "detect_cube": ccdcpp.detect_cube(dates, spectral, qas, threads=threads, **kw),
        "detect_cube_columns": without(
            ccdcpp.detect_cube_columns(dates, spectral, qas, threads=threads, **kw), "threads"),
    }


def strip_band_names(outs):
    return {
        "detect": [without(o, "band_names") for o in outs["detect"]],
        "detect_cube": outs["detect_cube"],
        "detect_cube_columns": without(outs["detect_cube_columns"], "band_names"),
    }


class Corpus(unittest.TestCase):
    def test_corpus_runs_every_procedure(self):
        h, lo = ccdcpp.HarmonicOptions(), ccdcpp.LassoOptions()
        out = EXT.detect_cube_columns(DATES, CUBE_DN, CODES, h, lo, 1)
        self.assertEqual(out["failures"], [])
        # Standard, PermanentSnow and InsufficientClear.
        self.assertTrue({8, 54, 44} <= set(out["curve_qa"].tolist()), out["curve_qa"])
        self.assertGreater(len(out["row"]), 6)


class LandsatC1DN(unittest.TestCase):
    def test_options_are_the_defaults_field_by_field(self):
        h, lo = presets.landsat_c1_dn().options()
        self.assertEqual(fields(h), fields(ccdcpp.HarmonicOptions()))
        self.assertEqual(fields(lo), fields(ccdcpp.LassoOptions()))
        for name in ("CHANGE_THRESHOLD", "OUTLIER_THRESHOLD"):
            self.assertEqual(getattr(h, name).hex(), getattr(ccdcpp.HarmonicOptions(), name).hex())
        self.assertGreater(len(fields(h)), 40)
        self.assertIsNone(presets.landsat_c1_dn().qa_table())

    def test_output_is_the_unprofiled_output(self):
        h, lo = ccdcpp.HarmonicOptions(), ccdcpp.LassoOptions()
        profiled = every_entry(DATES, CUBE_DN, CODES, profile=presets.landsat_c1_dn())
        plain = every_entry(DATES, CUBE_DN, CODES, hoptions=h, loptions=lo)
        assert_same(self, strip_band_names(profiled), plain)


class LandsatPhysical(unittest.TestCase):
    def test_options_are_the_explicit_physical_options(self):
        h, lo = explicit_physical()
        for preset in (presets.landsat_c1_physical, presets.landsat_c2):
            with self.subTest(preset=preset.__name__):
                ph, plo = preset().options()
                self.assertEqual(fields(ph), fields(h))
                self.assertEqual(fields(plo), fields(lo))

    def test_output_is_the_explicit_physical_output(self):
        h, lo = explicit_physical()
        profiled = every_entry(DATES, CUBE_PHYS, CODES, profile=presets.landsat_c1_physical())
        plain = every_entry(DATES, CUBE_PHYS, CODES, hoptions=h, loptions=lo)
        assert_same(self, strip_band_names(profiled), plain)

    def test_c2_raw_qa_pixel_is_the_decoded_codes(self):
        raw = np.vectorize(C2_RAW.get, otypes=[np.uint16])(CODES)
        profile = presets.landsat_c2()
        np.testing.assert_array_equal(profile.qa_table().codes[raw], CODES)
        decoded = every_entry(DATES, CUBE_PHYS, raw, profile=profile)
        codes = every_entry(DATES, CUBE_PHYS, CODES, profile=profile, qa_table=None)
        assert_same(self, decoded, codes)
        h, lo = explicit_physical()
        assert_same(self, strip_band_names(decoded),
                    every_entry(DATES, CUBE_PHYS, CODES, hoptions=h, loptions=lo))

    def test_presets_state_units_and_band_order(self):
        for preset in (presets.landsat_c1_dn, presets.landsat_c1_physical, presets.landsat_c2):
            with self.subTest(preset=preset.__name__):
                self.assertIn("Input:", preset.__doc__)
                self.assertIn("QA:", preset.__doc__)
                self.assertEqual(preset().band_names,
                                 ("blue", "green", "red", "nir", "swir1", "swir2", "thermal"))
                self.assertEqual(preset().name, preset.__name__)
        self.assertIn("blue, green, red, nir, swir1, swir2, thermal", presets.__doc__)
        self.assertEqual(presets.__all__, ["landsat_c1_dn", "landsat_c1_physical", "landsat_c2"])


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

    cols = dict(outs["detect_cube_columns"])
    for key in ("intercept", "coefficients", "rmse", "magnitude"):
        cols[key] = np.ascontiguousarray(cols[key][:, inv])
    cols["band_names"] = tuple(cols["band_names"][i] for i in inv)
    return {
        "detect": [dict(o, models=models(o["models"]),
                        band_names=tuple(o["band_names"][i] for i in inv))
                   for o in outs["detect"]],
        "detect_cube": [dict(p, models=models(p["models"])) for p in outs["detect_cube"]],
        "detect_cube_columns": cols,
    }


class Permutation(unittest.TestCase):
    def test_landsat(self):
        order = [6, 3, 0, 5, 1, 4, 2]
        base = presets.landsat_c1_dn()
        profile = permuted(base, order)
        h, _ = profile.options()
        self.assertEqual(h.GREEN_IDX, order.index(1))
        self.assertEqual(h.THERMAL_IDX, order.index(6))
        self.assertEqual(h.DETECTION_BANDS, [order.index(i) for i in (1, 2, 3, 4, 5)])
        expected = every_entry(DATES, CUBE_DN, CODES, profile=base)
        got = every_entry(DATES, np.ascontiguousarray(CUBE_DN[:, :, order]), CODES,
                          profile=profile)
        assert_same(self, unpermute(got, order), expected)

    def test_ten_bands(self):
        order = [9, 0, 8, 1, 7, 2, 6, 3, 5, 4]
        base = ten_band_profile()
        dates, cube, scl = ten_band_inputs()
        expected = every_entry(dates, cube, scl, profile=base)
        got = every_entry(dates, np.ascontiguousarray(cube[:, :, order]), scl,
                          profile=permuted(base, order))
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
            "detect": lambda: ccdcpp.detect(d, s[0, 0], q[0, 0], profile=profile),
            "detect_cube": lambda: ccdcpp.detect_cube(d, s, q, profile=profile),
            "detect_cube_columns": lambda: ccdcpp.detect_cube_columns(d, s, q, profile=profile),
        }
        for name, call in calls.items():
            with self.subTest(entry=name):
                with self.assertRaises(ValueError) as caught:
                    call()
                self.assertIn("SensorProfile 'landsat_c1_dn'", str(caught.exception))

    def test_not_a_profile(self):
        with self.assertRaises(TypeError):
            ccdcpp.detect(DATES, CUBE_DN[0, 0], CODES[0, 0], profile="landsat_c1_dn")

    def test_construction(self):
        base = presets.landsat_c1_dn()
        cases = [
            (dict(roles=base.roles[:6]), "7 band_names and 6 roles"),
            (dict(band_names=("a",) * 7), "band_names repeats"),
            (dict(roles=("blue",) * 7), "repeats a role other than 'other'"),
            (dict(roles=base.roles[:6] + ("infrared",)), "unknown roles ['infrared']"),
            (dict(roles=("blue", "other") + base.roles[2:]), "no 'green' band"),
            (dict(thermal=None), "a 'thermal' band and thermal= go together"),
            (dict(detection=("green", "coastal")), "detection names 'coastal'"),
            (dict(tmask=("green", "green")), "tmask repeats a role"),
            (dict(detection=()), "detection is empty"),
            (dict(tmask=()), "tmask is empty"),
            (dict(saturation=("green", "thermal")), "saturation holds 'thermal'"),
            (dict(cadence={"STAT_ORD": 1}), "cadence names 'STAT_ORD'"),
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
        other = dict(roles=base.roles[:5] + ("other", "thermal"),
                     detection=("green", "red", "nir", "swir1"))
        self.assertEqual(dataclasses.replace(base, **other, saturation=()).options()[0].NUM_BANDS, 7)
        with self.assertRaises(ValueError) as caught:
            dataclasses.replace(base, **other, saturation=("green", "other"))
        self.assertIn("saturation names 'other'", str(caught.exception))


class BoundFlags(unittest.TestCase):
    def test_defaults(self):
        for units in (ccdcpp.DN, ccdcpp.REFLECTANCE):
            self.assertEqual(units.refl_inclusive, (False, False))
        for thermal in (ccdcpp.KELVIN_TENTHS, ccdcpp.CELSIUS_HUNDREDTHS):
            self.assertEqual(thermal.inclusive, (False, False))

    def test_options_carry_the_flags(self):
        base = presets.landsat_c1_physical()
        for refl in ((True, True), (False, False), (True, False), (False, True)):
            for therm in ((False, False), (True, True), (False, True)):
                with self.subTest(refl=refl, therm=therm):
                    p = dataclasses.replace(
                        base,
                        units=dataclasses.replace(base.units, refl_inclusive=refl),
                        thermal=dataclasses.replace(base.thermal, inclusive=therm),
                    )
                    h = p.options()[0]
                    self.assertEqual((h.REFL_MIN_INCLUSIVE, h.REFL_MAX_INCLUSIVE), refl)
                    self.assertEqual((h.THERM_MIN_INCLUSIVE, h.THERM_MAX_INCLUSIVE), therm)

    def test_no_thermal_keeps_the_option_defaults(self):
        h = ten_band_profile().options()[0]
        default = ccdcpp.HarmonicOptions()
        self.assertEqual((h.THERM_MIN_INCLUSIVE, h.THERM_MAX_INCLUSIVE),
                         (default.THERM_MIN_INCLUSIVE, default.THERM_MAX_INCLUSIVE))

    def test_flags_are_two_bools(self):
        for bad in ((True,), (True, True, True), (1, 0), ("yes", "no"), True):
            with self.subTest(bad=bad):
                with self.assertRaises((ValueError, TypeError)):
                    ccdcpp.Units(0.0, 1.0, 0.04, refl_inclusive=bad)
                with self.assertRaises((ValueError, TypeError)):
                    ccdcpp.Thermal(1.0, 0.0, -0.932, 0.707, inclusive=bad)
        self.assertEqual(ccdcpp.Units(0.0, 1.0, 0.04, refl_inclusive=[False, True]).refl_inclusive,
                         (False, True))

    def test_hashable_and_picklable(self):
        u = ccdcpp.Units(0.0, 1.0, 0.04, refl_inclusive=(True, True))
        self.assertEqual(pickle.loads(pickle.dumps(u)), u)
        self.assertEqual(hash(u), hash(ccdcpp.Units(0.0, 1.0, 0.04, (True, True))))
        self.assertNotEqual(u, ccdcpp.REFLECTANCE)


class GreenFilterScopeField(unittest.TestCase):
    def test_default_is_stat_window(self):
        for preset in (presets.landsat_c1_dn, presets.landsat_c1_physical, presets.landsat_c2):
            with self.subTest(preset=preset.__name__):
                p = preset()
                self.assertEqual(p.green_filter_scope, "stat_window")
                self.assertEqual(p.options()[0].GREEN_FILTER_SCOPE, ccdcpp.GreenFilterScope.StatWindow)

    def test_names_and_values(self):
        base = presets.landsat_c1_dn()
        for name, value in ccdcpp.GREEN_FILTER_SCOPES.items():
            with self.subTest(name=name):
                p = dataclasses.replace(base, green_filter_scope=name)
                self.assertEqual(p.options()[0].GREEN_FILTER_SCOPE, value)
                q = dataclasses.replace(base, green_filter_scope=value)
                self.assertEqual(q.green_filter_scope, name)
                self.assertEqual(q, p)
                self.assertEqual(hash(q), hash(p))
                self.assertEqual(pickle.loads(pickle.dumps(p)), p)

    def test_unknown_name(self):
        with self.assertRaises(ValueError) as caught:
            dataclasses.replace(presets.landsat_c1_dn(), green_filter_scope="StatWindow")
        self.assertIn("green_filter_scope is 'StatWindow'", str(caught.exception))


class DateOrder(unittest.TestCase):
    def test_profile_entry_points_raise(self):
        dates = DATES.copy()
        dates[3], dates[4] = dates[4], dates[3]
        profile = presets.landsat_c1_dn()
        with self.assertRaises(ValueError) as caught:
            ccdcpp.detect(dates, CUBE_DN[0, 0], CODES[0, 0], profile=profile)
        self.assertIn("dates must be non-decreasing", str(caught.exception))
        for entry in (ccdcpp.detect_cube, ccdcpp.detect_cube_columns):
            with self.subTest(entry=entry.__name__):
                with self.assertRaises(ValueError):
                    entry(dates, CUBE_DN, CODES, threads=1, profile=profile)


class Overrides(unittest.TestCase):
    def test_hoptions_and_loptions_win(self):
        h, lo = ccdcpp.HarmonicOptions(), ccdcpp.LassoOptions()
        got = every_entry(DATES, CUBE_DN, CODES, hoptions=h, loptions=lo,
                          profile=presets.landsat_c1_physical())
        assert_same(self, strip_band_names(got),
                    every_entry(DATES, CUBE_DN, CODES, hoptions=h, loptions=lo))

    def test_each_one_wins_alone(self):
        h_phys, lo_phys = explicit_physical()
        h_dn, lo_dn = ccdcpp.HarmonicOptions(), ccdcpp.LassoOptions()
        s, q = CUBE_PHYS[0, 0], CODES[0, 0]
        profile = presets.landsat_c1_dn()
        assert_same(self, without(ccdcpp.detect(DATES, s, q, hoptions=h_phys, profile=profile),
                                  "band_names"),
                    EXT.detect(DATES, s, q, h_phys, lo_dn))
        assert_same(self, without(ccdcpp.detect(DATES, s, q, loptions=lo_phys, profile=profile),
                                  "band_names"),
                    EXT.detect(DATES, s, q, h_dn, lo_phys))

    def test_qa_table_wins(self):
        raw = np.vectorize(C2_RAW.get, otypes=[np.uint16])(CODES)
        profile = presets.landsat_c1_physical()
        table = ccdcpp.qa_tables.landsat_c2_qa_pixel()
        assert_same(self,
                    every_entry(DATES, CUBE_PHYS, raw, profile=profile, qa_table=table),
                    every_entry(DATES, CUBE_PHYS, CODES, profile=profile))
        # None turns the profile's table off.
        with self.assertRaises(TypeError):
            ccdcpp.detect_cube(DATES, CUBE_PHYS, raw, profile=presets.landsat_c2(), qa_table=None)

    def test_table_follows_the_run_options(self):
        h, lo = presets.landsat_c2().options()
        h.QA_FILL, h.QA_CLEAR, h.QA_WATER = 200, 10, 11
        h.QA_SHADOW, h.QA_SNOW, h.QA_CLOUD = 12, 13, 14
        profile = presets.landsat_c2()
        np.testing.assert_array_equal(
            profile.qa_table(h).codes, ccdcpp.qa_tables.landsat_c2_qa_pixel(h).codes)
        self.assertIs(profile.qa_table(), presets.landsat_c2().qa_table())
        raw = np.vectorize(C2_RAW.get, otypes=[np.uint16])(CODES)
        renumbered = np.vectorize(
            {FILL: 200, CLEAR: 10, WATER: 11, SHADOW: 12, SNOW: 13, CLOUD: 14}.get,
            otypes=[np.uint8])(CODES)
        assert_same(self,
                    every_entry(DATES, CUBE_PHYS, raw, hoptions=h, profile=profile),
                    every_entry(DATES, CUBE_PHYS, renumbered, hoptions=h, loptions=lo,
                                profile=profile, qa_table=None))


class BandNames(unittest.TestCase):
    def test_only_with_a_profile(self):
        h, lo = ccdcpp.HarmonicOptions(), ccdcpp.LassoOptions()
        names = presets.landsat_c1_dn().band_names
        plain = every_entry(DATES, CUBE_DN, CODES, hoptions=h, loptions=lo)
        profiled = every_entry(DATES, CUBE_DN, CODES, profile=presets.landsat_c1_dn())
        for out in plain["detect"] + [plain["detect_cube_columns"]]:
            self.assertNotIn("band_names", out)
        for out in profiled["detect"] + [profiled["detect_cube_columns"]]:
            self.assertEqual(out["band_names"], names)
            self.assertIsInstance(out["band_names"], tuple)
        for p in plain["detect_cube"] + profiled["detect_cube"]:
            self.assertNotIn("band_names", p)


class Unprofiled(unittest.TestCase):
    """Without profile= the package entry points are the extension's."""

    def test_same_output_every_call_form(self):
        h, lo = ccdcpp.HarmonicOptions(), ccdcpp.LassoOptions()
        d, s, q = DATES, CUBE_DN, CODES
        expected = without(EXT.detect_cube_columns(d, s, q, h, lo, 1), "threads")
        forms = {
            # The millworks nlcd plugin's call.
            "keywords": lambda: ccdcpp.detect_cube_columns(
                dates=d, spectral=s, qas=q, hoptions=h, loptions=lo, threads=1),
            "positional": lambda: ccdcpp.detect_cube_columns(d, s, q, h, lo, 1),
            "qa_table=None": lambda: ccdcpp.detect_cube_columns(d, s, q, h, lo, 1, qa_table=None),
        }
        for name, call in forms.items():
            with self.subTest(form=name):
                assert_same(self, without(call(), "threads"), expected)
        assert_same(self, ccdcpp.detect_cube(d, s, q, h, lo, 2),
                    EXT.detect_cube(d, s, q, h, lo, 2))
        assert_same(self, ccdcpp.detect(d, s[0, 0], q[0, 0], h, lo),
                    EXT.detect(d, s[0, 0], q[0, 0], h, lo))
        raw = np.vectorize(C2_RAW.get, otypes=[np.uint16])(q)
        table = ccdcpp.qa_tables.landsat_c2_qa_pixel()
        assert_same(self, ccdcpp.detect(d, s[0, 0], raw[0, 0], h, lo, qa_table=table),
                    EXT.detect(d, s[0, 0], raw[0, 0], h, lo, qa_table=table))

    def test_signatures_keep_names_and_positions(self):
        for name, positional in (
            ("detect", ["dates", "spectral", "qas", "hoptions", "loptions"]),
            ("detect_cube", ["dates", "spectral", "qas", "hoptions", "loptions", "threads"]),
            ("detect_cube_columns",
             ["dates", "spectral", "qas", "hoptions", "loptions", "threads"]),
        ):
            with self.subTest(entry=name):
                params = inspect.signature(getattr(ccdcpp, name)).parameters
                self.assertEqual(list(params),
                                 positional + ["qa_table", "preprocess", "profile"])
                kinds = [p.kind for p in params.values()]
                self.assertEqual(kinds, [inspect.Parameter.POSITIONAL_OR_KEYWORD] * len(positional)
                                 + [inspect.Parameter.KEYWORD_ONLY] * 3)
                if "threads" in params:
                    self.assertEqual(params["threads"].default, 0)

    def test_options_required_without_a_profile(self):
        h = ccdcpp.HarmonicOptions()
        with self.assertRaises(TypeError):
            ccdcpp.detect(DATES, CUBE_DN[0, 0], CODES[0, 0])
        with self.assertRaises(TypeError):
            ccdcpp.detect_cube_columns(DATES, CUBE_DN, CODES, h)
        with self.assertRaises(TypeError):
            ccdcpp.detect_cube(DATES, CUBE_DN, CODES, hoptions=h, threads=1)

    def test_extension_errors_pass_through(self):
        h, lo = ccdcpp.HarmonicOptions(), ccdcpp.LassoOptions()
        with self.assertRaises(ValueError) as caught:
            ccdcpp.detect(DATES, CUBE_DN[0, 0, :6], CODES[0, 0], h, lo)
        self.assertIn("expects 7", str(caught.exception))

    def test_help_and_pickle(self):
        for name in ("detect", "detect_cube", "detect_cube_columns"):
            with self.subTest(entry=name):
                entry = getattr(ccdcpp, name)
                self.assertEqual((entry.__module__, entry.__name__), ("ccdcpp", name))
                text = pydoc.render_doc(entry)
                for word in ("hoptions", "loptions", "qa_table", "preprocess", "profile",
                             "band_names"):
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
                self.assertEqual(fields(q.options()[0]), fields(p.options()[0]))
                with self.assertRaises(dataclasses.FrozenInstanceError):
                    p.alpha = 2.0

    def test_options_are_new_objects(self):
        p = presets.landsat_c1_dn()
        (h1, l1), (h2, l2) = p.options(), p.options()
        self.assertIsNot(h1, h2)
        h1.PEEK_SIZE = 99
        self.assertEqual(p.options()[0].PEEK_SIZE, 6)

    def test_cadence_and_stat_ord(self):
        p = dataclasses.replace(presets.landsat_c1_dn(), stat_ord=None,
                                cadence={"PEEK_REFERENCE_DAYS": 5.0, "MEOW_SIZE": 18})
        h, _ = p.options()
        self.assertEqual(p.cadence, (("MEOW_SIZE", 18), ("PEEK_REFERENCE_DAYS", 5.0)))
        self.assertEqual((h.PEEK_REFERENCE_DAYS, h.MEOW_SIZE), (5.0, 18))
        self.assertEqual(h.STAT_ORD, ccdcpp.NO_STAT_CUTOFF)
        self.assertEqual(h.STAT_ORD, 2**63 - 1)

    def test_change_probability_moves_the_threshold(self):
        p = dataclasses.replace(presets.landsat_c1_dn(), cadence={"CHANGE_PROBABILITY": 0.995})
        h, _ = p.options()
        self.assertEqual(h.CHANGE_THRESHOLD, ccdcpp.chi2_quantile(0.995, 5))
        self.assertEqual(h.OUTLIER_THRESHOLD, ccdcpp.HarmonicOptions().OUTLIER_THRESHOLD)


class DetectionCount(unittest.TestCase):
    def test_thresholds_follow_the_count(self):
        h, _ = ten_band_profile().options()
        self.assertEqual(len(h.DETECTION_BANDS), 6)
        self.assertEqual(h.CHANGE_THRESHOLD, ccdcpp.chi2_quantile(0.99, 6))
        self.assertEqual(h.OUTLIER_THRESHOLD,
                         ccdcpp.chi2_quantile(ccdcpp.OUTLIER_PROBABILITY, 6))
        defaults = ccdcpp.HarmonicOptions()
        self.assertEqual(ccdcpp.chi2_quantile(ccdcpp.OUTLIER_PROBABILITY, 5).hex(),
                         defaults.OUTLIER_THRESHOLD.hex())

    def test_warns_on_chi2_5_defaults(self):
        defaults = ccdcpp.HarmonicOptions()
        for name, field in (("CHANGE_THRESHOLD", "change_threshold"),
                            ("OUTLIER_THRESHOLD", "outlier_threshold")):
            with self.subTest(threshold=name):
                p = dataclasses.replace(ten_band_profile(), **{field: getattr(defaults, name)})
                with self.assertWarns(UserWarning) as caught:
                    h, _ = p.options()
                self.assertIn(f"6 detection bands but {name} is the chi2(5) default",
                              str(caught.warning))
                self.assertEqual(getattr(h, name), getattr(defaults, name))

    def test_no_warning_otherwise(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ten_band_profile().options()
            dataclasses.replace(ten_band_profile(), change_threshold=20.0).options()
            for preset in (presets.landsat_c1_dn, presets.landsat_c1_physical, presets.landsat_c2):
                preset().options()
                preset().qa_table()


#------------------------------------------------------------------------------
# A ten-band, thermal-less, non-Landsat profile, defined here and not shipped:
# a Sentinel-2-like layout in reflectance with SCL QA.
#------------------------------------------------------------------------------

def ten_band_profile():
    return ccdcpp.SensorProfile(
        name="test_ten_band",
        band_names=("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"),
        roles=("blue", "green", "red", "red_edge_1", "red_edge_2", "red_edge_3", "nir",
               "nir_narrow", "swir1", "swir2"),
        detection=("green", "red", "red_edge_2", "nir", "swir1", "swir2"),
        tmask=("green", "swir1"),
        saturation=("blue", "green", "red", "red_edge_1", "red_edge_2", "red_edge_3", "nir",
                    "nir_narrow", "swir1", "swir2"),
        units=ccdcpp.REFLECTANCE,
        alpha=1e-4,
        coef_floor=1e-4,
        cadence={"PEEK_REFERENCE_DAYS": 10.0},
        stat_ord=None,
        qa_table_builder="sentinel2_scl",
    )


#: Raw SCL values: 4 vegetation, 5 bare (clear); 6 water; 8, 9 cloud; 3 shadow;
#: 0 no data.
SCL_VALUES = np.array([4, 5, 6, 8, 9, 3, 0], dtype=np.uint8)


def ten_band_inputs(H=2, W=2, seed=5):
    """(dates, spectral (H, W, 10, T) reflectance, raw SCL (H, W, T)): a 5-day
    series from 2018, after the Landsat STAT_ORD, with a break per pixel."""
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
        out = ccdcpp.detect_cube_columns(dates, cube, scl, threads=2, profile=profile)
        self.assertEqual(out["failures"], [])
        self.assertEqual(out["num_bands"], 10)
        self.assertEqual(out["band_names"], profile.band_names)
        n = len(out["row"])
        self.assertGreater(n, 4)  # every pixel breaks
        self.assertEqual(set(out["curve_qa"].tolist()), {8})  # Standard, every segment
        self.assertEqual(out["coefficients"].shape, (n, 10, 7))
        self.assertTrue(np.isfinite(out["rmse"]).all())

        one = ccdcpp.detect(dates, cube[1, 0], scl[1, 0], profile=profile)
        self.assertEqual(one["band_names"], profile.band_names)
        self.assertTrue(all(len(m["bands"]) == 10 for m in one["models"]))

    def test_is_the_hand_built_run(self):
        profile = ten_band_profile()
        dates, cube, scl = ten_band_inputs()
        h = ccdcpp.HarmonicOptions()
        h.NUM_BANDS = 10
        h.DETECTION_BANDS = [1, 2, 4, 6, 8, 9]
        h.TMASK_BANDS = [1, 8]
        h.SATURATION_BANDS = list(range(10))
        h.GREEN_IDX = 1
        h.THERMAL_IDX = ccdcpp.NO_BAND
        h.REFL_MIN, h.REFL_MAX, h.MEDIAN_GREEN_FILTER = 0.0, 1.0, 0.04
        h.PEEK_REFERENCE_DAYS = 10.0
        h.STAT_ORD = 2**63 - 1
        h.CHANGE_THRESHOLD = ccdcpp.chi2_quantile(0.99, 6)
        h.OUTLIER_THRESHOLD = ccdcpp.chi2_quantile(1 - 1e-6, 6)
        lo = ccdcpp.LassoOptions()
        lo.alpha = lo.coef_floor = 1e-4
        table = ccdcpp.qa_tables.sentinel2_scl(h)
        assert_same(self,
                    strip_band_names(every_entry(dates, cube, scl, profile=profile)),
                    every_entry(dates, cube, scl, hoptions=h, loptions=lo, qa_table=table))

    def test_documented_example_runs(self):
        # The profile in ccdcpp.profiles' docstring, which is not shipped.
        doc = ccdcpp.profiles.__doc__
        self.assertIn("NOT SHIPPED", doc)
        code = doc.split("::\n", 1)[1].split("\n\nThe detection count", 1)[0]
        namespace = {}
        exec(compile(inspect.cleandoc("\n" + code), "<profiles docstring>", "exec"), namespace)
        profile = namespace["hls_l30"]()
        self.assertFalse(hasattr(presets, "hls_l30"))
        h, lo = profile.options()
        self.assertEqual((h.NUM_BANDS, h.GREEN_IDX, h.THERMAL_IDX), (8, 2, 7))
        self.assertEqual(h.DETECTION_BANDS, [2, 3, 4, 5, 6])
        self.assertIsInstance(profile.qa_table(), ccdcpp.QaTable)

    def test_stat_ord_matters_after_2017(self):
        dates, cube, scl = ten_band_inputs()
        landsat = dataclasses.replace(ten_band_profile(), stat_ord=736694)
        out = ccdcpp.detect_cube_columns(dates, cube, scl, threads=2, profile=landsat)
        self.assertNotEqual(set(out["curve_qa"].tolist()), {8})


if __name__ == "__main__":
    unittest.main()
