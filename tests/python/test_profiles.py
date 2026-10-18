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
from unittest import mock
from datetime import date

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
            (dict(roles=base.roles[:6] + ("nir_narrow",)), "unknown roles ['nir_narrow']"),
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


class Roles(unittest.TestCase):
    def test_roles(self):
        self.assertEqual(ccdcpp.ROLES, (
            "blue", "green", "red", "nir", "swir1", "swir2", "thermal", "coastal",
            "red_edge_1", "red_edge_2", "red_edge_3", "nir_broad", "other"))

    def test_nir_broad_is_a_band_role(self):
        # B08 as nir_broad next to B8A as nir: it is fitted, reported and may
        # be listed, and it is not NIR_IDX.
        p = ten_band_profile()
        h, _ = p.options()
        self.assertEqual(p.index("nir_broad"), 6)
        self.assertEqual(h.NIR_IDX, 7)
        self.assertIn(6, h.SATURATION_BANDS)
        q = dataclasses.replace(p, detection=("green", "red", "nir_broad", "nir", "swir1",
                                              "swir2"))
        self.assertEqual(q.options()[0].DETECTION_BANDS, [1, 2, 6, 7, 8, 9])
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
        np.testing.assert_array_equal(self.hls().qa_table().codes,
                                      ccdcpp.qa_tables.hls_fmask().codes)

    def test_builder_arguments_reach_the_builder(self):
        for aerosol in (False, True):
            with self.subTest(exclude_high_aerosol=aerosol):
                p = self.hls(qa_table_options={"exclude_high_aerosol": aerosol})
                h, _ = p.options()
                expected = ccdcpp.qa_tables.hls_fmask(h, exclude_high_aerosol=aerosol).codes
                got = p.qa_table().codes
                self.assertEqual(got.shape, ALL_VALUES.shape)
                np.testing.assert_array_equal(got, expected)
        # The two tables differ exactly where the aerosol bits are 11 and no
        # higher-precedence bit (fill, cloud, adjacent, cirrus) is set.
        on = self.hls(qa_table_options={"exclude_high_aerosol": True}).qa_table().codes
        off = self.hls().qa_table().codes
        differs = np.flatnonzero(on != off)
        self.assertTrue(len(differs) > 0)
        self.assertTrue(((differs >> 6) == 3).all())
        self.assertEqual(int(on[192]), CLOUD)
        self.assertEqual(int(off[192]), CLEAR)

    def test_follows_the_run_options(self):
        p = self.hls(qa_table_options={"exclude_high_aerosol": True})
        h, _ = p.options()
        h.QA_FILL, h.QA_CLEAR, h.QA_WATER = 200, 10, 11
        h.QA_SHADOW, h.QA_SNOW, h.QA_CLOUD = 12, 13, 14
        np.testing.assert_array_equal(p.qa_table(h).codes,
                                      ccdcpp.qa_tables.hls_fmask(h, True).codes)

    def test_cached_by_value(self):
        a = self.hls(qa_table_options={"exclude_high_aerosol": True})
        b = self.hls(qa_table_options=(("exclude_high_aerosol", True),))
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertIs(a.qa_table(), b.qa_table())
        self.assertIsNot(a.qa_table(), self.hls().qa_table())

    def test_stored_sorted_hashable_picklable(self):
        p = self.hls(qa_table_options={"exclude_high_aerosol": True})
        self.assertEqual(p.qa_table_options, (("exclude_high_aerosol", True),))
        q = pickle.loads(pickle.dumps(p))
        self.assertEqual(q, p)
        np.testing.assert_array_equal(q.qa_table().codes, p.qa_table().codes)

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
        table = ccdcpp.qa_tables.hls_fmask(screened.options()[0], True)
        got = every_entry(DATES, CUBE_PHYS, fmask, profile=screened)
        assert_same(self, got, every_entry(DATES, CUBE_PHYS, fmask, profile=base, qa_table=table))
        codes = np.where(aerosol, np.uint8(CLOUD), CODES)
        assert_same(self, got, every_entry(DATES, CUBE_PHYS, codes, profile=base, qa_table=None))
        self.assertNotEqual(
            ccdcpp.detect(DATES, CUBE_PHYS[0, 1], fmask[0, 1], profile=base)["processing_mask"]
            .tobytes(),
            ccdcpp.detect(DATES, CUBE_PHYS[0, 1], fmask[0, 1], profile=screened)["processing_mask"]
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
                self.assertIn(f"{builder}(options: ", signature)
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
                self.assertIs(p.qa_table(), profiles[0].qa_table())
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
        pixel = ["dates", "spectral", "qas", "hoptions", "loptions"]
        cube = pixel + ["threads"]
        common = ["qa_table", "preprocess", "profile", "band_names"]
        for name, positional, keywords in (
            ("detect", pixel, common + ["resume_record", "method"]),
            ("detect_cube", cube, common + ["method"]),
            ("detect_cube_columns", cube, common + ["resume_record", "method"]),
            ("resume", pixel, ["resume", "resume_header", "models"] + common + ["method"]),
            ("resume_cube_columns", cube,
             ["resume", "resume_header", "segments"] + common + ["method"]),
        ):
            with self.subTest(entry=name):
                params = inspect.signature(getattr(ccdcpp, name)).parameters
                self.assertEqual(list(params), positional + keywords)
                kinds = [p.kind for p in params.values()]
                self.assertEqual(kinds, [inspect.Parameter.POSITIONAL_OR_KEYWORD] * len(positional)
                                 + [inspect.Parameter.KEYWORD_ONLY] * len(keywords))
                self.assertIsNone(params["band_names"].default)
                if "threads" in params:
                    self.assertEqual(params["threads"].default, 0)
                if "resume_record" in params:
                    self.assertIs(params["resume_record"].default, False)
                self.assertIsNone(params["method"].default)

    def test_method(self):
        # method= picks the Standard procedure's change detection over the
        # call's options, with or without a profile; "ccdc" is the default
        # run, an unknown name is refused, and the caller's options are not
        # changed.
        h, lo = ccdcpp.HarmonicOptions(), ccdcpp.LassoOptions()
        d, s, q = DATES, CUBE_DN, CODES
        self.assertEqual(ccdcpp.METHODS, ("ccdc",))
        self.assertEqual(h.METHOD, "ccdc")
        expected = without(ccdcpp.detect_cube_columns(d, s, q, h, lo, 1), "threads")
        assert_same(self, without(ccdcpp.detect_cube_columns(d, s, q, h, lo, 1, method="ccdc"),
                                  "threads"), expected)
        profile = ccdcpp.presets.landsat_c1_dn()
        assert_same(self, without(ccdcpp.detect_cube_columns(d, s, q, threads=1, profile=profile,
                                                             method="ccdc"), "threads"),
                    without(ccdcpp.detect_cube_columns(d, s, q, threads=1, profile=profile), "threads"))
        self.assertEqual(ccdcpp.with_method(h, "ccdc").METHOD, "ccdc")
        with self.assertRaisesRegex(ValueError, "unknown change detection method"):
            ccdcpp.detect(d, s[0, 0], q[0, 0], h, lo, method="CCDC")
        with self.assertRaisesRegex(ValueError, "unknown change detection method"):
            h.METHOD = "other"
        self.assertEqual(h.METHOD, "ccdc")

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
        roles=("blue", "green", "red", "red_edge_1", "red_edge_2", "red_edge_3", "nir_broad",
               "nir", "swir1", "swir2"),
        detection=("green", "red", "red_edge_2", "nir", "swir1", "swir2"),
        tmask=("green", "swir1"),
        saturation=("blue", "green", "red", "red_edge_1", "red_edge_2", "red_edge_3",
                    "nir_broad", "nir", "swir1", "swir2"),
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
        h.DETECTION_BANDS = [1, 2, 4, 7, 8, 9]
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
        profile = namespace["landsat_oli_c2"]()
        self.assertFalse(hasattr(presets, "landsat_oli_c2"))
        h, lo = profile.options()
        self.assertEqual((h.NUM_BANDS, h.GREEN_IDX, h.THERMAL_IDX), (8, 2, 7))
        self.assertEqual(h.DETECTION_BANDS, [2, 3, 4, 5, 6])
        self.assertEqual(h.STAT_ORD, ccdcpp.HarmonicOptions().STAT_ORD)
        np.testing.assert_array_equal(profile.qa_table().codes,
                                      ccdcpp.qa_tables.landsat_c2_qa_pixel(h).codes)

    def test_stat_ord_matters_after_2017(self):
        dates, cube, scl = ten_band_inputs()
        landsat = dataclasses.replace(ten_band_profile(), stat_ord=736694)
        out = ccdcpp.detect_cube_columns(dates, cube, scl, threads=2, profile=landsat)
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


#: Each preset's options, written out: HarmonicOptions() fields that differ.
#: Every unnamed field, the thresholds included, is the default.
PHYSICAL = dict(REFL_MIN=0.0, REFL_MAX=1.0, MEDIAN_GREEN_FILTER=0.04, STAT_ORD=2**63 - 1)
EXPECTED = {
    "hls_l30": dict(
        PHYSICAL, NUM_BANDS=8, DETECTION_BANDS=[2, 3, 4, 5, 6], TMASK_BANDS=[2, 5],
        SATURATION_BANDS=[0, 1, 2, 3, 4, 5, 6], BLUE_IDX=1, GREEN_IDX=2, RED_IDX=3, NIR_IDX=4,
        SWIR1_IDX=5, SWIR2_IDX=6, THERMAL_IDX=7, THERMAL_SCALE=1.0, THERMAL_OFFSET=0.0,
        THERM_MIN=-0.932, THERM_MAX=0.707),
    "hls_s30": dict(
        PHYSICAL, NUM_BANDS=11, DETECTION_BANDS=[2, 3, 8, 9, 10], TMASK_BANDS=[2, 9],
        SATURATION_BANDS=list(range(11)), BLUE_IDX=1, GREEN_IDX=2, RED_IDX=3, NIR_IDX=8,
        SWIR1_IDX=9, SWIR2_IDX=10, THERMAL_IDX=ccdcpp.NO_BAND),
    "hls": dict(
        PHYSICAL, NUM_BANDS=7, DETECTION_BANDS=[2, 3, 4, 5, 6], TMASK_BANDS=[2, 5],
        SATURATION_BANDS=list(range(7)), BLUE_IDX=1, GREEN_IDX=2, RED_IDX=3, NIR_IDX=4,
        SWIR1_IDX=5, SWIR2_IDX=6, THERMAL_IDX=ccdcpp.NO_BAND),
    "sentinel2_l2a": dict(
        PHYSICAL, NUM_BANDS=10, DETECTION_BANDS=[1, 2, 7, 8, 9], TMASK_BANDS=[1, 8],
        SATURATION_BANDS=list(range(10)), BLUE_IDX=0, GREEN_IDX=1, RED_IDX=2, NIR_IDX=7,
        SWIR1_IDX=8, SWIR2_IDX=9, THERMAL_IDX=ccdcpp.NO_BAND),
}
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
    """(HarmonicOptions, LassoOptions, QaTable) of preset ``name``, by hand."""
    h, lo = ccdcpp.HarmonicOptions(), ccdcpp.LassoOptions()
    for key, value in EXPECTED[name].items():
        setattr(h, key, value)
    lo.alpha = lo.coef_floor = 1e-4
    return h, lo, getattr(ccdcpp.qa_tables, BUILDERS[name])(h)


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
        defaults = ccdcpp.HarmonicOptions()
        for preset in NEW_PRESETS:
            name = preset.__name__
            with self.subTest(preset=name):
                h, lo = preset().options()
                expected, elo, table = hand_built(name)
                self.assertEqual(fields(h), fields(expected))
                self.assertEqual(fields(lo), fields(elo))
                for key in ("CHANGE_THRESHOLD", "OUTLIER_THRESHOLD"):
                    self.assertEqual(getattr(h, key).hex(), getattr(defaults, key).hex())
                self.assertEqual(h.STAT_ORD, ccdcpp.NO_STAT_CUTOFF)
                self.assertEqual(preset().band_names, BAND_NAMES[name])
                self.assertEqual(preset().qa_table_options,
                                 dict(sentinel2_l2a=()).get(
                                     name, (("exclude_high_aerosol", False),)))
                np.testing.assert_array_equal(preset().qa_table().codes, table.codes)
                for key in ccdcpp.CADENCE_FIELDS:
                    self.assertEqual(getattr(h, key), getattr(defaults, key))

    def test_red_edge_and_broad_nir_are_fitted_not_detected(self):
        for preset in (presets.hls_s30, presets.sentinel2_l2a):
            with self.subTest(preset=preset.__name__):
                p = preset()
                h, _ = p.options()
                for role in ("red_edge_1", "red_edge_2", "red_edge_3", "nir_broad"):
                    self.assertNotIn(p.index(role), h.DETECTION_BANDS)
                    self.assertIn(p.index(role), h.SATURATION_BANDS)
                self.assertEqual(p.band_names[p.index("nir")], "B8A")
                self.assertEqual(p.band_names[p.index("nir_broad")], "B08")

    def test_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for preset in NEW_PRESETS:
                preset().options()
                preset().qa_table()

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
                h, lo, table = hand_built(name)
                profiled = every_entry(dates, cube, qa, threads=2, profile=preset())
                assert_same(self, strip_band_names(profiled),
                            every_entry(dates, cube, qa, threads=2, hoptions=h, loptions=lo,
                                        qa_table=table))
                cols = profiled["detect_cube_columns"]
                self.assertEqual(cols["failures"], [])
                self.assertEqual(cols["band_names"], BAND_NAMES[name])
                self.assertEqual(set(cols["curve_qa"].tolist()), {8})  # Standard
                self.assertGreater(len(cols["row"]), 4)  # every pixel breaks

    def test_raw_qa_is_the_decoded_codes(self):
        for preset, values, codes in ((presets.hls_l30, FMASK_VALUES, FMASK_CODES),
                                      (presets.sentinel2_l2a, SCL_ALL, SCL_CODES)):
            with self.subTest(preset=preset.__name__):
                np.testing.assert_array_equal(preset().qa_table().codes[values], codes)
                dates, cube, qa = preset_inputs(preset.__name__)
                lookup = dict(zip(values.tolist(), codes.tolist()))
                decoded = np.vectorize(lookup.get, otypes=[np.uint8])(qa)
                assert_same(self,
                            every_entry(dates, cube, qa, threads=2, profile=preset()),
                            every_entry(dates, cube, decoded, threads=2, profile=preset(),
                                        qa_table=None))

    def test_permutation(self):
        order = [10, 0, 9, 1, 8, 2, 7, 3, 6, 4, 5]
        base = presets.hls_s30()
        dates, cube, fmask = preset_inputs("hls_s30")
        expected = every_entry(dates, cube, fmask, threads=2, profile=base)
        got = every_entry(dates, np.ascontiguousarray(cube[:, :, order]), fmask, threads=2,
                          profile=permuted(base, order))
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
            "detect_cube_columns": lambda: ccdcpp.detect_cube_columns(
                dates, cube, qa, threads=2, **kw),
        }

    def test_profile_order_runs_unchanged(self):
        runs = [(p, preset_inputs(p.__name__)) for p in NEW_PRESETS]
        runs.append((presets.landsat_c1_dn, (DATES, CUBE_DN, CODES)))
        for preset, (dates, cube, qa) in runs:
            with self.subTest(preset=preset.__name__):
                profile = preset()
                expected = every_entry(dates, cube, qa, threads=2, profile=profile)
                for names in (profile.band_names, list(profile.band_names),
                              np.array(profile.band_names)):
                    assert_same(self, every_entry(dates, cube, qa, threads=2, profile=profile,
                                                  band_names=names), expected)

    def test_refuses_a_misordered_sentinel2_stack(self):
        profile = presets.sentinel2_l2a()
        dates, cube, scl = preset_inputs("sentinel2_l2a")
        store = [profile.band_names.index(b) for b in S2_STORE_ORDER]
        stacked = np.ascontiguousarray(cube[:, :, store])
        for name, call in self.entries(dates, stacked, scl, profile=profile,
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
        silent = ccdcpp.detect_cube_columns(dates, stacked, scl, threads=2, profile=profile)
        right = ccdcpp.detect_cube_columns(dates, cube, scl, threads=2, profile=profile)
        self.assertEqual(silent["failures"], [])
        self.assertNotEqual(str(without(silent, "threads")), str(without(right, "threads")))
        # Reordered by the names, it is the right run.
        order = [S2_STORE_ORDER.index(b) for b in profile.band_names]
        assert_same(self, ccdcpp.detect_cube_columns(
            dates, np.ascontiguousarray(stacked[:, :, order]), scl, threads=2,
            profile=profile, band_names=tuple(np.array(S2_STORE_ORDER)[order])), right)

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
                    ccdcpp.detect(dates, cube[0, 0], fmask[0, 0], profile=profile,
                                  band_names=given)
                self.assertIn(text, str(caught.exception))
                self.assertIn("the profile has B01, B02", str(caught.exception))
        for given in ("B01B02B03B04B05B06B07B10", b"B01", ("B01",) * 7 + (10,)):
            with self.subTest(given=given):
                with self.assertRaises(TypeError):
                    ccdcpp.detect(dates, cube[0, 0], fmask[0, 0], profile=profile,
                                  band_names=given)

    def test_needs_a_profile(self):
        h, lo = ccdcpp.HarmonicOptions(), ccdcpp.LassoOptions()
        names = presets.landsat_c1_dn().band_names
        for name, call in self.entries(DATES, CUBE_DN, CODES, hoptions=h, loptions=lo,
                                       band_names=names).items():
            with self.subTest(entry=name):
                with self.assertRaises(TypeError) as caught:
                    call()
                self.assertIn("needs profile=", str(caught.exception))

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
        return ccdcpp.detect(dates, np.ascontiguousarray(cube), fmask, profile=profile)

    def test_scale(self):
        dates, cube, _, raw = hls_inputs(8, thermal=True)
        self.assertEqual(raw.dtype, np.int16)
        np.testing.assert_array_equal(cube, raw * 1e-4)
        # Thermal B10 is degrees Celsius x 100; times 0.0001 it is degrees
        # Celsius / 100, inside CELSIUS_HUNDREDTHS' -0.932..0.707.
        h, _ = presets.hls_l30().options()
        self.assertTrue(((cube[:, :, 7] > h.THERM_MIN) & (cube[:, :, 7] < h.THERM_MAX)).all())

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
        out = ccdcpp.detect_cube_columns(dates, cube, scl, threads=2,
                                         profile=presets.sentinel2_l2a())
        self.assertEqual(out["failures"], [])
        self.assertGreater(len(out["row"]), 4)

    def test_uncorrected_input_differs(self):
        # dn / 10000 without the offset: 0.1 too bright from 2022-01-25.
        dates, cube, scl, dn = s2_inputs(breaks=False)
        profile = presets.sentinel2_l2a()
        corrected = ccdcpp.detect_cube_columns(dates, cube, scl, threads=2, profile=profile)
        raw = ccdcpp.detect_cube_columns(dates, dn / 10000.0, scl, threads=2, profile=profile)
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
                            profile=presets.sentinel2_l2a())
        self.assertFalse(out["processing_mask"][before].any())
        self.assertFalse(out["processing_mask"][after].any())
        self.assertTrue(out["processing_mask"][clear[1::30]].all())

if __name__ == "__main__":
    unittest.main()
