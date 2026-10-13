"""Python entry points: input validation, the band layout and __version__.

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

    def test_empty_saturation_list_is_allowed(self):
        h = self.broken(SATURATION_BANDS=[])
        ccdcpp.detect(*inputs(10, 40), h, ccdcpp.LassoOptions())


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


class Version(unittest.TestCase):
    def pyproject_version(self):
        with open(os.path.join(SOURCE_DIR, "pyproject.toml"), "rb") as f:
            return tomllib.load(f)["project"]["version"]

    def test_extension_matches_pyproject(self):
        self.assertEqual(ccdcpp.__version__, self.pyproject_version())

    def test_package_exposes_it(self):
        # The wheel layout: python/ccdcpp/__init__.py next to the extension.
        # Run in a fresh interpreter so the extension imported above is not
        # the one found.
        (extension,) = glob.glob(os.path.join(MODULE_DIR, "ccdcpp.*"))
        with tempfile.TemporaryDirectory() as root:
            package = os.path.join(root, "ccdcpp")
            os.mkdir(package)
            shutil.copy(
                os.path.join(SOURCE_DIR, "python", "ccdcpp", "__init__.py"), package)
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
