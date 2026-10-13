"""Python entry points: input validation and __version__.

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
