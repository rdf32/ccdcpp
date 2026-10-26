"""The package as the wheel lays it out: what it exposes of the extension, its types, its
version, its build information (and detect_cube's warning on a build without OpenMP) and the
qa_tables submodule.

usage: python test_package.py <module_dir> <source_dir> [unittest args]

<module_dir> holds the built extension; <source_dir> is the repository root.
The package is assembled as the wheel lays it out (_harness.use_package()).
ctest runs this as `python_package`.
"""
from __future__ import annotations

import os
import tomllib
import unittest
import warnings

from _harness import SOURCE_DIR, use_package

use_package()

import ccdcpp  # noqa: E402
import numpy as np  # noqa: E402


class Package(unittest.TestCase):
    def test_it_holds_every_extension_name(self):
        # The package wraps detect, detect_cube and plan_of; every other public name of the
        # extension is the extension's own.
        wrapped = {"detect", "detect_cube", "plan_of"}
        for name in dir(ccdcpp.ccdcpp):
            if name.startswith("_") and name != "__version__":
                continue
            with self.subTest(name=name):
                if name in wrapped:
                    self.assertIsNot(getattr(ccdcpp, name), getattr(ccdcpp.ccdcpp, name))
                else:
                    self.assertIs(getattr(ccdcpp, name), getattr(ccdcpp.ccdcpp, name))

    def test_it_ships_its_types(self):
        for name in ("py.typed", "ccdcpp.pyi", "qa_tables.pyi"):
            self.assertTrue(os.path.isfile(os.path.join(os.path.dirname(ccdcpp.__file__), name)),
                            name)

    def test_its_version_is_pyprojects(self):
        with open(os.path.join(SOURCE_DIR, "pyproject.toml"), "rb") as f:
            self.assertEqual(ccdcpp.__version__, tomllib.load(f)["project"]["version"])

    def test_build_info_says_what_this_build_is(self):
        info = ccdcpp.build_info()
        types = {"version": str, "commit": str, "openmp": bool, "openmp_version": int,
                 "compiler": str, "cxx_standard": int, "observe": bool, "build_type": str}
        self.assertEqual(sorted(info), sorted(types))
        for key, kind in types.items():
            with self.subTest(key=key):
                self.assertIs(type(info[key]), kind)
        self.assertEqual(info["version"], ccdcpp.__version__)
        self.assertGreaterEqual(info["cxx_standard"], 202002)
        self.assertEqual(info["openmp_version"] > 0, info["openmp"])

    def test_detect_cube_warns_when_the_build_has_no_openmp(self):
        # Without OpenMP, threads=0 (the default) and threads > 1 warn, once per call, that
        # the cube runs on one thread; threads=1 does not. With OpenMP nothing warns.
        T = 30
        dates = 724276 + 16 * np.arange(T, dtype=np.int64)
        spectral = np.full((1, 2, 7, T), 1000.0)
        qas = np.zeros((1, 2, T), dtype=np.uint8)
        openmp = ccdcpp.build_info()["openmp"]
        for threads in (0, 1, 2):
            with self.subTest(threads=threads), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                out = ccdcpp.detect_cube(dates, spectral, qas, spec=ccdcpp.DataSpec(),
                                         threads=threads)
                no_openmp = [w for w in caught if issubclass(w.category, RuntimeWarning)
                             and "no OpenMP" in str(w.message)]
                self.assertEqual(len(no_openmp), 0 if openmp or threads == 1 else 1)
                if no_openmp:   # the caller's line
                    self.assertEqual(no_openmp[0].filename, __file__)
                if not openmp:
                    self.assertEqual(int(out["threads"]), 1)

    def test_qa_tables_imports_as_its_submodule(self):
        import ccdcpp.qa_tables as q
        self.assertIs(q, ccdcpp.qa_tables)
        self.assertIs(q, ccdcpp.ccdcpp.qa_tables)
        self.assertIn("Landsat", q.__doc__ or "")
        self.assertIsInstance(q.hls_fmask(), ccdcpp.QaTable)


if __name__ == "__main__":
    unittest.main()
