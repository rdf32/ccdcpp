"""The package as the wheel lays it out: what it exposes of the extension, its types, its
version and the qa_tables submodule.

usage: python test_package.py <module_dir> <source_dir> [unittest args]

<module_dir> holds the built extension; <source_dir> is the repository root.
The package is assembled as the wheel lays it out (_harness.use_package()).
ctest runs this as `python_package`.
"""
from __future__ import annotations

import os
import tomllib
import unittest

from _harness import SOURCE_DIR, use_package

use_package()

import ccdcpp  # noqa: E402


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

    def test_qa_tables_imports_as_its_submodule(self):
        import ccdcpp.qa_tables as q
        self.assertIs(q, ccdcpp.qa_tables)
        self.assertIs(q, ccdcpp.ccdcpp.qa_tables)
        self.assertIn("Landsat", q.__doc__ or "")
        self.assertIsInstance(q.hls_fmask(), ccdcpp.QaTable)


if __name__ == "__main__":
    unittest.main()
