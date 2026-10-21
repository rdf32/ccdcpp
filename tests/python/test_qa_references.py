"""The Landsat QA tables against the implementations they reproduce.

usage: python test_qa_references.py <module_dir> <source_dir> [unittest args]

  * landsat_c2_qa_pixel against the table the millworks nlcd plugin builds
    (nlcd.changedetection.io._QA_TABLE, from nlcd.science.qa_code), all
    65,536 entries;
  * landsat_c1_pixel_qa against pyccd's ccd.qa.qabitval under pyccd's
    bit-packed parameters, all 65,536 entries.

Both references are found on sys.path. Point PYTHONPATH at them for this one
command, never in the shell's environment, e.g.

    PYTHONPATH=<millworks>/plugins/nlcd/src:<pyccd> python test_qa_references.py ...

ctest runs this as `python_qa_references`, with PYTHONPATH set from the
CCD_REFERENCE_PYTHONPATH cache variable for that test alone. A reference that
is not importable skips its test and says why; when both are skipped the
script exits 77, which ctest reports as skipped.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
import types
import unittest
from typing import TYPE_CHECKING

from _harness import use_extension

use_extension()

import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

if TYPE_CHECKING:
    # The bare extension module, which ccdcpp/ccdcpp.pyi types.
    from ccdcpp import ccdcpp
else:
    import ccdcpp


def nlcd_table() -> tuple[NDArray[np.uint8] | None, types.ModuleType | None, str | None]:
    """The nlcd plugin's QA table and its science module, or the reason they are
    unavailable."""
    try:
        from nlcd.changedetection import io  # pyright: ignore[reportMissingImports]  # optional reference
        from nlcd import science  # pyright: ignore[reportMissingImports]  # optional reference
    except ImportError as e:
        return None, None, (
            f"millworks nlcd plugin not importable ({e}); run with "
            "PYTHONPATH=<millworks>/plugins/nlcd/src for this command")
    return io._QA_TABLE(np), science, None


def pyccd_qa() -> tuple[types.ModuleType | None, str | None]:
    """pyccd's ccd.qa module, or the reason it is unavailable.

    ccd/__init__.py imports the whole of pyccd, scikit-learn included, and
    qabitval needs none of it, so when the package does not import, ccd.qa is
    loaded from the package directory without running __init__.py."""
    spec = importlib.util.find_spec("ccd")
    if spec is None or not spec.submodule_search_locations:
        return None, (
            "pyccd (package `ccd`) not importable; run with "
            "PYTHONPATH=<pyccd checkout or unpacked lcmap-pyccd wheel> for this command")
    try:
        return importlib.import_module("ccd.qa"), None
    except ImportError:
        pass
    for name in [m for m in sys.modules if m == "ccd" or m.startswith("ccd.")]:
        del sys.modules[name]
    package = types.ModuleType("ccd")
    package.__path__ = list(spec.submodule_search_locations)
    sys.modules["ccd"] = package
    try:
        return importlib.import_module("ccd.qa"), None
    except ImportError as e:
        return None, f"pyccd's ccd.qa does not import ({e})"


class NlcdC2(unittest.TestCase):
    def test_all_entries_equal_the_nlcd_table(self):
        table, science, reason = nlcd_table()
        if table is None or science is None:
            self.skipTest(reason)

        # The plugin's codes are the Input's defaults.
        h = ccdcpp.Input()
        self.assertEqual(
            [science.QA_FILL, science.QA_CLEAR, science.QA_WATER,
             science.QA_SHADOW, science.QA_SNOW, science.QA_CLOUD],
            [h.qa_fill, h.qa_clear, h.qa_water, h.qa_shadow, h.qa_snow, h.qa_cloud])

        ours = ccdcpp.qa_tables.landsat_c2_qa_pixel().codes
        self.assertEqual(table.dtype, np.uint8)
        self.assertEqual(table.shape, (65536,))
        differ = np.flatnonzero(ours != table)
        self.assertEqual(
            differ.size, 0,
            f"{differ.size} of 65536 entries differ; first {differ[:10].tolist()}")
        print(f"\nlandsat_c2_qa_pixel == nlcd _QA_TABLE on all 65536 entries "
              f"({science.__file__})", file=sys.stderr)


class PyccdC1(unittest.TestCase):
    def test_all_entries_equal_qabitval(self):
        qa, reason = pyccd_qa()
        if qa is None:
            self.skipTest(reason)

        # pyccd's bit-packed parameters: each QA_* is a bit offset, and
        # qabitval returns the offset of the class it chose.
        params = types.SimpleNamespace(
            QA_FILL=0, QA_CLEAR=1, QA_WATER=2, QA_SHADOW=3, QA_SNOW=4,
            QA_CLOUD=5, QA_CIRRUS1=8, QA_CIRRUS2=9, QA_OCCLUSION=10)
        h = ccdcpp.Input()
        code_of_offset = {
            0: h.qa_fill, 1: h.qa_clear, 2: h.qa_water,
            3: h.qa_shadow, 4: h.qa_snow, 5: h.qa_cloud}

        ours = ccdcpp.qa_tables.landsat_c1_pixel_qa().codes
        rejected = 0
        for v in range(65536):
            try:
                expected = code_of_offset[qa.qabitval(v, params)]
            except ValueError:
                # pyccd raises; the table decodes to fill.
                expected = h.qa_fill
                rejected += 1
            self.assertEqual(ours[v], expected, v)
        print(f"\nlandsat_c1_pixel_qa == pyccd qabitval on all 65536 entries, "
              f"{rejected} of them values pyccd rejects ({qa.__file__})", file=sys.stderr)


if __name__ == "__main__":
    result = unittest.main(exit=False, verbosity=2).result
    if not result.wasSuccessful():
        sys.exit(1)
    if result.testsRun and len(result.skipped) == result.testsRun:
        sys.exit(77)
