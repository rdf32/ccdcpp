"""QA lookup tables through the Python entry points.

usage: python test_qa_tables.py <module_dir> <source_dir> [unittest args]

ctest runs this as `python_qa_tables`. The comparisons against the millworks
nlcd table and against pyccd itself are in test_qa_references.py.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from _harness import assemble, use_extension

use_extension()

import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

if TYPE_CHECKING:
    # The bare extension module, which ccdcpp/ccdcpp.pyi types.
    from ccdcpp import ccdcpp
else:
    import ccdcpp

START = 724276  # 1984-01-01

FILL, CLEAR, WATER, SHADOW, SNOW, CLOUD = 255, 0, 1, 2, 3, 4

#: Raw values per class, per product: several per class, with flag
#: combinations that exercise the precedence. Every one decodes to its class.
SCHEMES = {
    "landsat_c2_qa_pixel": {
        FILL: [1, 1 | 8, 0xFFFF],
        CLEAR: [21824, 64, 0],
        WATER: [21952, 128],
        SHADOW: [16, 16 | 32, 16 | 128, 23888],
        SNOW: [32, 32 | 128, 30048],
        CLOUD: [8, 2, 4, 8 | 16 | 128, 2 | 32, 22280, 55052],
    },
    "landsat_c1_pixel_qa": {
        FILL: [1, 1 | 32, 0xFFFF],
        CLEAR: [2, 2 | 64, 322, 0x300, 0x400, 0x300 | 64],
        WATER: [4, 4 | 2, 324],
        SHADOW: [8, 8 | 16 | 4, 8 | 2, 328],
        SNOW: [16, 16 | 4 | 2, 336],
        CLOUD: [32, 32 | 8 | 4, 32 | 2, 352, 480, 992],
    },
    "hls_fmask": {
        FILL: [255],
        CLEAR: [0, 0x40, 0x80, 0xC0],
        WATER: [32, 32 | 0x40],
        SHADOW: [8, 8 | 16 | 32, 8 | 0x40],
        SNOW: [16, 16 | 32, 16 | 0x80],
        CLOUD: [2, 4, 1, 2 | 8 | 32, 4 | 16, 1 | 0xC0],
    },
    "sentinel2_scl": {
        FILL: [0, 1],
        CLEAR: [4, 5],
        WATER: [6],
        SHADOW: [2, 3],
        SNOW: [11],
        CLOUD: [7, 8, 9, 10],
    },
}

NARROW = {"hls_fmask", "sentinel2_scl"}


def build(name: str, options: ccdcpp.Input | None = None) -> ccdcpp.QaTable:
    builder = getattr(ccdcpp.qa_tables, name)
    return builder() if options is None else builder(options)


def encode(name: str, codes: NDArray[np.uint8]) -> NDArray[np.uint16]:
    """Raw values of product `name` for an array of class codes, cycling
    through each class's raw values by position. Code 8 is encoded as shadow;
    see DecodedEqualsCodes."""
    flat = np.where(codes == 8, SHADOW, codes).ravel()
    raw = np.empty(flat.shape, dtype=np.uint16)
    covered = np.zeros(flat.shape, dtype=bool)
    for code, values in SCHEMES[name].items():
        where = np.flatnonzero(flat == code)
        raw[where] = np.asarray(values, dtype=np.uint16)[where % len(values)]
        covered[where] = True
    assert covered.all(), f"codes {np.unique(flat[~covered])} have no raw value"
    return raw.reshape(codes.shape)


def synthetic(H: int = 3, W: int = 4,
              seed: int = 7) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.uint8]]:
    """(dates, spectral (H, W, 7, T) DN, codes (H, W, T)).

    Pixel 0 is snow-dominated (PermanentSnow), pixel 1 cloud-dominated
    (InsufficientClear), the rest clear-dominated with a break, so every
    procedure runs."""
    rng = np.random.default_rng(seed)
    steps = np.cumsum(rng.choice([16, 16, 16, 32, 48], size=700))
    dates = (START + steps[steps < 26 * 365]).astype(np.int64)
    T = dates.size
    phase = 2 * np.pi * (dates - START) / 365.25

    spectral = np.empty((H, W, 7, T))
    codes = np.empty((H, W, T), dtype=np.uint8)
    for p in range(H * W):
        r, c = divmod(p, W)
        base = 600 + 300 * np.arange(7, dtype=float)[:, None]
        season = (150 + 20 * p) * np.sin(phase + p)[None, :]
        level = np.where(np.arange(T) > T // 2 + 7 * p, 900.0, 0.0)[None, :]
        s = base + season + level + rng.normal(0, 40, (7, T))
        s[6] = 2900 + 80 * np.sin(phase) + rng.normal(0, 10, T)
        spectral[r, c] = np.round(s)

        if p == 0:
            probs = [0.02, 0.05, 0.01, 0.05, 0.85, 0.02, 0.0]
        elif p == 1:
            probs = [0.02, 0.17, 0.03, 0.05, 0.02, 0.66, 0.05]
        else:
            probs = [0.04, 0.70, 0.06, 0.05, 0.03, 0.08, 0.04]
        codes[r, c] = rng.choice(
            np.array([FILL, CLEAR, WATER, SHADOW, SNOW, CLOUD, 8], dtype=np.uint8),
            size=T, p=probs)
    return dates, spectral, codes


def options() -> ccdcpp.Input:
    return ccdcpp.Input()


def with_table(table: ccdcpp.QaTable | None) -> ccdcpp.Input:
    """The default input decoding its raw QA through `table` (None: codes)."""
    return ccdcpp.Input(qa_table=table)


def assert_same(test: unittest.TestCase, a: Any, b: Any, where: str = "out") -> None:
    """Exact, structural equality of entry point outputs."""
    if isinstance(a, dict):
        test.assertEqual(set(a), set(b), where)
        for k in a:
            assert_same(test, a[k], b[k], f"{where}[{k!r}]")
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b), where)
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(test, x, y, f"{where}[{i}]")
    elif isinstance(a, np.ndarray):
        test.assertEqual(a.dtype, b.dtype, where)
        np.testing.assert_array_equal(a, b, err_msg=where)
    else:
        test.assertEqual(a, b, where)


class Tables(unittest.TestCase):
    def test_builders(self):
        for name in SCHEMES:
            with self.subTest(name=name):
                table = build(name)
                self.assertIsInstance(table, ccdcpp.QaTable)
                codes = table.codes
                self.assertEqual(codes.dtype, np.uint8)
                self.assertEqual(codes.shape, (ccdcpp.QA_TABLE_SIZE,))
                self.assertEqual(ccdcpp.QA_TABLE_SIZE, 65536)
                self.assertTrue(set(np.unique(codes)) <= {FILL, CLEAR, WATER, SHADOW, SNOW, CLOUD})
                for code, values in SCHEMES[name].items():
                    np.testing.assert_array_equal(codes[values], code, err_msg=f"{code}")

    def test_submodule_imports(self):
        import ccdcpp.qa_tables as q
        self.assertIs(q, ccdcpp.qa_tables)
        self.assertIn("Landsat", q.__doc__ or "")

    def test_docstrings(self):
        def flat(text: str | None) -> str:
            assert text is not None
            return " ".join(text.split())
        q = ccdcpp.qa_tables
        fmask, scl, module = flat(q.hls_fmask.__doc__), flat(q.sentinel2_scl.__doc__), flat(q.__doc__)
        # HLS v2.0 reserves Fmask bit 0; it is not cirrus.
        self.assertNotIn("cirrus)", fmask)
        self.assertIn("bit 0, reserved and unused in v2.0", fmask)
        self.assertIn("bits 1, 2 (cloud, adjacent to cloud or shadow) cloud", fmask)
        self.assertIn("2, 3 (cast shadows, cloud shadows) shadow", scl)
        self.assertNotIn("dark area", scl.lower())
        for doc, present in ((fmask, "all 43 values present in the T16SGC 2020-2023 sample (38 in L30, 43 in S30)"),
                             (scl, "every class present in the T16SGC 2020-2023 sample (0, 2-11)")):
            self.assertIn(present, doc)
            self.assertIn("no production run has used it yet", doc)
            self.assertNotIn("all 38 values", doc)
            self.assertNotIn("Not yet validated", doc)
        self.assertIn("(MGRS tile T16SGC, 2020-2023); no production run has used them yet",
                      module)
        self.assertNotIn("not yet validated", module)

    def test_codes_is_a_copy(self):
        table = ccdcpp.qa_tables.sentinel2_scl()
        codes = table.codes
        codes[:] = 7
        self.assertEqual(table.codes[4], CLEAR)

    def test_renumbered_options(self):
        h = ccdcpp.Input(qa_fill=200, qa_clear=10, qa_water=11, qa_shadow=12, qa_snow=13,
                         qa_cloud=14)
        renumber = {FILL: 200, CLEAR: 10, WATER: 11, SHADOW: 12, SNOW: 13, CLOUD: 14}
        for name in SCHEMES:
            with self.subTest(name=name):
                default = build(name).codes
                custom = build(name, h).codes
                expected = np.vectorize(renumber.get)(default).astype(np.uint8)
                np.testing.assert_array_equal(custom, expected)

    def test_hls_aerosol_is_opt_in(self):
        plain = ccdcpp.qa_tables.hls_fmask().codes
        strict = ccdcpp.qa_tables.hls_fmask(exclude_high_aerosol=True).codes
        high = np.array([v for v in range(255) if v >> 6 == 3])
        self.assertEqual(plain[0xC0], CLEAR)
        self.assertEqual(strict[0xC0], CLOUD)
        np.testing.assert_array_equal(strict[high], CLOUD)
        rest = np.setdiff1d(np.arange(65536), high)
        np.testing.assert_array_equal(plain[rest], strict[rest])

    def test_custom_table(self):
        codes = np.full(65536, FILL, dtype=np.uint8)
        codes[0] = CLEAR
        table = ccdcpp.QaTable(codes)
        np.testing.assert_array_equal(table.codes, codes)
        with self.assertRaises(ValueError):
            ccdcpp.QaTable(codes[:256])
        with self.assertRaises(ValueError):
            ccdcpp.QaTable(codes.reshape(256, 256))
        with self.assertRaises(TypeError):
            ccdcpp.QaTable(codes.astype(np.float64))


#: pyccd's qabitval (lcmap-pyccd 2021.7.19, ccd/qa.py), written out with the
#: bit offsets of its ccd/parameters.py: fill 0, clear 1, water 2, shadow 3,
#: snow 4, cloud 5, cirrus 8 and 9, occlusion 10. pyccd raises where this
#: returns None. test_qa_references.py runs pyccd itself when it is importable.
def pyccd_qabitval(v: int) -> int | None:
    def bit(o: int) -> int:
        return (v >> o) & 1
    if bit(0):
        return FILL
    if bit(5):
        return CLOUD
    if bit(3):
        return SHADOW
    if bit(4):
        return SNOW
    if bit(2):
        return WATER
    if bit(1):
        return CLEAR
    if bit(8) & bit(9):
        return CLEAR
    if bit(10):
        return CLEAR
    return None


#: The nlcd plugin's QA_BITS (millworks plugins/nlcd science.py), written out.
#: test_qa_references.py compares against the plugin itself.
NLCD_QA_BITS = ((0, FILL), (3, CLOUD), (1, CLOUD), (2, CLOUD), (4, SHADOW), (5, SNOW), (7, WATER))


class Precedence(unittest.TestCase):
    def test_c1_is_pyccd_qabitval(self):
        codes = ccdcpp.qa_tables.landsat_c1_pixel_qa().codes
        for v in range(65536):
            expected = pyccd_qabitval(v)
            self.assertEqual(codes[v], FILL if expected is None else expected, v)

    def test_c2_is_most_excluding_first(self):
        codes = ccdcpp.qa_tables.landsat_c2_qa_pixel().codes
        for v in range(65536):
            expected = next((code for b, code in NLCD_QA_BITS if v >> b & 1), CLEAR)
            self.assertEqual(codes[v], expected, v)


class DecodedEqualsCodes(unittest.TestCase):
    """Raw QA plus a table is the run of the decoded codes passed as uint8.

    The codes include 8 (qa_cirrus1) beside the six classes. The procedures
    read no code but qa_fill, qa_clear, qa_water, qa_snow and qa_cloud, so 8
    is read as qa_shadow is, and it is encoded as each product's shadow."""

    @classmethod
    def setUpClass(cls):
        cls.dates, cls.spectral, cls.codes = synthetic()
        h = options()
        cls.columns = ccdcpp.detect_cube(
            cls.dates, cls.spectral, cls.codes, input=h, threads=1)
        assert cls.columns["failures"] == []
        assert len(cls.columns["row"]) > cls.codes.shape[0] * cls.codes.shape[1]
        # PermanentSnow, InsufficientClear and Standard all ran.
        assert {44, 54, 8} <= set(cls.columns["curve_qa"].tolist())
        # Shadow and code 8 are read alike; everything else is a class.
        cls.classes = np.where(cls.codes == 8, SHADOW, cls.codes)

    def raw_inputs(self, name: str) -> Iterator[tuple[str, NDArray[np.uint8] | NDArray[np.uint16]]]:
        raw = encode(name, self.codes)
        np.testing.assert_array_equal(build(name).codes[raw], self.classes)
        yield "uint16", raw
        if name in NARROW:
            yield "uint8", raw.astype(np.uint8)
        # (H, W, T) seen through (T, H, W) storage.
        yield "strided", np.moveaxis(np.ascontiguousarray(np.moveaxis(raw, 2, 0)), 0, 2)

    def test_detect_cube(self):
        for name in SCHEMES:
            h = with_table(build(name))
            for kind, raw in self.raw_inputs(name):
                for threads in (1, 8):
                    with self.subTest(name=name, qas=kind, threads=threads):
                        before = raw.copy()
                        out = ccdcpp.detect_cube(
                            self.dates, self.spectral, raw, input=h, threads=threads)
                        out.pop("threads")
                        expected = dict(self.columns)
                        expected.pop("threads")
                        assert_same(self, out, expected)
                        np.testing.assert_array_equal(raw, before)

    def test_detect(self):
        h = options()
        H, W, _ = self.codes.shape
        for name in SCHEMES:
            decoding = with_table(build(name))
            raw = encode(name, self.codes)
            for r in range(H):
                for c in range(W):
                    with self.subTest(name=name, pixel=(r, c)):
                        expected = ccdcpp.detect(
                            self.dates, self.spectral[r, c], self.codes[r, c], input=h)
                        assert_same(self, ccdcpp.detect(
                            self.dates, self.spectral[r, c], raw[r, c], input=decoding),
                            expected)
                        if name in NARROW:
                            assert_same(self, ccdcpp.detect(
                                self.dates, self.spectral[r, c], raw[r, c].astype(np.uint8),
                                input=decoding), expected)

    def test_uint16_spectral(self):
        h = options()
        spectral = np.clip(self.spectral, 0, 65535).astype(np.uint16)
        reference = ccdcpp.detect_cube(
            self.dates, spectral, self.codes, input=h, threads=2)
        out = ccdcpp.detect_cube(
            self.dates, spectral, encode("landsat_c2_qa_pixel", self.codes),
            input=with_table(ccdcpp.qa_tables.landsat_c2_qa_pixel()), threads=2)
        assert_same(self, out, reference)

    def test_uint8_codes_through_an_identity_table(self):
        identity = np.full(65536, FILL, dtype=np.uint8)
        identity[:256] = np.arange(256, dtype=np.uint8)
        out = ccdcpp.detect_cube(
            self.dates, self.spectral, self.codes,
            input=with_table(ccdcpp.QaTable(identity)), threads=2)
        out.pop("threads")
        expected = dict(self.columns)
        expected.pop("threads")
        assert_same(self, out, expected)


class Validation(unittest.TestCase):
    def setUp(self):
        self.dates, self.spectral, codes = synthetic(1, 2)
        self.codes = codes
        self.raw = encode("landsat_c2_qa_pixel", codes)

    def entries(self, qas: NDArray[np.integer[Any]],
                table: ccdcpp.QaTable | None = None) -> dict[str, Callable[[], object]]:
        d, s, h = self.dates, self.spectral, with_table(table)
        return {
            "detect": lambda: ccdcpp.detect(d, s[0, 0], qas[0, 0], input=h),
            "detect_cube": lambda: ccdcpp.detect_cube(d, s, qas, input=h, threads=1),
        }

    def test_uint16_without_a_table(self):
        for name, call in self.entries(self.raw).items():
            with self.subTest(entry=name):
                with self.assertRaises(TypeError) as caught:
                    call()
                self.assertIn("uint16", str(caught.exception))
                self.assertIn("qa_table", str(caught.exception))

    def test_table_of_the_wrong_type(self):
        wrong = (
            "landsat_c2_qa_pixel",
            ccdcpp.qa_tables.landsat_c2_qa_pixel().codes,
            ccdcpp.qa_tables.landsat_c2_qa_pixel,
            ccdcpp.Input(),
            1,
        )
        for value in wrong:
            with self.subTest(qa_table=type(value).__name__):
                with self.assertRaises(TypeError):
                    ccdcpp.Input(qa_table=value)  # pyright: ignore[reportArgumentType]  # not a QaTable, refused

    def test_the_input_is_keyword_only(self):
        with self.assertRaises(TypeError):
            ccdcpp.detect_cube(self.dates, self.spectral, self.raw, options())  # pyright: ignore[reportCallIssue]  # positional, refused
        with self.assertRaises(TypeError):
            ccdcpp.detect(self.dates, self.spectral[0, 0], self.raw[0, 0], options())  # pyright: ignore[reportCallIssue]  # positional, refused

    def test_the_table_is_part_of_the_input(self):
        table = ccdcpp.qa_tables.landsat_c2_qa_pixel()
        h = with_table(table)
        self.assertEqual(h, with_table(ccdcpp.qa_tables.landsat_c2_qa_pixel()))
        self.assertNotEqual(h, options())
        self.assertNotEqual(h.digest(), options().digest())
        self.assertEqual(h.digest(), with_table(ccdcpp.QaTable(table.codes)).digest())
        changed = table.codes
        changed[21824] = CLOUD
        self.assertNotEqual(h, with_table(ccdcpp.QaTable(changed)))
        self.assertIsNone(options().qa_table)

    def test_unsafe_qa_dtype_with_a_table(self):
        table = ccdcpp.qa_tables.landsat_c2_qa_pixel()
        for name, call in self.entries(self.raw.astype(np.int64), table).items():
            with self.subTest(entry=name):
                with self.assertRaises(TypeError):
                    call()

    def test_uint8_codes_without_a_table_unchanged(self):
        for name, call in self.entries(self.codes).items():
            with self.subTest(entry=name):
                call()


class Package(unittest.TestCase):
    def test_package_exposes_qa_tables(self):
        # The wheel layout, in a fresh interpreter; see test_bindings.py.
        with tempfile.TemporaryDirectory() as root:
            assemble(root)
            out = subprocess.run(
                [sys.executable, "-I", "-c",
                 "import sys; sys.path.insert(0, sys.argv[1]); "
                 "import ccdcpp.qa_tables as q; import ccdcpp; "
                 "print(q is ccdcpp.qa_tables, isinstance(q.hls_fmask(), ccdcpp.QaTable))",
                 root],
                check=True, capture_output=True, text=True,
            ).stdout.split()
        self.assertEqual(out, ["True", "True"])


if __name__ == "__main__":
    unittest.main()
