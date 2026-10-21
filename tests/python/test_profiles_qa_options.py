"""A profile's qa_table_options: what reaches the QA table builder, and what is refused.

usage: python test_profiles_qa_options.py <module_dir> <source_dir> [unittest args]

<module_dir> holds the built extension; <source_dir> is the repository root.
The package is assembled as the wheel lays it out (_harness.use_package()).
ctest runs this as `python_profiles_qa_options`.
"""
from __future__ import annotations

import dataclasses
import pickle
import unittest
from typing import Any
from unittest import mock

from _harness import use_package

use_package()

import numpy as np  # noqa: E402

import ccdcpp  # noqa: E402
from ccdcpp import presets  # noqa: E402

from _compare import assert_same  # noqa: E402
from _corpus import (CLEAR, CLOUD, CODES, CUBE_PHYS, DATES, FILL, SHADOW, SNOW,  # noqa: E402
                     WATER, every_entry, strip_band_names)
from _profiles import doc_of, qa_table_of, with_table  # noqa: E402


#: One raw HLS v2.0 Fmask value per class: fill 255, bit 5 water, bit 3
#: shadow, bit 4 snow, bit 1 cloud; 0 is clear with climatology aerosol.
FMASK_RAW = {FILL: 255, CLEAR: 0, WATER: 32, SHADOW: 8, SNOW: 16, CLOUD: 2}


#: Every raw HLS Fmask value, for comparing whole tables.
ALL_VALUES = np.arange(65536)


class QaTableOptionsField(unittest.TestCase):
    def hls(self, **kw: Any) -> ccdcpp.SensorProfile:
        return dataclasses.replace(presets.landsat_c1_physical(), qa_table_builder="hls_fmask",
                                   **kw)

    def test_default_is_empty(self):
        for preset in (presets.landsat_c1_dn, presets.landsat_c1_physical, presets.landsat_c2):
            with self.subTest(preset=preset.__name__):
                self.assertEqual(preset().qa_table_options, ())
        np.testing.assert_array_equal(qa_table_of(self.hls().input()).codes,
                                      ccdcpp.qa_tables.hls_fmask().codes)

    def test_builder_arguments_reach_the_builder(self):
        for aerosol in (False, True):
            with self.subTest(exclude_high_aerosol=aerosol):
                p = self.hls(qa_table_options={"exclude_high_aerosol": aerosol})
                h = p.input()
                expected = ccdcpp.qa_tables.hls_fmask(h, exclude_high_aerosol=aerosol).codes
                got = qa_table_of(p.input()).codes
                self.assertEqual(got.shape, ALL_VALUES.shape)
                np.testing.assert_array_equal(got, expected)
        # The two tables differ exactly where the aerosol bits are 11 and no
        # higher-precedence bit (fill, cloud, adjacent, cirrus) is set.
        on = qa_table_of(self.hls(qa_table_options={"exclude_high_aerosol": True}).input()).codes
        off = qa_table_of(self.hls().input()).codes
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
        np.testing.assert_array_equal(qa_table_of(q.input()).codes, qa_table_of(p.input()).codes)

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
        got = strip_band_names(every_entry(DATES, CUBE_PHYS, fmask, input=screened))
        assert_same(self, got, every_entry(DATES, CUBE_PHYS, fmask,
                                           input=with_table(base.input(), table)))
        codes = np.where(aerosol, np.uint8(CLOUD), CODES)
        assert_same(self, got, every_entry(DATES, CUBE_PHYS, codes,
                                           input=with_table(base.input(), None)))
        self.assertNotEqual(
            ccdcpp.detect(DATES, CUBE_PHYS[0, 1], fmask[0, 1], input=base)["processing_mask"]
            .tobytes(),
            ccdcpp.detect(DATES, CUBE_PHYS[0, 1], fmask[0, 1], input=screened)["processing_mask"]
            .tobytes())

    def test_options_the_builder_does_not_take_are_refused(self):
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
                signature = doc_of(getattr(ccdcpp.qa_tables, builder)).splitlines()[0]
                self.assertEqual(signature.count(": "), 1 + len(declared), signature)
                self.assertIn(f"{builder}(input: ", signature)
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
        options = p.qa_table_options
        assert isinstance(options, tuple)
        self.assertIs(type(options[0][0]), str)

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


if __name__ == "__main__":
    unittest.main()
