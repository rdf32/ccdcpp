"""The Landsat presets: their inputs and outputs against the inputs written out,
their bounds, and a profile with a configuration.

usage: python test_profiles_landsat.py <module_dir> <source_dir> [unittest args]

<module_dir> holds the built extension; <source_dir> is the repository root.
The package is assembled as the wheel lays it out (_harness.use_package()).
ctest runs this as `python_profiles_landsat`.
"""
from __future__ import annotations

import dataclasses
import pickle
import unittest

from _harness import use_package

use_package()


import ccdcpp  # noqa: E402
# The extension module, which ccdcpp/ccdcpp.pyi types.
from ccdcpp import ccdcpp as ext  # noqa: E402
from ccdcpp import presets  # noqa: E402

from _compare import assert_same  # noqa: E402
from _corpus import (CODES, CUBE_DN, CUBE_PHYS, DATES, PYCCD, every_entry,  # noqa: E402
                     strip_band_names, without)
from _profiles import (change_threshold, doc_of, fields, outlier_threshold,  # noqa: E402
                       with_table)


LANDSAT_PRESETS = (presets.landsat_c1_dn, presets.landsat_c1_physical, presets.landsat_c2)


def explicit_physical() -> tuple[ccdcpp.Input, ccdcpp.Config]:
    """The physical-unit input written out, as the golden tests write it, and the
    configuration it runs: (Input, Config)."""
    h = ccdcpp.Input(reflectance_scale=1.0, thermal_scale=1.0, thermal_offset=0.0,
                     thermal_at_100c=1.0)
    return h, PYCCD


class Corpus(unittest.TestCase):
    def test_corpus_runs_every_procedure(self):
        h = ccdcpp.Input()
        out = ext.detect_cube(DATES, CUBE_DN, CODES, input=h, threads=1)
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
        profiled = every_entry(DATES, CUBE_DN, CODES, input=presets.landsat_c1_dn())
        plain = every_entry(DATES, CUBE_DN, CODES, input=h)
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
        profiled = every_entry(DATES, CUBE_PHYS, CODES, input=presets.landsat_c1_physical(),
                               config=cfg)
        plain = every_entry(DATES, CUBE_PHYS, CODES, input=h, config=cfg)
        assert_same(self, strip_band_names(profiled), plain)

    def test_presets_state_units_and_band_order(self):
        for preset in LANDSAT_PRESETS:
            with self.subTest(preset=preset.__name__):
                self.assertIn("Input:", doc_of(preset))
                self.assertIn("QA:", doc_of(preset))
                self.assertEqual(preset().band_names,
                                 ("blue", "green", "red", "nir", "swir1", "swir2", "thermal"))
                self.assertEqual(preset().name, preset.__name__)
        self.assertIn("blue, green, red, nir, swir1, swir2, thermal", doc_of(presets))
        self.assertEqual(presets.__all__, ["landsat_c1_dn", "landsat_c1_physical", "landsat_c2",
                                           "hls_l30", "hls_s30", "hls", "sentinel2_l2a"])


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
                        preprocess__band_range__inclusive=list(refl),
                        preprocess__thermal_range__inclusive=list(therm))
                    snow = ccdcpp.plan_of(cfg, base).snow.chain
                    standard = ccdcpp.plan_of(cfg, base).standard.chain
                    for thermal, band in ((snow[1], snow[2]), (snow[6], snow[2]),
                                          (standard[1], standard[2])):
                        self.assertEqual(thermal.inclusive, therm)
                        self.assertEqual(band.inclusive, refl)

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
            ccdcpp.detect(dates, CUBE_DN[0, 0], CODES[0, 0], input=profile)
        self.assertIn("dates must be non-decreasing", str(caught.exception))
        with self.assertRaises(ValueError):
            ccdcpp.detect_cube(dates, CUBE_DN, CODES, threads=1, input=profile)


class ProfileAndConfig(unittest.TestCase):
    def test_a_profile_is_its_input(self):
        # input=profile runs profile.input(); config= is the configuration
        # as given, the same with or without a profile.
        h_phys, cfg_phys = explicit_physical()
        s, q = CUBE_PHYS[0, 0], CODES[0, 0]
        profile = presets.landsat_c1_physical()
        for cfg in (ccdcpp.config("ccdc"), cfg_phys):
            assert_same(self, without(ccdcpp.detect(DATES, s, q, input=profile, config=cfg),
                                      "band_names"),
                        ext.detect(DATES, s, q, input=h_phys, config=cfg._algorithm))
        # The default is "ccdc", whatever the profile.
        assert_same(self, ccdcpp.detect(DATES, s, q, input=profile),
                    ccdcpp.detect(DATES, s, q, input=profile, config=ccdcpp.config("ccdc")))


class BandNames(unittest.TestCase):
    def test_only_with_a_profile(self):
        h = ccdcpp.Input()
        names = presets.landsat_c1_dn().band_names
        plain = every_entry(DATES, CUBE_DN, CODES, input=h)
        profiled = every_entry(DATES, CUBE_DN, CODES, input=presets.landsat_c1_dn())
        for out in plain["detect"] + [plain["detect_cube"]]:
            self.assertNotIn("band_names", out)
        for out in profiled["detect"] + [profiled["detect_cube"]]:
            self.assertEqual(out["band_names"], names)
            self.assertIsInstance(out["band_names"], tuple)


if __name__ == "__main__":
    unittest.main()
