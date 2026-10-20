"""Preprocessing plans: the bound classes, ccdcpp.Preprocess and preprocess=.

usage: python test_preprocess.py <module_dir> <source_dir> [unittest args]

<module_dir> holds the built extension; <source_dir> is the repository root.
The package, the corpus cube and the comparison helpers are test_profiles'
(imported from beside this file). ctest runs this as `python_preprocess`.
"""
import dataclasses
import os
import pickle
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_profiles as tp  # noqa: E402  (reads and removes argv[1:3])

import numpy as np  # noqa: E402

import ccdcpp  # noqa: E402
from ccdcpp import presets  # noqa: E402

from ccdcpp import (  # noqa: E402
    AddQa,
    BandRange,
    Dedupe,
    GreenMedian,
    KeepQa,
    Preprocess,
    ProcedureChain,
    ThermalRange,
    Transform,
)

EXT = ccdcpp.ccdcpp
DATES, CUBE_DN, CUBE_PHYS, CODES = tp.DATES, tp.CUBE_DN, tp.CUBE_PHYS, tp.CODES
SNOW = tp.SNOW
assert_same, every_entry, without = tp.assert_same, tp.every_entry, tp.without


def strip(outs):
    return tp.strip_band_names(outs)


def keeps_snow():
    """The Standard procedure keeping snow observations, by role."""
    return Preprocess(standard=ProcedureChain(
        steps=(KeepQa(("clear", "water", "snow")), ThermalRange(), BandRange(), Dedupe()),
    ))


def keeps_snow_plan(h):
    """keeps_snow(), built on the bound classes."""
    plan = ccdcpp.default_plan(h)
    plan.standard.chain[0] = ccdcpp.Step.keep_qa([h.QA_CLEAR, h.QA_WATER, h.QA_SNOW])
    return plan


def written_default(thermal_scale=10.0, thermal_offset=-27315.0):
    """The default plan of a Landsat preset, written out step by step."""
    common = (KeepQa(), ThermalRange(), BandRange(), Dedupe())
    return Preprocess(
        standard=ProcedureChain(
            steps=common,
            transform=Transform("thermal", thermal_scale, thermal_offset),
        ),
        snow=ProcedureChain(steps=common + (AddQa(), Dedupe())),
        insufficient_clear=ProcedureChain(steps=common + (GreenMedian(), Dedupe())),
    )


def written_default_explicit():
    """landsat_c1_dn's default plan with every value spelled out."""
    sat = ("blue", "green", "red", "nir", "swir1", "swir2")
    common = (
        KeepQa(("clear", 1)),
        ThermalRange(min=-9320.0, max=7070.0, inclusive=(False, False)),
        BandRange(roles=sat, min=0.0, max=10000.0, inclusive=(False, False)),
        Dedupe(),
    )
    return Preprocess(
        standard=ProcedureChain(steps=common, transform=Transform("thermal", 10.0, -27315.0)),
        snow=ProcedureChain(steps=common + (AddQa((SNOW,)), Dedupe())),
        insufficient_clear=ProcedureChain(steps=common + (
            GreenMedian(offset=400.0, scope="stat_window", max_date=736694, role="green"),
            Dedupe(),
        )),
    )


def snow_kept(out, codes):
    return int(np.sum((out["processing_mask"] == 1) & (codes == SNOW)))


class BoundClasses(unittest.TestCase):
    def test_default_plan_shape(self):
        h = ccdcpp.HarmonicOptions()
        plan = ccdcpp.default_plan(h)
        K = ccdcpp.StepKind
        common = [K.KeepQa, K.ThermalRange, K.BandRange, K.Dedupe]
        self.assertEqual([s.kind for s in plan.standard.chain], common)
        self.assertEqual([s.kind for s in plan.snow.chain], common + [K.AddQa, K.Dedupe])
        self.assertEqual([s.kind for s in plan.insufficient_clear.chain],
                         common + [K.GreenMedian, K.Dedupe])
        t = plan.standard.transform
        self.assertEqual((t.band, t.scale, t.offset), (6, 10.0, -27315.0))
        self.assertIsNone(plan.snow.transform)
        self.assertIsNone(plan.insufficient_clear.transform)
        keep, therm, band, _ = plan.standard.chain
        self.assertEqual(keep.codes, [h.QA_CLEAR, h.QA_WATER])
        self.assertEqual((therm.band, therm.min, therm.max, therm.inclusive),
                         (6, h.THERM_MIN, h.THERM_MAX, (False, False)))
        self.assertEqual((band.bands, band.min, band.max), (h.SATURATION_BANDS, 0.0, 10000.0))
        green = plan.insufficient_clear.chain[4]
        self.assertEqual((green.band, green.green_offset, green.max_date, green.scope),
                         (1, 400.0, h.STAT_ORD, ccdcpp.GreenFilterScope.StatWindow))
        self.assertEqual(plan.snow.chain[-2].codes, [h.QA_SNOW])

    def test_without_thermal_and_identity_transform(self):
        h = ccdcpp.HarmonicOptions()
        h.THERMAL_IDX = ccdcpp.NO_BAND
        plan = ccdcpp.default_plan(h)
        self.assertNotIn(ccdcpp.StepKind.ThermalRange, [s.kind for s in plan.standard.chain])
        self.assertIsNone(plan.standard.transform)
        h = ccdcpp.HarmonicOptions()
        h.THERMAL_SCALE, h.THERMAL_OFFSET = 1.0, 0.0
        self.assertIsNone(ccdcpp.default_plan(h).standard.transform)

    def test_equality_and_mutation(self):
        h = ccdcpp.HarmonicOptions()
        plan = ccdcpp.default_plan(h)
        self.assertEqual(plan, ccdcpp.default_plan(h))
        plan.standard.chain[0].codes = [0, 1, 3]
        self.assertEqual(plan.standard.chain[0].codes, [0, 1, 3])
        self.assertNotEqual(plan, ccdcpp.default_plan(h))
        self.assertEqual(plan, keeps_snow_plan(h))
        chain = ccdcpp.Chain([ccdcpp.Step.dedupe()])
        chain.append(ccdcpp.Step.keep_qa([0]))
        self.assertEqual(len(chain), 2)
        self.assertEqual(chain[-1].kind, ccdcpp.StepKind.KeepQa)
        with self.assertRaises(IndexError):
            chain[2]

    def test_describe(self):
        text = ccdcpp.default_plan(ccdcpp.HarmonicOptions()).describe()
        self.assertEqual(text, repr(ccdcpp.default_plan(ccdcpp.HarmonicOptions())))
        for line in ("standard", "  transform  band 6 <- x * 10 - 27315",
                     "  0  KeepQa codes {0, 1}", "  1  ThermalRange band 6, -9320 < x < 7070",
                     "  2  BandRange bands [0, 1, 2, 3, 4, 5], 0 < x < 10000", "  3  Dedupe",
                     "snow", "  transform  none", "  4  AddQa codes {3}", "insufficient_clear",
                     "  4  GreenMedian band 1, drop x >= median(dates <= 736694) + 400, "
                     "scope StatWindow"):
            self.assertIn(line, text.splitlines())
        step = ccdcpp.Step.band_range([0, 1], 0.0, 1.0, (True, False))
        self.assertEqual(repr(step), "<Step BandRange bands [0, 1], 0 <= x < 1>")

    def test_readme_lists_the_default_plan(self):
        with open(os.path.join(tp.SOURCE_DIR, "README.md"), encoding="utf-8") as f:
            readme = f.read()
        self.assertIn(ccdcpp.default_plan(ccdcpp.HarmonicOptions()).describe(), readme)


class DefaultPlan(unittest.TestCase):
    """The default plan, passed explicitly, is the null plan to the bit."""

    def test_extension_default_is_null(self):
        h = ccdcpp.HarmonicOptions()
        plan = ccdcpp.default_plan(h)
        assert_same(self, every_entry(DATES, CUBE_DN, CODES, hoptions=h, preprocess=plan),
                    every_entry(DATES, CUBE_DN, CODES, hoptions=h))
        for dtype in (np.float32, np.uint16):
            with self.subTest(dtype=dtype.__name__):
                s = CUBE_DN.astype(dtype)
                assert_same(self,
                            without(EXT.detect_cube_columns(DATES, s, CODES, h, 2,
                                                            preprocess=plan), "threads"),
                            without(EXT.detect_cube_columns(DATES, s, CODES, h, 2),
                                    "threads"))

    def test_profile_plans_are_null(self):
        for preset, cube in ((presets.landsat_c1_dn, CUBE_DN),
                             (presets.landsat_c1_physical, CUBE_PHYS)):
            with self.subTest(preset=preset.__name__):
                profile = preset()
                plan = profile.preprocess_plan()
                self.assertEqual(plan, ccdcpp.default_plan(profile.options()))
                assert_same(self, every_entry(DATES, cube, CODES, profile=profile,
                                              preprocess=plan),
                            every_entry(DATES, cube, CODES, profile=profile))

    def test_written_default_equals_default(self):
        dn = presets.landsat_c1_dn()
        h = dn.options()
        self.assertEqual(Preprocess().plan(dn), ccdcpp.default_plan(h))
        self.assertEqual(written_default().plan(dn), ccdcpp.default_plan(h))
        self.assertEqual(written_default_explicit().plan(dn), ccdcpp.default_plan(h))
        # Physical units: the identity transform is no transform.
        phys = presets.landsat_c1_physical()
        written = Preprocess(
            standard=ProcedureChain(steps=written_default().standard.steps, transform=None),
            snow=written_default().snow,
            insufficient_clear=written_default().insufficient_clear,
        )
        self.assertEqual(written.plan(phys), ccdcpp.default_plan(phys.options()))

    def test_written_default_output(self):
        for preprocess in (Preprocess(), written_default(), written_default_explicit()):
            with self.subTest(preprocess=preprocess):
                profile = dataclasses.replace(presets.landsat_c1_dn(), preprocess=preprocess)
                assert_same(self, every_entry(DATES, CUBE_DN, CODES, profile=profile),
                            every_entry(DATES, CUBE_DN, CODES, profile=presets.landsat_c1_dn()))

    def test_transform_default_none_and_given(self):
        dn = presets.landsat_c1_dn()
        h = dn.options()
        default = ccdcpp.default_plan(h)
        self.assertEqual(ProcedureChain().transform, ccdcpp.DEFAULT_TRANSFORM)
        plan = Preprocess(standard=ProcedureChain(steps=(KeepQa(), Dedupe()))).plan(dn)
        self.assertEqual(plan.standard.transform, default.standard.transform)
        plan = Preprocess(standard=ProcedureChain(transform=None),
                          snow=ProcedureChain(transform=Transform("thermal", 2.0, 1.0))).plan(dn)
        self.assertIsNone(plan.standard.transform)
        self.assertEqual(plan.standard.chain, default.standard.chain)
        self.assertEqual(plan.snow.transform, ccdcpp.Affine(6, 2.0, 1.0))
        with self.assertRaises(TypeError):
            ProcedureChain(transform=ccdcpp.Affine(6, 2.0, 1.0))

    def test_ten_band_default(self):
        profile = tp.ten_band_profile()
        dates, cube, scl = tp.ten_band_inputs()
        common = (KeepQa(), BandRange(), Dedupe())
        written = Preprocess(
            standard=ProcedureChain(steps=common),
            snow=ProcedureChain(steps=common + (AddQa(), Dedupe())),
            insufficient_clear=ProcedureChain(steps=common + (GreenMedian(), Dedupe())),
        )
        self.assertEqual(written.plan(profile), ccdcpp.default_plan(profile.options()))
        with_plan = dataclasses.replace(profile, preprocess=written)
        assert_same(self,
                    ccdcpp.detect_cube_columns(dates, cube, scl, threads=2, profile=with_plan),
                    ccdcpp.detect_cube_columns(dates, cube, scl, threads=2, profile=profile))


class CustomPlan(unittest.TestCase):
    def test_standard_keeps_snow(self):
        dates, ref, qa = tp.reference_pixel()
        h = ccdcpp.HarmonicOptions()
        base = EXT.detect(dates, ref, qa, h)
        snowy = EXT.detect(dates, ref, qa, h, preprocess=keeps_snow_plan(h))
        self.assertNotIn(54, [m["curve_qa"] for m in snowy["models"]])
        self.assertEqual(snow_kept(base, qa), 0)
        self.assertGreater(snow_kept(snowy, qa), 0)
        self.assertNotEqual(repr(base), repr(snowy))
        profile = dataclasses.replace(presets.landsat_c1_dn(), preprocess=keeps_snow())
        self.assertEqual(profile.preprocess_plan(), keeps_snow_plan(h))
        assert_same(self, without(ccdcpp.detect(dates, ref, qa, profile=profile), "band_names"),
                    snowy)

    def test_transform_on_every_procedure(self):
        # Thermal handed over in Kelvin x 10 and converted on every
        # procedure is the default plan on input converted beforehand with
        # the identity transform.
        to_c100 = Transform("thermal", 10.0, -27315.0)
        everywhere = Preprocess(
            standard=ProcedureChain(transform=to_c100),
            snow=ProcedureChain(transform=to_c100),
            insufficient_clear=ProcedureChain(transform=to_c100),
        )
        profile = dataclasses.replace(presets.landsat_c1_dn(), preprocess=everywhere)
        cube = CUBE_DN.copy()
        cube[..., 6, ::7] = 1000.0
        converted = cube.copy()
        converted[..., 6, :] = converted[..., 6, :] * 10.0 - 27315.0
        h = ccdcpp.HarmonicOptions()
        h.THERMAL_SCALE, h.THERMAL_OFFSET = 1.0, 0.0
        got = ccdcpp.detect_cube(DATES, cube, CODES, threads=2, profile=profile)
        assert_same(self, got, ccdcpp.detect_cube(DATES, converted, CODES, h, 2))
        self.assertNotEqual(
            repr(got), repr(ccdcpp.detect_cube(DATES, cube, CODES, threads=2,
                                               profile=presets.landsat_c1_dn())))


class ProfileValues(unittest.TestCase):
    def test_hashable_and_picklable(self):
        a = dataclasses.replace(presets.landsat_c1_dn(), preprocess=keeps_snow())
        b = dataclasses.replace(presets.landsat_c1_dn(), preprocess=keeps_snow())
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(a, presets.landsat_c1_dn())
        self.assertNotEqual(a, dataclasses.replace(a, preprocess=written_default()))
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            back = pickle.loads(pickle.dumps(a, protocol=proto))
            self.assertEqual(back, a)
            self.assertEqual(hash(back), hash(a))
            self.assertEqual(back.preprocess_plan(), a.preprocess_plan())
        self.assertEqual(len({a, b, presets.landsat_c1_dn()}), 2)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            a.preprocess.standard.steps[0].codes = ()

    def test_profiles_store_dataclasses(self):
        p = dataclasses.replace(presets.landsat_c2(), preprocess=written_default())
        stack = [p.preprocess]
        while stack:
            value = stack.pop()
            self.assertFalse(type(value).__module__.endswith("ccdcpp.ccdcpp"), value)
            if dataclasses.is_dataclass(value):
                stack.extend(getattr(value, f.name) for f in dataclasses.fields(value))
            elif isinstance(value, tuple):
                stack.extend(value)

    def test_presets_have_no_preprocess(self):
        for preset in (presets.landsat_c1_dn, presets.landsat_c1_physical, presets.landsat_c2):
            self.assertIsNone(preset().preprocess)

    def test_codes_follow_the_run_options(self):
        h = presets.landsat_c1_dn().options()
        h.QA_CLEAR, h.QA_WATER, h.QA_SNOW = 10, 11, 13
        plan = keeps_snow().plan(presets.landsat_c1_dn(), h)
        self.assertEqual(plan.standard.chain[0].codes, [10, 11, 13])
        self.assertEqual(plan.snow.chain[4].codes, [13])
        profile = dataclasses.replace(presets.landsat_c1_dn(), preprocess=keeps_snow())
        renumbered = np.vectorize({255: 255, 0: 10, 1: 11, 2: 2, 3: 13, 4: 4}.get,
                                  otypes=[np.uint8])(CODES)
        assert_same(self,
                    every_entry(DATES, CUBE_DN, renumbered, hoptions=h, profile=profile),
                    every_entry(DATES, CUBE_DN, CODES, profile=profile))


class Validation(unittest.TestCase):
    def test_validate_plan(self):
        h = ccdcpp.HarmonicOptions()
        ccdcpp.validate_plan(ccdcpp.default_plan(h), 7)
        with self.assertRaisesRegex(ValueError, "standard step 1 .ThermalRange.: band 6"):
            ccdcpp.validate_plan(ccdcpp.default_plan(h), 6)
        lone = ccdcpp.PreprocessPlan(
            snow=ccdcpp.ProcedurePlan(chain=ccdcpp.Chain([ccdcpp.Step.add_qa([3])])))
        with self.assertRaisesRegex(ValueError, "snow step 0 .AddQa.: must be followed by"):
            ccdcpp.validate_plan(lone, 7)
        no_band = ccdcpp.default_plan(h)
        no_band.insufficient_clear.chain[4].band = ccdcpp.NO_BAND
        with self.assertRaisesRegex(ValueError, "insufficient_clear step 4 .GreenMedian."):
            ccdcpp.validate_plan(no_band, 7)
        tested = ccdcpp.default_plan(h)
        tested.standard.transform = ccdcpp.Affine(0, 2.0, 0.0)
        with self.assertRaisesRegex(ValueError, "standard transform: band 0 is also tested"):
            ccdcpp.validate_plan(tested, 7)

    def test_construction(self):
        with self.assertRaisesRegex(ValueError, "at most 8 steps"):
            ccdcpp.Chain([ccdcpp.Step.dedupe()] * 9)
        chain = ccdcpp.Chain([ccdcpp.Step.dedupe()] * 8)
        with self.assertRaises(ValueError):
            chain.append(ccdcpp.Step.dedupe())
        with self.assertRaises(ValueError):
            ccdcpp.Step.keep_qa([256])
        with self.assertRaises(ValueError):
            ccdcpp.Step.band_range([ccdcpp.CCD_MAX_BANDS], 0.0, 1.0)

    def test_entry_points_raise_before_running(self):
        h = ccdcpp.HarmonicOptions()
        bad = ccdcpp.default_plan(h)
        bad.insufficient_clear.chain[4].band = 9
        with self.assertRaisesRegex(ValueError, "band 9 is not below the band count 7"):
            EXT.detect(DATES, CUBE_DN[0, 0], CODES[0, 0], h, preprocess=bad)
        for entry in (ccdcpp.detect_cube, ccdcpp.detect_cube_columns):
            with self.subTest(entry=entry.__name__):
                with self.assertRaises(ValueError):
                    entry(DATES, CUBE_DN, CODES, h, 1, preprocess=bad)
        with self.assertRaises(TypeError):
            EXT.detect(DATES, CUBE_DN[0, 0], CODES[0, 0], h, preprocess="default")

    def test_dataclasses(self):
        with self.assertRaisesRegex(ValueError, "unknown QA code name 'cirrus'"):
            KeepQa(("clear", "cirrus"))
        with self.assertRaises(ValueError):
            AddQa((300,))
        with self.assertRaisesRegex(ValueError, "scope is 'StatWindow'"):
            GreenMedian(scope="StatWindow")
        self.assertEqual(GreenMedian(scope=ccdcpp.GreenFilterScope.Off), GreenMedian(scope="off"))
        with self.assertRaises(ValueError):
            ThermalRange(inclusive=(True,))
        with self.assertRaises(TypeError):
            ProcedureChain(steps=(ccdcpp.Step.dedupe(),))
        with self.assertRaises(TypeError):
            Preprocess(standard=(KeepQa(),))

    def test_profile_roles(self):
        dn = presets.landsat_c1_dn()
        with self.assertRaisesRegex(ValueError, "preprocess names 'coastal'"):
            dataclasses.replace(dn, preprocess=Preprocess(
                standard=ProcedureChain(steps=(BandRange(roles=("coastal",)),))))
        with self.assertRaisesRegex(ValueError, "preprocess names 'thermal'"):
            dataclasses.replace(tp.ten_band_profile(), preprocess=Preprocess(
                snow=ProcedureChain(steps=(ThermalRange(),))))
        with self.assertRaisesRegex(ValueError, "not a ccdcpp.Preprocess"):
            dataclasses.replace(dn, preprocess=ccdcpp.default_plan(ccdcpp.HarmonicOptions()))
        tested = dataclasses.replace(dn, preprocess=Preprocess(
            standard=ProcedureChain(transform=Transform("blue", 2.0))))
        with self.assertRaisesRegex(ValueError, "band 0 is also tested"):
            tested.preprocess_plan()
        with self.assertRaises(ValueError):
            ccdcpp.detect(DATES, CUBE_DN[0, 0], CODES[0, 0], profile=tested)
        unpaired = dataclasses.replace(dn, preprocess=Preprocess(
            snow=ProcedureChain(steps=(KeepQa(), AddQa()))))
        with self.assertRaisesRegex(ValueError, "must be followed by a Dedupe"):
            unpaired.preprocess_plan()


class Sentinel(unittest.TestCase):
    """preprocess= follows qa_table=: the profile's by default, an explicit
    value instead, None for the default plan."""

    def setUp(self):
        self.dates, self.ref, self.qa = tp.reference_pixel()
        self.h = ccdcpp.HarmonicOptions()
        self.snowy = EXT.detect(self.dates, self.ref, self.qa, self.h,
                                preprocess=keeps_snow_plan(self.h))
        self.base = EXT.detect(self.dates, self.ref, self.qa, self.h)
        self.profile = dataclasses.replace(presets.landsat_c1_dn(), preprocess=keeps_snow())

    def run_detect(self, **kw):
        return without(ccdcpp.detect(self.dates, self.ref, self.qa, **kw), "band_names")

    def test_profile_plan_by_default(self):
        assert_same(self, self.run_detect(profile=self.profile), self.snowy)

    def test_none_is_the_default_plan(self):
        assert_same(self, self.run_detect(profile=self.profile, preprocess=None), self.base)

    def test_explicit_plan_wins(self):
        assert_same(self, self.run_detect(profile=presets.landsat_c1_dn(),
                                          preprocess=keeps_snow_plan(self.h)), self.snowy)
        assert_same(self, self.run_detect(profile=self.profile,
                                          preprocess=ccdcpp.default_plan(self.h)), self.base)
        assert_same(self, self.run_detect(profile=presets.landsat_c1_dn(),
                                          preprocess=keeps_snow()), self.snowy)

    def test_explicit_options_keep_the_profile_plan(self):
        assert_same(self, self.run_detect(profile=self.profile, hoptions=self.h,
                                          config=ccdcpp.config("ccdc")), self.snowy)

    def test_without_a_profile(self):
        assert_same(self, ccdcpp.detect(self.dates, self.ref, self.qa, self.h), self.base)
        assert_same(self, ccdcpp.detect(self.dates, self.ref, self.qa, self.h,
                                        preprocess=None), self.base)
        assert_same(self, ccdcpp.detect(self.dates, self.ref, self.qa, self.h,
                                        preprocess=keeps_snow_plan(self.h)), self.snowy)
        with self.assertRaisesRegex(TypeError, "needs profile="):
            ccdcpp.detect(self.dates, self.ref, self.qa, self.h, preprocess=keeps_snow())

    def test_cube_entry_points(self):
        h = self.h
        plain = every_entry(DATES, CUBE_DN, CODES, hoptions=h, preprocess=keeps_snow_plan(h))
        assert_same(self, strip(every_entry(DATES, CUBE_DN, CODES, profile=self.profile)), plain)
        assert_same(self, strip(every_entry(DATES, CUBE_DN, CODES, profile=self.profile,
                                            preprocess=None)),
                    every_entry(DATES, CUBE_DN, CODES, hoptions=h))

    def test_plan_is_copied_at_the_call(self):
        plan = keeps_snow_plan(self.h)
        first = EXT.detect(self.dates, self.ref, self.qa, self.h, preprocess=plan)
        plan.standard.chain[0] = ccdcpp.Step.keep_qa([self.h.QA_CLEAR, self.h.QA_WATER])
        assert_same(self, first, self.snowy)
        assert_same(self, EXT.detect(self.dates, self.ref, self.qa, self.h, preprocess=plan),
                    self.base)


if __name__ == "__main__":
    unittest.main()
