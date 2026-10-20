"""Preprocessing as configuration: the step classes, Config's preprocess section, plan_of.

usage: python test_preprocess.py <module_dir> <source_dir> [unittest args]

<module_dir> holds the built extension; <source_dir> is the repository root.
The package, the corpus cube and the comparison helpers are test_profiles'
(imported from beside this file). ctest runs this as `python_preprocess`.
"""
import os
import sys
import tomllib
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
)

DATES, CUBE_DN, CUBE_PHYS, CODES = tp.DATES, tp.CUBE_DN, tp.CUBE_PHYS, tp.CODES
SNOW = tp.SNOW
assert_same, every_entry, without = tp.assert_same, tp.every_entry, tp.without
CCDC = ccdcpp.config("ccdc")


def strip(outs):
    return tp.strip_band_names(outs)


def keeps_snow(base=CCDC):
    """`base` with the Standard procedure keeping snow observations, by name."""
    return base.replace(preprocess__standard=ProcedureChain(
        steps=(KeepQa(("clear", "water", "snow")), ThermalRange(), BandRange(), Dedupe()),
        thermal_transform=True,
    ))


def written_default():
    """pyccd's chain, written out step by step with every field left to its default."""
    common = (KeepQa(), BandRange(), Dedupe())
    return Preprocess(
        standard=ProcedureChain(steps=(KeepQa(), ThermalRange(), BandRange(), Dedupe()),
                                thermal_transform=True),
        snow=ProcedureChain(steps=common + (AddQa(), Dedupe())),
        insufficient_clear=ProcedureChain(steps=common + (GreenMedian(), Dedupe())),
    )


def written_default_explicit():
    """pyccd's chain with every value spelled out, in physical terms."""
    sat = ("blue", "green", "red", "nir", "swir1", "swir2")
    keep = KeepQa(("clear", "water"))
    band = BandRange(bands=sat, min=0.0, max=1.0, inclusive=(False, False))
    common = (keep, band, Dedupe())
    return Preprocess(
        standard=ProcedureChain(
            steps=(keep, ThermalRange(min=-93.2, max=70.7, inclusive=(False, False)), band,
                   Dedupe()),
            thermal_transform=True),
        snow=ProcedureChain(steps=common + (AddQa(("snow",)), Dedupe())),
        insufficient_clear=ProcedureChain(steps=common + (
            GreenMedian(offset=0.04, scope="stat_window"), Dedupe(),
        )),
    )


def snow_kept(out, codes):
    return int(np.sum((out["processing_mask"] == 1) & (codes == SNOW)))


class Plans(unittest.TestCase):
    def test_ccdc_plan_shape(self):
        h = ccdcpp.Input()
        plan = ccdcpp.plan_of("ccdc", h)
        K = ccdcpp.StepKind
        common = [K.KeepQa, K.BandRange, K.Dedupe]
        self.assertEqual([s.kind for s in plan.standard.chain],
                         [K.KeepQa, K.ThermalRange, K.BandRange, K.Dedupe])
        self.assertEqual([s.kind for s in plan.snow.chain], common + [K.AddQa, K.Dedupe])
        self.assertEqual([s.kind for s in plan.insufficient_clear.chain],
                         common + [K.GreenMedian, K.Dedupe])
        t = plan.standard.transform
        self.assertEqual((t.band, t.scale, t.offset), (6, 10.0, -27315.0))
        self.assertIsNone(plan.snow.transform)
        self.assertIsNone(plan.insufficient_clear.transform)
        keep, therm, band, _ = plan.standard.chain
        self.assertEqual(keep.codes, [h.qa_clear, h.qa_water])
        # pyccd's DN bounds, exactly: -93.2 and 70.7 degrees Celsius, reflectance (0, 1).
        self.assertEqual((therm.band, therm.min, therm.max, therm.inclusive),
                         (6, -9320.0, 7070.0, (False, False)))
        self.assertEqual((band.bands, band.min, band.max), ([0, 1, 2, 3, 4, 5], 0.0, 10000.0))
        green = plan.insufficient_clear.chain[3]
        self.assertEqual((green.band, green.green_offset, green.max_date, green.scope),
                         (1, 400.0, ccdcpp.NO_STATISTICS_END, ccdcpp.GreenFilterScope.StatWindow))
        # The step's values and the shared statistics_end, in the input's units.
        cfg = CCDC.replace(preprocess__green_median__offset=0.05,
                           preprocess__green_median__scope="all", shared__statistics_end=730000)
        green = ccdcpp.plan_of(cfg, h).insufficient_clear.chain[3]
        self.assertEqual((green.green_offset, green.max_date, green.scope),
                         (0.05 * 10000.0, 730000, ccdcpp.GreenFilterScope.All))
        self.assertEqual(plan.snow.chain[-2].codes, [h.qa_snow])

    def test_physical_plan(self):
        phys = presets.landsat_c1_physical()
        plan = ccdcpp.plan_of("ccdc", phys)
        self.assertIsNone(plan.standard.transform)
        _, therm, band, _ = plan.standard.chain
        self.assertEqual((therm.min, therm.max), (-0.932, 0.707))
        self.assertEqual((band.min, band.max), (0.0, 1.0))
        self.assertEqual(plan.insufficient_clear.chain[3].green_offset, 0.04)
        self.assertEqual(plan, ccdcpp.plan_of(CCDC, phys.input()))

    def test_without_thermal_and_identity_transform(self):
        h = ccdcpp.Input(roles=ccdcpp.Input().roles[:6])
        plan = ccdcpp.plan_of("ccdc", h)
        self.assertNotIn(ccdcpp.StepKind.ThermalRange, [s.kind for s in plan.standard.chain])
        self.assertIsNone(plan.standard.transform)
        h = ccdcpp.Input(thermal_scale=1.0, thermal_offset=0.0)
        self.assertIsNone(ccdcpp.plan_of("ccdc", h).standard.transform)

    def test_codes_follow_the_input(self):
        h = ccdcpp.Input(qa_clear=10, qa_water=11, qa_snow=13)
        plan = ccdcpp.plan_of(keeps_snow(), h)
        self.assertEqual(plan.standard.chain[0].codes, [10, 11, 13])
        self.assertEqual(plan.snow.chain[3].codes, [13])
        # Ints are codes as given.
        plan = ccdcpp.plan_of(CCDC.replace(preprocess__snow__add_qa__codes=[7, 3]), h)
        self.assertEqual(plan.snow.chain[3].codes, [3, 7])

    def test_band_range_by_role(self):
        cfg = CCDC.replace(preprocess__band_range__bands=["swir2", "green"])
        band = ccdcpp.plan_of(cfg, ccdcpp.Input()).snow.chain[1]
        self.assertEqual(band.bands, [5, 1])

    def test_equality(self):
        h = ccdcpp.Input()
        plan = ccdcpp.plan_of(tp.PYCCD, h)
        self.assertEqual(plan, ccdcpp.plan_of(tp.PYCCD, h))
        self.assertNotEqual(plan, ccdcpp.plan_of(keeps_snow(tp.PYCCD), h))
        self.assertNotEqual(plan, ccdcpp.plan_of("ccdc", h))   # the green median's date
        chain = plan.standard.chain
        self.assertEqual(len(chain), 4)
        self.assertEqual(chain[-1].kind, ccdcpp.StepKind.Dedupe)
        with self.assertRaises(IndexError):
            chain[4]

    def test_describe(self):
        plan = ccdcpp.plan_of(tp.PYCCD, ccdcpp.Input())
        text = plan.describe()
        self.assertEqual(text, repr(plan))
        for line in ("standard", "  transform  band 6 <- x * 10 - 27315",
                     "  0  KeepQa codes {0, 1}", "  1  ThermalRange band 6, -9320 < x < 7070",
                     "  2  BandRange bands [0, 1, 2, 3, 4, 5], 0 < x < 10000", "  3  Dedupe",
                     "snow", "  transform  none", "  3  AddQa codes {3}", "insufficient_clear",
                     "  3  GreenMedian band 1, drop x >= median(dates <= 736694) + 400, "
                     "scope StatWindow"):
            self.assertIn(line, text.splitlines())
        inclusive = CCDC.replace(preprocess__standard__band_range__min_inclusive=True,
                                 preprocess__standard__band_range__bands=["blue", "green"],
                                 preprocess__standard__band_range__max=1.0)
        step = ccdcpp.plan_of(inclusive, ccdcpp.presets.landsat_c1_physical()).standard.chain[2]
        self.assertEqual(repr(step), "<Step BandRange bands [0, 1], 0 <= x < 1>")

    def test_readme_lists_the_ccdc_plan(self):
        with open(os.path.join(tp.SOURCE_DIR, "README.md"), encoding="utf-8") as f:
            readme = f.read()
        self.assertIn(ccdcpp.plan_of("ccdc", ccdcpp.Input()).describe(), readme)

    def test_profile_and_input_agree(self):
        for preset in (presets.landsat_c1_dn, presets.landsat_c1_physical, presets.hls_s30):
            with self.subTest(preset=preset.__name__):
                self.assertEqual(ccdcpp.plan_of("ccdc", preset()),
                                 ccdcpp.plan_of("ccdc", preset().input()))


class WrittenChains(unittest.TestCase):
    """pyccd's chain written out is "ccdc"'s, configuration and output."""

    def test_written_default_is_the_preset(self):
        for written in (written_default(), written_default_explicit()):
            with self.subTest(written=written):
                cfg = CCDC.replace(preprocess=written)
                for preset in (presets.landsat_c1_dn, presets.landsat_c1_physical):
                    self.assertEqual(ccdcpp.plan_of(cfg, preset()),
                                     ccdcpp.plan_of(CCDC, preset()))
        # Left to their defaults, the steps spell out to the preset's.
        self.assertEqual(CCDC.replace(preprocess=written_default()), CCDC)

    def test_written_default_output(self):
        for written in (written_default(), written_default_explicit()):
            with self.subTest(written=written):
                cfg = tp.PYCCD.replace(preprocess=written)
                assert_same(self, every_entry(DATES, CUBE_DN, CODES, encoding=ccdcpp.Input(),
                                              config=cfg),
                            every_entry(DATES, CUBE_DN, CODES, encoding=ccdcpp.Input(),
                                        config=tp.PYCCD))

    def test_ten_band_default(self):
        profile = tp.ten_band_profile()
        dates, cube, scl = tp.ten_band_inputs()
        common = (KeepQa(), BandRange(), Dedupe())
        written = Preprocess(
            standard=ProcedureChain(steps=common, thermal_transform=True),
            snow=ProcedureChain(steps=common + (AddQa(), Dedupe())),
            insufficient_clear=ProcedureChain(steps=common + (GreenMedian(), Dedupe())),
        )
        cfg = tp.TEN_BAND_CONFIG.replace(preprocess=written)
        self.assertEqual(ccdcpp.plan_of(cfg, profile),
                         ccdcpp.plan_of(tp.TEN_BAND_CONFIG, profile))
        assert_same(self,
                    without(ccdcpp.detect_cube(dates, cube, scl, threads=2, encoding=profile,
                                               config=cfg), "threads"),
                    without(ccdcpp.detect_cube(dates, cube, scl, threads=2, encoding=profile,
                                               config=tp.TEN_BAND_CONFIG), "threads"))


class CustomChains(unittest.TestCase):
    def test_standard_keeps_snow(self):
        dates, ref, qa = tp.reference_pixel()
        h = ccdcpp.Input()
        base = ccdcpp.detect(dates, ref, qa, encoding=h, config=tp.PYCCD)
        snowy = ccdcpp.detect(dates, ref, qa, encoding=h, config=keeps_snow(tp.PYCCD))
        self.assertNotIn(54, [m["curve_qa"] for m in snowy["models"]])
        self.assertEqual(snow_kept(base, qa), 0)
        self.assertGreater(snow_kept(snowy, qa), 0)
        self.assertNotEqual(repr(base), repr(snowy))
        # The same chain on a profile's input.
        assert_same(self, without(ccdcpp.detect(dates, ref, qa, encoding=presets.landsat_c1_dn(),
                                                config=keeps_snow(tp.PYCCD)), "band_names"),
                    snowy)

    def test_cube_entry_point(self):
        h = ccdcpp.Input()
        plain = every_entry(DATES, CUBE_DN, CODES, encoding=h, config=keeps_snow())
        assert_same(self, strip(every_entry(DATES, CUBE_DN, CODES, encoding=presets.landsat_c1_dn(),
                                            config=keeps_snow())), plain)
        self.assertNotEqual(repr(plain), repr(every_entry(DATES, CUBE_DN, CODES, encoding=h)))

    def test_codes_follow_the_run_input(self):
        h = ccdcpp.Input(qa_clear=10, qa_water=11, qa_snow=13)
        renumbered = np.vectorize({255: 255, 0: 10, 1: 11, 2: 2, 3: 13, 4: 4}.get,
                                  otypes=[np.uint8])(CODES)
        assert_same(self,
                    every_entry(DATES, CUBE_DN, renumbered, encoding=h, config=keeps_snow()),
                    every_entry(DATES, CUBE_DN, CODES, encoding=ccdcpp.Input(),
                                config=keeps_snow()))

    def test_transform_on_every_procedure(self):
        # Thermal handed over in Kelvin x 10 and transformed on every procedure
        # is the preset on input converted beforehand with the identity
        # transform.
        everywhere = CCDC.replace(preprocess__snow__thermal_transform=True,
                                  preprocess__insufficient_clear__thermal_transform=True)
        cube = CUBE_DN.copy()
        cube[..., 6, ::7] = 1000.0
        converted = cube.copy()
        converted[..., 6, :] = converted[..., 6, :] * 10.0 - 27315.0
        h = ccdcpp.Input(thermal_scale=1.0, thermal_offset=0.0)
        got = without(ccdcpp.detect_cube(DATES, cube, CODES, threads=2, encoding=ccdcpp.Input(),
                                         config=everywhere), "threads")
        assert_same(self, got, without(ccdcpp.detect_cube(DATES, converted, CODES, encoding=h,
                                                          threads=2), "threads"))
        self.assertNotEqual(
            repr(got), repr(without(ccdcpp.detect_cube(DATES, cube, CODES, threads=2,
                                                       encoding=ccdcpp.Input()), "threads")))


class Replace(unittest.TestCase):
    def test_each_form(self):
        cfg = CCDC.replace(preprocess__insufficient_clear__green_median__offset=0.05)
        self.assertEqual(cfg.preprocess("insufficient_clear")["steps"][3]["offset"], 0.05)
        self.assertEqual(cfg.preprocess("snow"), CCDC.preprocess("snow"))
        every = CCDC.replace(preprocess__band_range__min=-0.5)
        for procedure in ccdcpp.PROCEDURES:
            (band,) = [s for s in every.preprocess(procedure)["steps"] if s["step"] == "band_range"]
            self.assertEqual(band["min"], -0.5)
        flag = CCDC.replace(preprocess__snow__thermal_transform=True)
        self.assertTrue(flag.preprocess("snow")["thermal_transform"])
        chain = CCDC.replace(preprocess__snow=[KeepQa(), Dedupe()])
        self.assertEqual([s["step"] for s in chain.preprocess("snow")["steps"]],
                         ["keep_qa", "dedupe"])
        self.assertFalse(chain.preprocess("snow")["thermal_transform"])
        as_dict = CCDC.replace(preprocess__snow={"steps": [{"step": "dedupe"}]})
        self.assertEqual(as_dict.preprocess("snow")["steps"], [{"step": "dedupe"}])
        whole = CCDC.replace(preprocess=Preprocess(snow=ProcedureChain(steps=(Dedupe(),))))
        self.assertEqual(whole.preprocess("snow"), as_dict.preprocess("snow"))
        self.assertEqual(whole.preprocess("standard"), CCDC.preprocess("standard"))
        # An int given for a real is a real.
        self.assertEqual(
            CCDC.replace(preprocess__band_range__max=1).preprocess("snow")["steps"][1]["max"],
            1.0)

    def test_refusals_in_replace(self):
        with self.assertRaisesRegex(ValueError, "2 'dedupe' steps"):
            CCDC.replace(preprocess__snow__dedupe__x=1)
        with self.assertRaisesRegex(ValueError, "step 'green_median' parameter 'max_date' is "
                                                "the shared value 'statistics_end'"):
            CCDC.replace(preprocess__green_median__max_date=1)
        with self.assertRaisesRegex(ValueError, "has no parameter 'width'"):
            CCDC.replace(preprocess__band_range__width=1)
        with self.assertRaisesRegex(ValueError, "no procedure 'cloudy'"):
            CCDC.replace(preprocess__cloudy=[Dedupe()])
        with self.assertRaisesRegex(ValueError, "no 'green_mean' step"):
            CCDC.replace(preprocess__green_mean__offset=0.1)

    def test_toml_round_trip(self):
        cfg = keeps_snow().replace(preprocess__insufficient_clear__green_median__offset=0.05,
                                   preprocess__snow__add_qa__codes=[3, 7])
        text = cfg.to_toml()
        self.assertIn("[preprocess.standard]", text)
        tomllib.loads(text)
        path = os.path.join(tp.MODULE_DIR, "test_preprocess_round_trip.toml")
        with open(path, "w") as f:
            f.write(text)
        try:
            self.assertEqual(ccdcpp.config(path), cfg)
        finally:
            os.remove(path)
        # A file that extends "ccdc" and gives one procedure.
        path = os.path.join(tp.MODULE_DIR, "test_preprocess_extends.toml")
        with open(path, "w") as f:
            f.write('extends = "ccdc"\n\n[preprocess.snow]\n'
                    'steps = [{step = "keep_qa", codes = ["clear", "water"]}, '
                    '{step = "dedupe"}]\n')
        try:
            got = ccdcpp.config(path)
        finally:
            os.remove(path)
        self.assertEqual(got, CCDC.replace(preprocess__snow=[KeepQa(), Dedupe()]))
        self.assertEqual(got.preprocess("standard"), CCDC.preprocess("standard"))


class Validation(unittest.TestCase):
    def test_refused_when_built(self):
        # Building a Config compiles it (Registry::check()) and refuses it there.
        def without_snow():
            tree = CCDC.to_dict()
            del tree["preprocess"]["snow"]
            return ccdcpp.Config(tree)

        def with_codes():
            tree = CCDC.to_dict()
            tree["preprocess"]["snow"]["steps"][0]["codes"] = ["clear", "cirrus"]
            return ccdcpp.Config(tree)

        def with_max_date():
            tree = CCDC.to_dict()
            tree["preprocess"]["insufficient_clear"]["steps"][3]["max_date"] = 1
            return ccdcpp.Config(tree)

        cases = {
            "not a preprocessing step": lambda: CCDC.replace(preprocess__snow=[{"step": "smooth"}]),
            "green_median.* parameter 'scope' is 'some'":
                lambda: CCDC.replace(preprocess__green_median__scope="some"),
            "at most 8": lambda: CCDC.replace(preprocess__snow=[Dedupe()] * 9),
            "no 'snow' procedure": without_snow,
            "'cirrus', which is not a QA code name": with_codes,
            "green_median.* 'max_date' is the shared value 'statistics_end'": with_max_date,
        }
        for message, build in cases.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    build()
        # What needs the input's band count is refused when the run resolves the
        # configuration against it.
        d, s, q = DATES, CUBE_DN[0, 0], CODES[0, 0]
        h = ccdcpp.Input()
        no_dedupe = CCDC.replace(preprocess__snow=[KeepQa(), AddQa()])
        with self.assertRaisesRegex(ValueError, "must be followed by a Dedupe"):
            ccdcpp.detect(d, s, q, encoding=h, config=no_dedupe)
        # A band the input lacks, against the input.
        with self.assertRaisesRegex(ValueError, "'coastal', which the input does not have"):
            ccdcpp.plan_of(CCDC.replace(preprocess__band_range__bands=["coastal"]), h)

    def test_step_classes(self):
        with self.assertRaisesRegex(ValueError, "unknown QA code name 'cirrus'"):
            KeepQa(("clear", "cirrus"))
        with self.assertRaisesRegex(ValueError, "all names or all ints"):
            KeepQa(("clear", 1))
        with self.assertRaises(ValueError):
            AddQa((300,))
        with self.assertRaises(ValueError):
            ThermalRange(inclusive=(True,))
        with self.assertRaises(TypeError):
            ProcedureChain(steps=(object(),))
        with self.assertRaises(TypeError):
            ProcedureChain(thermal_transform=1)
        with self.assertRaises(TypeError):
            Preprocess(standard=(KeepQa(),))
        self.assertEqual(KeepQa().to_dict(), {"step": "keep_qa"})
        self.assertEqual(ThermalRange(min=-80.0, inclusive=(True, False)).to_dict(),
                         {"step": "thermal_range", "min": -80.0, "min_inclusive": True,
                          "max_inclusive": False})

    def test_steps_are_listed(self):
        # The library's steps; an extension may register more.
        steps = ccdcpp.steps()
        self.assertLessEqual({"keep_qa", "add_qa", "thermal_range", "band_range",
                              "green_median", "dedupe"}, set(steps))
        units = {p["name"]: p["unit"] for p in steps["thermal_range"]}
        self.assertEqual(units["min"], "celsius")
        self.assertEqual({p["name"]: p["unit"] for p in steps["keep_qa"]}["codes"], "qa_codes")
        (max_date,) = [p for p in steps["green_median"] if p["name"] == "max_date"]
        self.assertEqual((max_date["source"], max_date["from"]), ("shared", "statistics_end"))


if __name__ == "__main__":
    unittest.main()
