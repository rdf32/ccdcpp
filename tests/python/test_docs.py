"""What the docstrings state: the presets' inputs, QA and validation records, the QA
tables' products, and that the presets' prose is wrapped and filled.

usage: python test_docs.py <module_dir> <source_dir> [unittest args]

<module_dir> holds the built extension; <source_dir> is the repository root.
The package is assembled as the wheel lays it out (_harness.use_package()).
ctest runs this as `python_docs`.
"""
from __future__ import annotations

import unittest

from _harness import use_package

use_package()

import ccdcpp  # noqa: E402
from ccdcpp import presets  # noqa: E402

from _profiles import BAND_NAMES, BUILDERS, HLS_AND_SENTINEL2, doc_of  # noqa: E402


class PresetDocstrings(unittest.TestCase):
    def test_each_states_its_input_qa_and_bands(self):
        for preset in HLS_AND_SENTINEL2:
            with self.subTest(preset=preset.__name__):
                text = doc_of(preset)
                self.assertNotIn("not validated", text.lower())
                self.assertIn("Input:", text)
                self.assertIn("QA:", text)
                self.assertIn(BUILDERS[preset.__name__], text)
                self.assertEqual(preset().name, preset.__name__)
                for band in BAND_NAMES[preset.__name__]:
                    self.assertIn(band, text)
        self.assertNotIn("not validated", doc_of(presets).lower())
        self.assertIn("BOA_ADD_OFFSET", doc_of(presets.sentinel2_l2a))
        self.assertIn("put L30 before S30", doc_of(presets.hls))
        # S30 detects on bands HLS adjusts to OLI, but B01 and B02 are
        # adjusted too, and Sentinel-2C is part of the cadence.
        self.assertIn("HLS adjusts B03, B04, B8A, B11 and B12, and B01 and B02, to the OLI\n"
                      "    bandpasses", doc_of(presets.hls_s30))
        for preset in (presets.hls_s30, presets.sentinel2_l2a):
            with self.subTest(preset=preset.__name__):
                self.assertIn("2C from 2024", doc_of(preset))

    def test_each_states_its_validation(self):
        def flat(text: str | None) -> str:
            assert text is not None
            return " ".join(text.split())
        # The record every one was validated on.
        for preset in HLS_AND_SENTINEL2:
            with self.subTest(preset=preset.__name__):
                doc = flat(preset.__doc__)
                self.assertIn("Validation: one cell of MGRS tile T16SGC near Atlanta", doc)
                self.assertIn("2020-2023, read from Planetary Computer", doc)
        self.assertIn("2020-2023, read from Planetary Computer, four years of one cell",
                      flat(presets.__doc__))
        # hls_l30 is promoted outright; the summary line carries no caveat.
        self.assertEqual(doc_of(presets.hls_l30).splitlines()[0],
                         "HLS v2.0 L30 (Landsat 8/9 OLI/TIRS at 30 m).")
        self.assertIn("kappa 0.44", flat(presets.hls_l30.__doc__))
        # S30 and hls: the Fmask-over-bright-surfaces caveat, with numbers.
        for preset in (presets.hls_s30, presets.hls):
            with self.subTest(preset=preset.__name__):
                doc = flat(preset.__doc__)
                self.assertIn("Validated, with a caveat: Fmask over bright surfaces.",
                              flat(doc.split("\n\n")[0]))
                self.assertIn("Caveat, Fmask over bright surfaces", doc)
        s30 = flat(presets.hls_s30.__doc__)
        self.assertIn("Over all 1,862 quarry (land cover 31) pixels on the HLS grid it flags "
                      "63% of S30 observations as cloud, and 1,145 of those pixels go to the "
                      "InsufficientClear procedure (68% cloud over them)", s30)
        self.assertIn("170 of Landsat ARD's 173 such pixels, and both of L30's 2", s30)
        self.assertNotIn("61%", s30)
        self.assertIn("85 same-date pairs (84 dates", flat(presets.hls.__doc__))
        self.assertIn("apart from 24 water pixels (0.60% of water pixels against a 0.08% "
                      "baseline)", flat(presets.sentinel2_l2a.__doc__))
        self.assertIn("1,741 of 300,288 pixels go to the InsufficientClear",
                      flat(presets.hls.__doc__))
        # Sentinel-2: median level, PC's processing, not ESA's.
        label = ("Validated at median level on Planetary Computer's Esri/MSFT/S2RP L2A, "
                 "2020–23; ESA's own processing not sampled.")
        s2 = flat(presets.sentinel2_l2a.__doc__)
        self.assertTrue(s2.startswith("Sentinel-2 L2A (ESA Sen2Cor), native bands. " + label), s2)
        self.assertIn(label, flat(presets.__doc__).replace(":func:`sentinel2_l2a` is v", "V"))
        # SCL 7 stays cloud; the water trade-off with the measured counts.
        self.assertIn("7-10 cloud (7, unclassified, included)", s2)
        self.assertIn("sends 1,355 of 3,983 water pixels, and 3,426 pixels in all", s2)
        self.assertIn("decoding 7 as fill would send 2,545, as clear 587. It stays cloud.", s2)
        # The measured hls cadence, not the 2-4 days once documented.
        hls_doc = flat(presets.hls.__doc__)
        self.assertIn("median gap of 4-5 days, and the peek window grew to 19-24", hls_doc)
        self.assertNotIn("2-4 days", hls_doc)
        self.assertNotIn("32-48", hls_doc)

    def test_each_is_wrapped_and_filled(self):
        for obj in [presets] + [getattr(presets, name) for name in presets.__all__]:
            with self.subTest(name=obj.__name__):
                lines = doc_of(obj).splitlines()
                for line in lines:
                    self.assertLessEqual(len(line), 79, line)
                # Prose is filled: a line is not broken where the next word,
                # or the next ``literal`` whole, fits with room to spare.
                for line, following in zip(lines, lines[1:]):
                    indent = len(line) - len(line.lstrip())
                    if (not line.strip() or not following.strip() or indent > 4
                            or len(following) - len(following.lstrip()) != indent
                            or line.endswith(":")):
                        continue
                    word = following.split()[0]
                    if word.startswith("``"):
                        rest = following.strip()
                        end = rest.find("``", 2)
                        word = rest if end < 0 else rest[:end + 2]
                    self.assertGreater(len(line) + 1 + len(word), 70, (line, following))


class QaTableDocstrings(unittest.TestCase):
    def test_they_state_what_each_product_defines(self):
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


if __name__ == "__main__":
    unittest.main()
