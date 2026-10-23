"""Resume from Python: detect(record=True) and detect(resume=) on one pixel,
detect_cube(record=True) and detect_cube(resume=) on a cube.

usage: python test_resume.py <module_dir> <source_dir> [unittest args]

The C++ suite (tests/ccd/golden/test_resume.cpp) holds resume to the full
run at thousands of cuts; this one holds the bindings to the C++: the
continuation of a recorded run equals detect() with statistics_end capped at the
cut, array for array, the record a resumed run returns equals a full run's,
and segments stored the way nlcd stores them (no observation count, a
placeholder break day and a NaN magnitude where there is no break) give the
same table once blanked the same way. Under the preset's cutoff, none, a resume
is refused unless freeze_statistics=True; the tests that compare with the
capped run pass it.
"""
from __future__ import annotations

import re
import unittest
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from _harness import use_package

use_package()

import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

# The extension module, which ccdcpp/ccdcpp.pyi types.
from ccdcpp import ccdcpp as ext  # noqa: E402
from ccdcpp.results import (CubeColumns, CubeRecord, Model, PixelRecord,  # noqa: E402
                            PixelResult, ResumeCounts, ResumeHeader)

START = 724642          # 1985-01-01
CUT_2010 = 733773       # 2010-01-01
CUT_2019 = 737060       # 2019-01-01


Series = tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.uint8]]


def pixel(seed: int, step_at: int | None = None) -> Series:
    """(dates, spectral (7, T), qas) of a 16-day Landsat-like series, 1985-2020:
    a seasonal cycle, noise, cloudy dates, and a step change."""
    rng = np.random.default_rng(seed)
    dates = START + 16 * np.arange(820, dtype=np.int64)
    t = (dates - START) / 365.25
    base = np.array([500, 800, 700, 2500, 1800, 1000, 2950], dtype=np.float64)
    amp = np.array([80, 120, 140, 600, 300, 200, 80], dtype=np.float64)
    spectral = base[:, None] + amp[:, None] * np.cos(2 * np.pi * t + rng.uniform(0, 6.28))
    spectral += rng.normal(0, 25, spectral.shape)
    at = step_at if step_at is not None else 410 + int(rng.integers(-100, 100))
    spectral[:6, at:] += rng.uniform(-400, 900, (6, 1))
    qas = np.where(rng.uniform(size=dates.size) < 0.3, 4, 0).astype(np.uint8)
    return dates, np.round(spectral), qas


def capped(cut: int) -> ext.Algorithm:
    """The ccdc preset with its statistics cutoff capped at `cut`."""
    config = ext._config_preset("ccdc")
    end = config["shared"]["statistics_end"]
    config["shared"]["statistics_end"] = cut if end == "none" else min(end, cut)
    return ext.Algorithm(config)


def upto(dates: NDArray[np.int64], cut: int) -> int:
    return int(np.searchsorted(dates, cut, side="right"))


# What a call made with record=True or resume= also returns.

def header_of(out: PixelResult | CubeColumns) -> ResumeHeader:
    assert "resume_header" in out
    return out["resume_header"]


def record_of(out: PixelResult) -> PixelRecord:
    assert "resume" in out
    return out["resume"]


def records_of(out: CubeColumns) -> CubeRecord:
    assert "resume" in out
    return out["resume"]


def status_of(out: PixelResult) -> str:
    """With resume=."""
    assert "status" in out
    return out["status"]


def statuses_of(out: CubeColumns) -> NDArray[np.uint8]:
    """With resume=: each pixel's status."""
    records = records_of(out)
    assert "status" in records
    return records["status"]


def counts_of(out: CubeColumns) -> ResumeCounts:
    """With resume=."""
    assert "counts" in out
    return out["counts"]


def columns(records: CubeRecord) -> dict[str, NDArray[Any]]:
    """The record's columns by name, for editing."""
    return {key: np.asarray(value) for key, value in records.items()}


def assert_same_models(test: unittest.TestCase, a: Sequence[Model], b: Sequence[Model],
                       where: str) -> None:
    test.assertEqual(len(a), len(b), where)
    for i, (x, y) in enumerate(zip(a, b)):
        for key in ("start_day", "end_day", "break_day", "observation_count",
                    "change_probability", "curve_qa"):
            test.assertEqual(x[key], y[key], f"{where} segment {i} {key}")
        for bx, by in zip(x["bands"], y["bands"]):
            test.assertEqual(tuple(bx["coefficients"]), tuple(by["coefficients"]), where)
            for key in ("intercept", "rmse", "magnitude"):
                test.assertEqual(bx[key], by[key], f"{where} segment {i} {key}")


def assert_same_record(test: unittest.TestCase, a: Mapping[str, object],
                       b: Mapping[str, object], where: str) -> None:
    test.assertEqual(sorted(a), sorted(b), where)
    for key, value in a.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, np.asarray(b[key]), err_msg=f"{where} {key}")
        else:
            test.assertEqual(value, b[key], f"{where} {key}")


class OnePixel(unittest.TestCase):

    def test_resume_equals_the_capped_full_run(self):
        for seed in range(6):
            dates, spectral, qas = pixel(seed)
            for cut in (CUT_2010, CUT_2019):
                n = upto(dates, cut)
                where = f"seed {seed} cut {cut}"
                stored = ext.detect(dates[:n], spectral[:, :n], qas[:n],
                                    spec=ext.DataSpec(), record=True)
                self.assertEqual(header_of(stored)["series_end"], dates[n - 1])
                again = ext.detect(dates, spectral, qas, spec=ext.DataSpec(), resume=stored,
                                   freeze_statistics=True)
                full = ext.detect(dates, spectral, qas, spec=ext.DataSpec(),
                                  config=capped(dates[n - 1]), record=True)

                self.assertEqual(status_of(again), "resumed", where)
                assert_same_models(self, again["models"], full["models"], where)
                np.testing.assert_array_equal(again["processing_mask"], full["processing_mask"])
                assert_same_record(self, record_of(again), record_of(full), where)
                self.assertEqual(header_of(again)["statistics_end"], dates[n - 1])

    def test_a_changed_observation_is_rerun(self):
        dates, spectral, qas = pixel(3)
        n = upto(dates, CUT_2019)
        stored = ext.detect(dates[:n], spectral[:, :n], qas[:n],
                            spec=ext.DataSpec(), record=True)
        changed = spectral.copy()
        changed[3, int(np.flatnonzero(qas[:n] == 0)[50])] += 1.0
        again = ext.detect(dates, changed, qas, spec=ext.DataSpec(), resume=stored,
                           freeze_statistics=True)
        full = ext.detect(dates, changed, qas, spec=ext.DataSpec(),
                          config=capped(dates[n - 1]))
        self.assertEqual(status_of(again), "rerun_input_changed")
        assert_same_models(self, again["models"], full["models"], "changed")

    def test_refusals(self):
        dates, spectral, qas = pixel(1)
        n = upto(dates, CUT_2019)
        stored = ext.detect(dates[:n], spectral[:, :n], qas[:n],
                            spec=ext.DataSpec(), record=True)

        def call(h: ext.DataSpec | None = None, header: Mapping[str, Any] | None = None,
                 until: int | None = None,
                 config: ext.Algorithm | None = None) -> PixelResult:
            m = dates.size if until is None else until
            return ext.detect(dates[:m], spectral[:, :m], qas[:m],
                              spec=h or ext.DataSpec(), config=config,
                              resume=dict(stored,
                                          resume_header=header or header_of(stored)),
                              freeze_statistics=True)

        other = ext.DataSpec(qa_occlusion=ext.DataSpec().qa_occlusion + 1)
        with self.assertRaisesRegex(ValueError, "another data spec or configuration"):
            call(h=other)
        decoding = ext.DataSpec(qa_table=ext.qa_tables.landsat_c1_pixel_qa())
        with self.assertRaisesRegex(ValueError, "another data spec or configuration"):
            call(h=decoding)
        stricter = ext._config_preset("ccdc")
        stricter["shared"]["change_probability"] = 0.995
        with self.assertRaises(ValueError):
            call(config=ext.Algorithm(stricter))
        # The preset, given, is the default the record was made under.
        call(config=ext.Algorithm(ext._config_preset("ccdc")))
        with self.assertRaises(ValueError):
            call(header=dict(header_of(stored), results_version=ext.RESULTS_VERSION + 1))
        with self.assertRaises(ValueError):
            call(header=dict(header_of(stored), format=ext.RESUME_FORMAT + 1))
        with self.assertRaises(ValueError):
            call(until=n - 1)

        # A record that does not fit the series.
        for key, value in (("window_stop", 10 ** 6), ("magnitude", np.zeros(1)),
                           ("final_segments", -1)):
            with self.subTest(key=key):
                bad: dict[str, object] = dict(record_of(stored), **{key: value})
                if key == "magnitude":
                    bad["exit"] = 6   # SeriesEnd
                with self.assertRaises(ValueError):
                    ext.detect(dates, spectral, qas, spec=ext.DataSpec(),
                               resume=dict(stored, resume=bad), freeze_statistics=True)

        # An output without a record.
        plain = ext.detect(dates[:n], spectral[:, :n], qas[:n], spec=ext.DataSpec())
        with self.assertRaisesRegex(ValueError, "record=True"):
            ext.detect(dates, spectral, qas, spec=ext.DataSpec(), resume=plain)


def refusal(stored_end: int, config_end: int | None = None) -> str:
    """The message of a resume that would cap a statistics cutoff: the configuration's
    (None: "none"), after the stored run's end."""
    statistics = ("cover the whole series" if config_end is None
                  else f"run to {date.fromordinal(config_end).isoformat()}")
    return (f"this configuration's statistics {statistics}; a resume keeps them at the "
            f"stored run's end ({date.fromordinal(stored_end).isoformat()}). Pass "
            "freeze_statistics=True to resume that way, or rerun in full to use the new "
            "dates in the statistics")


class FreezeStatistics(unittest.TestCase):

    def test_a_pixel_resume_under_no_cutoff_is_refused(self):
        dates, spectral, qas = pixel(2)
        n = upto(dates, CUT_2019)
        stored = ext.detect(dates[:n], spectral[:, :n], qas[:n], spec=ext.DataSpec(), record=True)
        with self.assertRaisesRegex(ValueError, f"^{re.escape(refusal(int(dates[n - 1])))}$"):
            ext.detect(dates, spectral, qas, spec=ext.DataSpec(), resume=stored)
        with self.assertRaisesRegex(ValueError, "cover the whole series"):
            ext.detect(dates, spectral, qas, spec=ext.DataSpec(), resume=stored,
                       freeze_statistics=False)

    def test_a_later_cutoff_is_refused_by_its_date(self):
        dates, spectral, qas = pixel(2)
        n = upto(dates, CUT_2010)
        later = ext._config_preset("ccdc")
        later["shared"]["statistics_end"] = CUT_2019
        config = ext.Algorithm(later)
        stored = ext.detect(dates[:n], spectral[:, :n], qas[:n], spec=ext.DataSpec(),
                            config=config, record=True)
        message = refusal(int(dates[n - 1]), CUT_2019)
        self.assertIn("run to 2019-01-01", message)
        with self.assertRaisesRegex(ValueError, f"^{re.escape(message)}$"):
            ext.detect(dates, spectral, qas, spec=ext.DataSpec(), config=config, resume=stored)
        again = ext.detect(dates, spectral, qas, spec=ext.DataSpec(), config=config,
                           resume=stored, freeze_statistics=True)
        full = ext.detect(dates, spectral, qas, spec=ext.DataSpec(), config=capped(dates[n - 1]))
        assert_same_models(self, again["models"], full["models"], "frozen")

    def test_a_cutoff_at_or_before_the_stored_end_needs_no_flag(self):
        pyccd = ext._config_preset("ccdc")
        pyccd["shared"]["statistics_end"] = 736694      # 2017-12-31
        config = ext.Algorithm(pyccd)
        for seed in range(3):
            dates, spectral, qas = pixel(seed)
            n = upto(dates, CUT_2019)
            where = f"seed {seed}"
            stored = ext.detect(dates[:n], spectral[:, :n], qas[:n], spec=ext.DataSpec(),
                                config=config, record=True)
            again = ext.detect(dates, spectral, qas, spec=ext.DataSpec(), config=config,
                               resume=stored)
            frozen = ext.detect(dates, spectral, qas, spec=ext.DataSpec(), config=config,
                                resume=stored, freeze_statistics=True)
            full = ext.detect(dates, spectral, qas, spec=ext.DataSpec(), config=config,
                              record=True)
            assert_same_models(self, again["models"], full["models"], where)
            assert_same_models(self, frozen["models"], full["models"], where)
            np.testing.assert_array_equal(again["processing_mask"], full["processing_mask"])
            assert_same_record(self, record_of(again), record_of(full), where)

    def test_a_cube_resume_under_no_cutoff_is_refused(self):
        dates, spectral, qas = cube(range(60, 66), 2, 3)
        n = upto(dates, CUT_2019)
        stored = ext.detect_cube(dates[:n], spectral[..., :n], qas[..., :n],
                                 spec=ext.DataSpec(), threads=2, record=True)
        with self.assertRaisesRegex(ValueError, f"^{re.escape(refusal(int(dates[n - 1])))}$"):
            ext.detect_cube(dates, spectral, qas, spec=ext.DataSpec(), threads=2, resume=stored)
        again = ext.detect_cube(dates, spectral, qas, spec=ext.DataSpec(), threads=2,
                                resume=stored, freeze_statistics=True)
        self.assertEqual(counts_of(again), {"resumed": 6, "rerun_input_changed": 0, "failed": 0})


def cube(seeds: Sequence[int], rows: int, cols: int) -> Series:
    pixels = [pixel(s) for s in seeds]
    dates = pixels[0][0]
    spectral = np.stack([p[1] for p in pixels]).reshape(rows, cols, 7, dates.size)
    qas = np.stack([p[2] for p in pixels]).reshape(rows, cols, dates.size)
    return dates, np.ascontiguousarray(spectral), np.ascontiguousarray(qas)


SEGMENT_COLUMNS = ("row", "col", "start_day", "end_day", "break_day", "curve_qa",
                   "observation_count", "change_probability", "intercept",
                   "coefficients", "rmse", "magnitude")


def segment_columns(out: CubeColumns) -> dict[str, NDArray[Any]]:
    """The segment columns by name."""
    by_name: Mapping[str, object] = out
    return {k: np.asarray(by_name[k]) for k in SEGMENT_COLUMNS}


def nlcd_blanked(result: CubeColumns) -> dict[str, NDArray[Any]]:
    """The segment columns as nlcd stores them: no observation count, and a
    placeholder break day and NaN magnitudes where there is no break."""
    out = {k: v.copy() for k, v in segment_columns(result).items()}
    no_break = out["change_probability"] == 0
    out["break_day"][no_break] = 0
    out["magnitude"][no_break] = np.nan
    out["observation_count"][:] = 0
    return out


class Cube(unittest.TestCase):

    def test_resume_cube_equals_the_capped_full_run(self):
        dates, spectral, qas = cube(range(12), 3, 4)
        for cut in (CUT_2010, CUT_2019):
            n = upto(dates, cut)
            for threads in (1, 3):
                where = f"cut {cut} threads {threads}"
                stored = ext.detect_cube(
                    dates[:n], spectral[..., :n], qas[..., :n], spec=ext.DataSpec(),
                    threads=threads, record=True)
                self.assertEqual(records_of(stored)["row"].size, 12)
                again = ext.detect_cube(dates, spectral, qas, spec=ext.DataSpec(),
                                        threads=threads, resume=stored,
                                        freeze_statistics=True)
                full = ext.detect_cube(
                    dates, spectral, qas, spec=ext.DataSpec(), threads=threads, record=True,
                    config=capped(dates[n - 1]))

                a, b = segment_columns(again), segment_columns(full)
                for key in SEGMENT_COLUMNS:
                    np.testing.assert_array_equal(a[key], b[key], err_msg=f"{where} {key}")
                resumed, ran = columns(records_of(again)), columns(records_of(full))
                for key in ran:
                    np.testing.assert_array_equal(resumed[key], ran[key], err_msg=f"{where} {key}")
                np.testing.assert_array_equal(statuses_of(again), 0)
                self.assertEqual(counts_of(again),
                                 {"resumed": 12, "rerun_input_changed": 0, "failed": 0})

    def test_segments_stored_the_way_nlcd_stores_them(self):
        dates, spectral, qas = cube(range(20, 32), 3, 4)
        n = upto(dates, CUT_2019)
        stored = ext.detect_cube(
            dates[:n], spectral[..., :n], qas[..., :n], spec=ext.DataSpec(),
            threads=2, record=True)
        again = ext.detect_cube(
            dates, spectral, qas, spec=ext.DataSpec(), threads=2,
            resume={**nlcd_blanked(stored), "resume": records_of(stored),
                    "resume_header": header_of(stored)},
            freeze_statistics=True)
        full = ext.detect_cube(dates, spectral, qas, spec=ext.DataSpec(), threads=2,
                               config=capped(dates[n - 1]))
        a, b = nlcd_blanked(again), nlcd_blanked(full)
        for key in SEGMENT_COLUMNS:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)

    def test_a_changed_pixel_is_counted(self):
        dates, spectral, qas = cube(range(40, 46), 2, 3)
        n = upto(dates, CUT_2019)
        stored = ext.detect_cube(
            dates[:n], spectral[..., :n], qas[..., :n], spec=ext.DataSpec(),
            threads=1, record=True)
        changed = spectral.copy()
        changed[1, 2, 2, int(np.flatnonzero(qas[1, 2, :n] == 0)[30])] += 5.0
        again = ext.detect_cube(dates, changed, qas, spec=ext.DataSpec(), threads=1,
                                resume=stored, freeze_statistics=True)
        self.assertEqual(counts_of(again), {"resumed": 5, "rerun_input_changed": 1, "failed": 0})
        self.assertEqual(int(statuses_of(again)[1 * 3 + 2]), 1)

    def test_columns_that_do_not_fit_are_refused(self):
        dates, spectral, qas = cube(range(50, 56), 2, 3)
        n = upto(dates, CUT_2019)
        stored = ext.detect_cube(
            dates[:n], spectral[..., :n], qas[..., :n], spec=ext.DataSpec(),
            threads=1, record=True)
        segments = segment_columns(stored)
        records = columns(records_of(stored))

        def call(resume: Mapping[str, Any] | None = None,
                 segs: Mapping[str, Any] | None = None) -> CubeColumns:
            return ext.detect_cube(
                dates, spectral, qas, spec=ext.DataSpec(), threads=1,
                resume={**(segs or segments), "resume": resume or records,
                        "resume_header": header_of(stored)},
                freeze_statistics=True)

        short = dict(segments, start_day=segments["start_day"][:1])
        with self.assertRaises(ValueError):
            call(segs=short)
        for key in ("window_stop", "magnitude", "excluded_offsets"):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    call(resume=dict(records, **{key: records[key][:-1]}))
        reordered = {k: v[[0, 3, 1, 4, 2, 5]] if v.shape[0] == 6 else v
                     for k, v in records.items()}
        with self.assertRaises(ValueError):
            call(resume=reordered)
        offsets = records["excluded_offsets"].copy()
        offsets[1] = offsets[-1] + 1        # past excluded_days, then decreasing
        bad_offsets = dict(records, excluded_offsets=offsets)
        with self.assertRaises(ValueError):
            call(resume=bad_offsets)

        # A window past the series is that pixel's failure, not the call's.
        far = dict(records)
        far["window_stop"] = far["window_stop"].copy()
        far["window_start"] = far["window_start"].copy()
        standard = np.flatnonzero(far["procedure"] == 0)
        far["window_start"][standard[0]] = 100000
        far["window_stop"][standard[0]] = 100012
        out = call(resume=far)
        self.assertEqual(len(out["failures"]), 1)
        self.assertEqual(counts_of(out)["failed"], 1)
        self.assertEqual(int(statuses_of(out)[standard[0]]), 2)

    def test_a_0_10_header_is_refused_by_its_format(self):
        # 0.10.0's header: format 1, results version 2, options_digest where 1.0 has digest.
        def old(header: ResumeHeader) -> dict[str, Any]:
            return {"format": 1, "options_digest": header["digest"], "results_version": 2,
                    "series_end": header["series_end"],
                    "statistics_end": header["statistics_end"]}

        message = (r"^the resume record's format is 1; this ccdcpp reads "
                   rf"{ext.RESUME_FORMAT}$")
        self.assertEqual((ext.RESUME_FORMAT, ext.RESULTS_VERSION), (2, 3))
        dates, spectral, qas = pixel(1)
        n = upto(dates, CUT_2019)
        stored = ext.detect(dates[:n], spectral[:, :n], qas[:n],
                            spec=ext.DataSpec(), record=True)
        with self.assertRaisesRegex(ValueError, message):
            ext.detect(dates, spectral, qas, spec=ext.DataSpec(),
                       resume=dict(stored, resume_header=old(header_of(stored))))

        dates, spectral, qas = cube(range(50, 56), 2, 3)
        stored_cube = ext.detect_cube(
            dates[:n], spectral[..., :n], qas[..., :n], spec=ext.DataSpec(),
            threads=1, record=True)
        with self.assertRaisesRegex(ValueError, message):
            ext.detect_cube(dates, spectral, qas, spec=ext.DataSpec(), threads=1,
                            resume=dict(stored_cube,
                                        resume_header=old(header_of(stored_cube))))

    def test_the_schema_names_the_columns(self):
        dates, spectral, qas = cube(range(2), 1, 2)
        out = ext.detect_cube(dates, spectral, qas, spec=ext.DataSpec(),
                              threads=1, record=True)
        schema = ext.resume_schema()
        records = columns(records_of(out))
        self.assertEqual(set(records), set(schema) - {"status"})
        for key, array in records.items():
            self.assertEqual(array.dtype, np.dtype(schema[key]), key)


if __name__ == "__main__":
    unittest.main()
