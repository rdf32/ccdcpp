"""Resume from Python: detect(resume_record=True) and resume() on one pixel,
detect_cube_columns(resume_record=True) and resume_cube_columns() on a cube.

usage: python test_resume.py <module_dir> <source_dir> [unittest args]

The C++ suite (tests/ccd/golden/test_resume.cpp) holds resume to the full
run at thousands of cuts; this one holds the bindings to the C++: the
continuation of a recorded run equals detect() with stat_ord capped at the
cut, array for array, the record a resumed run returns equals a full run's,
and segments stored the way nlcd stores them (no observation count, a
placeholder break day and a NaN magnitude where there is no break) give the
same table once blanked the same way.
"""
import sys
import unittest

MODULE_DIR, SOURCE_DIR = sys.argv[1], sys.argv[2]
del sys.argv[1:3]
sys.path.insert(0, MODULE_DIR)

import numpy as np  # noqa: E402

import ccdcpp  # noqa: E402

START = 724642          # 1985-01-01
CUT_2010 = 733773       # 2010-01-01, before stat_ord: the cap matters
CUT_2019 = 737060       # 2019-01-01, after it


def pixel(seed, step_at=None):
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


def capped(cut):
    """The ccdc preset with its statistics cutoff capped at `cut`."""
    config = ccdcpp._config_preset("ccdc")
    config["shared"]["stat_ord"] = min(config["shared"]["stat_ord"], cut)
    return config


def upto(dates, cut):
    return int(np.searchsorted(dates, cut, side="right"))


def assert_same_models(test, a, b, where):
    test.assertEqual(len(a), len(b), where)
    for i, (x, y) in enumerate(zip(a, b)):
        for key in ("start_day", "end_day", "break_day", "observation_count",
                    "change_probability", "curve_qa"):
            test.assertEqual(x[key], y[key], f"{where} segment {i} {key}")
        for bx, by in zip(x["bands"], y["bands"]):
            test.assertEqual(tuple(bx["coefficients"]), tuple(by["coefficients"]), where)
            for key in ("intercept", "rmse", "magnitude"):
                test.assertEqual(bx[key], by[key], f"{where} segment {i} {key}")


def assert_same_record(test, a, b, where):
    test.assertEqual(sorted(a), sorted(b), where)
    for key in a:
        if isinstance(a[key], np.ndarray):
            np.testing.assert_array_equal(a[key], b[key], err_msg=f"{where} {key}")
        else:
            test.assertEqual(a[key], b[key], f"{where} {key}")


class OnePixel(unittest.TestCase):

    def test_resume_equals_the_capped_full_run(self):
        for seed in range(6):
            dates, spectral, qas = pixel(seed)
            for cut in (CUT_2010, CUT_2019):
                n = upto(dates, cut)
                where = f"seed {seed} cut {cut}"
                stored = ccdcpp.detect(dates[:n], spectral[:, :n], qas[:n],
                                       ccdcpp.Input(), resume_record=True)
                self.assertEqual(stored["resume_header"]["series_end"], dates[n - 1])
                again = ccdcpp.resume(
                    dates, spectral, qas, ccdcpp.Input(),
                    resume=stored["resume"], resume_header=stored["resume_header"],
                    models=stored["models"])
                full = ccdcpp.detect(dates, spectral, qas, ccdcpp.Input(), config=capped(dates[n - 1]),
                                     resume_record=True)

                self.assertEqual(again["status"], "resumed", where)
                assert_same_models(self, again["models"], full["models"], where)
                np.testing.assert_array_equal(again["processing_mask"], full["processing_mask"])
                assert_same_record(self, again["resume"], full["resume"], where)
                self.assertEqual(again["resume_header"]["statistics_end"], dates[n - 1])

    def test_a_changed_observation_is_rerun(self):
        dates, spectral, qas = pixel(3)
        n = upto(dates, CUT_2019)
        stored = ccdcpp.detect(dates[:n], spectral[:, :n], qas[:n],
                               ccdcpp.Input(), resume_record=True)
        changed = spectral.copy()
        changed[3, int(np.flatnonzero(qas[:n] == 0)[50])] += 1.0
        again = ccdcpp.resume(dates, changed, qas, ccdcpp.Input(),
                              resume=stored["resume"], resume_header=stored["resume_header"],
                              models=stored["models"])
        full = ccdcpp.detect(dates, changed, qas, ccdcpp.Input(), config=capped(dates[n - 1]))
        self.assertEqual(again["status"], "rerun_input_changed")
        assert_same_models(self, again["models"], full["models"], "changed")

    def test_refusals(self):
        dates, spectral, qas = pixel(1)
        n = upto(dates, CUT_2019)
        stored = ccdcpp.detect(dates[:n], spectral[:, :n], qas[:n],
                               ccdcpp.Input(), resume_record=True)

        def call(h=None, header=None, until=None, config=None):
            m = dates.size if until is None else until
            return ccdcpp.resume(dates[:m], spectral[:, :m], qas[:m],
                                 h or ccdcpp.Input(),
                                 resume=stored["resume"],
                                 resume_header=header or stored["resume_header"],
                                 models=stored["models"], config=config)

        other = ccdcpp.Input()
        other.qa_occlusion += 1
        with self.assertRaises(ValueError):
            call(h=other)
        stricter = ccdcpp._config_preset("ccdc")
        stricter["shared"]["change_probability"] = 0.995
        with self.assertRaises(ValueError):
            call(config=stricter)
        # The preset, given, is the default the record was made under.
        call(config=ccdcpp._config_preset("ccdc"))
        with self.assertRaises(ValueError):
            call(header=dict(stored["resume_header"], results_version=ccdcpp.RESULTS_VERSION + 1))
        with self.assertRaises(ValueError):
            call(header=dict(stored["resume_header"], format=ccdcpp.RESUME_FORMAT + 1))
        with self.assertRaises(ValueError):
            call(until=n - 1)

        # A record that does not fit the series.
        for key, value in (("window_stop", 10 ** 6), ("magnitude", np.zeros(1)),
                           ("final_segments", -1)):
            with self.subTest(key=key):
                bad = dict(stored["resume"], **{key: value})
                if key == "magnitude":
                    bad["exit"] = 6   # SeriesEnd
                with self.assertRaises(ValueError):
                    ccdcpp.resume(dates, spectral, qas, ccdcpp.Input(),
                                  resume=bad, resume_header=stored["resume_header"],
                                  models=stored["models"])


def cube(seeds, rows, cols):
    pixels = [pixel(s) for s in seeds]
    dates = pixels[0][0]
    spectral = np.stack([p[1] for p in pixels]).reshape(rows, cols, 7, dates.size)
    qas = np.stack([p[2] for p in pixels]).reshape(rows, cols, dates.size)
    return dates, np.ascontiguousarray(spectral), np.ascontiguousarray(qas)


SEGMENT_COLUMNS = ("row", "col", "start_day", "end_day", "break_day", "curve_qa",
                   "observation_count", "change_probability", "intercept",
                   "coefficients", "rmse", "magnitude")


def nlcd_blanked(columns):
    """The segment columns as nlcd stores them: no observation count, and a
    placeholder break day and NaN magnitudes where there is no break."""
    out = {k: np.array(columns[k], copy=True) for k in SEGMENT_COLUMNS}
    no_break = out["change_probability"] == 0
    out["break_day"][no_break] = 0
    out["magnitude"][no_break] = np.nan
    out["observation_count"][:] = 0
    return out


class Cube(unittest.TestCase):

    def test_resume_cube_columns_equals_the_capped_full_run(self):
        dates, spectral, qas = cube(range(12), 3, 4)
        for cut in (CUT_2010, CUT_2019):
            n = upto(dates, cut)
            for threads in (1, 3):
                where = f"cut {cut} threads {threads}"
                stored = ccdcpp.detect_cube_columns(
                    dates[:n], spectral[..., :n], qas[..., :n], ccdcpp.Input(),
                    threads, resume_record=True)
                self.assertEqual(stored["resume"]["row"].size, 12)
                again = ccdcpp.resume_cube_columns(
                    dates, spectral, qas, ccdcpp.Input(), threads,
                    resume=stored["resume"], resume_header=stored["resume_header"],
                    segments={k: stored[k] for k in SEGMENT_COLUMNS})
                full = ccdcpp.detect_cube_columns(
                    dates, spectral, qas, ccdcpp.Input(), threads, resume_record=True,
                    config=capped(dates[n - 1]))

                for key in SEGMENT_COLUMNS:
                    np.testing.assert_array_equal(again[key], full[key], err_msg=f"{where} {key}")
                for key in full["resume"]:
                    np.testing.assert_array_equal(
                        again["resume"][key], full["resume"][key], err_msg=f"{where} {key}")
                np.testing.assert_array_equal(again["resume"]["status"], 0)
                self.assertEqual(again["counts"],
                                 {"resumed": 12, "rerun_input_changed": 0, "failed": 0})

    def test_segments_stored_the_way_nlcd_stores_them(self):
        dates, spectral, qas = cube(range(20, 32), 3, 4)
        n = upto(dates, CUT_2019)
        stored = ccdcpp.detect_cube_columns(
            dates[:n], spectral[..., :n], qas[..., :n], ccdcpp.Input(),
            2, resume_record=True)
        again = ccdcpp.resume_cube_columns(
            dates, spectral, qas, ccdcpp.Input(), 2,
            resume=stored["resume"], resume_header=stored["resume_header"],
            segments=nlcd_blanked(stored))
        full = ccdcpp.detect_cube_columns(dates, spectral, qas, ccdcpp.Input(), 2,
                                          config=capped(dates[n - 1]))
        a, b = nlcd_blanked(again), nlcd_blanked(full)
        for key in SEGMENT_COLUMNS:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)

    def test_a_changed_pixel_is_counted(self):
        dates, spectral, qas = cube(range(40, 46), 2, 3)
        n = upto(dates, CUT_2019)
        stored = ccdcpp.detect_cube_columns(
            dates[:n], spectral[..., :n], qas[..., :n], ccdcpp.Input(),
            1, resume_record=True)
        changed = spectral.copy()
        changed[1, 2, 2, int(np.flatnonzero(qas[1, 2, :n] == 0)[30])] += 5.0
        again = ccdcpp.resume_cube_columns(
            dates, changed, qas, ccdcpp.Input(), 1,
            resume=stored["resume"], resume_header=stored["resume_header"],
            segments={k: stored[k] for k in SEGMENT_COLUMNS})
        self.assertEqual(again["counts"], {"resumed": 5, "rerun_input_changed": 1, "failed": 0})
        self.assertEqual(int(again["resume"]["status"][1 * 3 + 2]), 1)

    def test_columns_that_do_not_fit_are_refused(self):
        dates, spectral, qas = cube(range(50, 56), 2, 3)
        n = upto(dates, CUT_2019)
        stored = ccdcpp.detect_cube_columns(
            dates[:n], spectral[..., :n], qas[..., :n], ccdcpp.Input(),
            1, resume_record=True)
        segments = {k: stored[k] for k in SEGMENT_COLUMNS}

        def call(resume=None, segs=None):
            return ccdcpp.resume_cube_columns(
                dates, spectral, qas, ccdcpp.Input(), 1,
                resume=resume or stored["resume"], resume_header=stored["resume_header"],
                segments=segs or segments)

        short = dict(segments, start_day=segments["start_day"][:1])
        with self.assertRaises(ValueError):
            call(segs=short)
        for key in ("window_stop", "magnitude", "excluded_offsets"):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    call(resume=dict(stored["resume"], **{key: stored["resume"][key][:-1]}))
        reordered = {k: v[[0, 3, 1, 4, 2, 5]] if v.shape[0] == 6 else v
                     for k, v in stored["resume"].items()}
        with self.assertRaises(ValueError):
            call(resume=reordered)
        offsets = stored["resume"]["excluded_offsets"].copy()
        offsets[1] = offsets[-1] + 1        # past excluded_days, then decreasing
        bad_offsets = dict(stored["resume"], excluded_offsets=offsets)
        with self.assertRaises(ValueError):
            call(resume=bad_offsets)

        # A window past the series is that pixel's failure, not the call's.
        far = dict(stored["resume"])
        far["window_stop"] = far["window_stop"].copy()
        far["window_start"] = far["window_start"].copy()
        standard = np.flatnonzero(far["procedure"] == 0)
        far["window_start"][standard[0]] = 100000
        far["window_stop"][standard[0]] = 100012
        out = call(resume=far)
        self.assertEqual(len(out["failures"]), 1)
        self.assertEqual(out["counts"]["failed"], 1)
        self.assertEqual(int(out["resume"]["status"][standard[0]]), 2)

    def test_the_schema_names_the_columns(self):
        dates, spectral, qas = cube(range(2), 1, 2)
        out = ccdcpp.detect_cube_columns(dates, spectral, qas, ccdcpp.Input(),
                                         1, resume_record=True)
        schema = ccdcpp.resume_schema()
        self.assertEqual(set(out["resume"]), set(schema) - {"status"})
        for key, array in out["resume"].items():
            self.assertEqual(array.dtype, np.dtype(schema[key]), key)


if __name__ == "__main__":
    unittest.main()
