"""patch_pyccd.py <pyccd ccd dir> <out dir>: a copy of PyCCD with ccdcpp's corrections.

Copies the `ccd` package of lcmap-pyccd 2021.7.19 to <out dir>/ccd and applies the corrections
README.md lists under "Differences from pyccd" (all but the coefficient counts, the same at the
defaults), each changed line marked
"# ccdcpp intent patch", so that bench_pyccd.py can time PyCCD and ccdcpp on the same algorithm:
on the synthetic corpus the patched copy and ccdcpp's defaults give the same segments on every
pixel. Fails if any patch does not apply exactly once.
"""
import os
import shutil
import sys

src, out = sys.argv[1], sys.argv[2]
dst = os.path.join(out, "ccd")
shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))

PATCHES = [
    # RLM convergence: the absolute value of the difference, not of the comparison.
    ("models/robust_fit.py",
     "return not numpy.any(numpy.fabs(x0 - x > tol))",
     "return not numpy.any(numpy.fabs(x0 - x) > tol)  # ccdcpp intent patch"),
    # A stable date sort: the first observation of a date survives the duplicate filter.
    ("__init__.py",
     "return np.argsort(dates)",
     'return np.argsort(dates, kind="stable")  # ccdcpp intent patch'),
    # find_closest_doy: ties broken in window order.
    ("change.py",
     "return np.argsort(d_yr)[:num]",
     'return np.argsort(d_yr, kind="stable")[:num]  # ccdcpp intent patch'),
    # The green median filter written through to the mask, not to a copy.
    ("qa.py",
     "standard_mask[standard_mask][:green_mask.shape[0]] &= green_mask",
     "standard_mask[np.flatnonzero(standard_mask)[:green_mask.shape[0]]] &= green_mask"
     "  # ccdcpp intent patch"),
    # Lookback: the full peek far from the break, and never below the previous break near it.
    ("procedures.py",
     """        if model_window.start - previous_break > peek_size:
            peek_window = slice(model_window.start - 1, model_window.start - peek_size, -1)
        elif model_window.start - peek_size <= 0:
            peek_window = slice(model_window.start - 1, None, -1)
        else:
            peek_window = slice(model_window.start - 1, previous_break - 1, -1)""",
     """        # ccdcpp intent patch: the peek far from the break reads peek_size
        # observations, and near it reads down to the previous break, never below.
        if model_window.start - previous_break > peek_size:
            peek_window = slice(model_window.start - 1, model_window.start - peek_size - 1, -1)
        else:
            peek_window = slice(model_window.start - 1, previous_break - 1 if previous_break > 0 else None, -1)"""),
    # Refit at 24: <= 24 for the refit as for the RMSE comparison.
    ("procedures.py",
     "        if not models or model_window.stop - model_window.start < 24 or model_span >= 1.33 * fit_span:",
     "        if not models or model_window.stop - model_window.start <= 24 or model_span >= 1.33 * fit_span:"
     "  # ccdcpp intent patch"),
    # The series end: stop before the grow at the last full peek.
    ("procedures.py",
     """        if model_window.stop + peek_size > period.shape[0]:
            break

        model_window = slice(model_window.start, model_window.stop + 1)""",
     """        if model_window.stop + peek_size >= period.shape[0]:  # ccdcpp intent patch
            break

        model_window = slice(model_window.start, model_window.stop + 1)"""),
    # A curve that reached the series end is the last: no restart after it.
    ("procedures.py",
     """        # Step 5: Iterate
        previous_end = model_window.stop
        model_window = slice(model_window.stop, model_window.stop + meow_size)""",
     """        # Step 5: Iterate
        previous_end = model_window.stop
        model_window = slice(model_window.stop, model_window.stop + meow_size)
        # ccdcpp intent patch: a curve that reached the series end is the last.
        if result['change_probability'] == 0:
            break"""),
]

for name, old, new in PATCHES:
    path = os.path.join(dst, name)
    with open(path) as f:
        text = f.read()
    if text.count(old) != 1:
        sys.exit(f"{name}: the patch's text occurs {text.count(old)} times, not once")
    with open(path, "w") as f:
        f.write(text.replace(old, new))
print(f"patched {len(PATCHES)} places into {dst}")
