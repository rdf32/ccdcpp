"""make_procedure_reference.py --procedure {clear,snow,standard} [--out DIR]

Writes pyccd's reference for one procedure on pyccd's reference pixel
(test_3657_3610_observations.csv, beside this script): <procedure>_reference.csv,
one row per segment (its days, count, change probability and curve_qa, then per band
its rmse, intercept, magnitude and coefficients), and <procedure>_mask_reference.csv,
the processing mask. test_pyccd_reference.cpp compares ccdcpp with them. Needs pyccd
(lcmap-pyccd 2021.7.19) and its dependencies; DIR is this script's directory by default.
"""
import argparse
import os
from typing import Any

import numpy as np
import pandas as pd

from ccd.procedures import (  # pyright: ignore[reportMissingImports]  # pyccd, the reference
    insufficient_clear_procedure,
    permanent_snow_procedure,
    standard_procedure,
)

from ccd import (  # pyright: ignore[reportMissingImports]  # pyccd, the reference
    app,  # pyright: ignore[reportAttributeAccessIssue]
    attr_from_str,  # pyright: ignore[reportAttributeAccessIssue]
    __check_inputs,  # pyright: ignore[reportAttributeAccessIssue]
    __sort_dates  # pyright: ignore[reportAttributeAccessIssue]
)

HERE = os.path.dirname(os.path.abspath(__file__))

PROCEDURES: dict[str, Any] = {
    "clear": insufficient_clear_procedure,
    "snow": permanent_snow_procedure,
    "standard": standard_procedure,
}

PARAMS = {
    'QA_BITPACKED': False,
    'QA_FILL': 255,
    'QA_CLEAR': 0,
    'QA_WATER': 1,
    'QA_SHADOW': 2,
    'QA_SNOW': 3,
    'QA_CLOUD': 4
}

BAND_NAMES = ["blue", "green", "red", "nir", "swir1", "swir2", "thermal"]


def read_data(path: str) -> np.ndarray:
    """The pixel file's columns as rows: day number, the seven bands, QA."""
    return np.genfromtxt(path, delimiter=',', dtype=np.int64).T


def run(procedure: str) -> tuple[Any, np.ndarray]:
    """pyccd's segments and processing mask of `procedure` on the reference pixel."""
    data = read_data(os.path.join(HERE, 'test_3657_3610_observations.csv'))
    dates, blues, greens, reds, nirs, swir1s, swir2s, thermals, qas = data

    proc_params = app.get_default_params()
    proc_params.update(PARAMS)

    dates = np.asarray(dates)
    qas = np.asarray(qas)
    spectra = np.stack((blues, greens, reds, nirs, swir1s, swir2s, thermals))

    fitter_fn = attr_from_str(proc_params.FITTER_FN)

    __check_inputs(dates, qas, spectra)

    # Sort chronologically
    indices = __sort_dates(dates)
    dates = dates[indices]
    spectra = spectra[:, indices]
    qas = qas[indices]

    prev_results = None
    return PROCEDURES[procedure](dates, spectra, fitter_fn, qas, prev_results, proc_params)


def write(procedure: str, results: Any, mask: np.ndarray, out: str) -> None:
    """The segments and the mask as <procedure>_reference.csv and
    <procedure>_mask_reference.csv in `out`."""
    rows: list[dict[str, Any]] = []
    for model in results:
        row: dict[str, Any] = {
            "start_day": model['start_day'],
            "end_day": model['end_day'],
            "break_day": model['break_day'],
            "observation_count": model['observation_count'],
            "change_probability": model['change_probability'],
            "curve_qa": model['curve_qa'],
        }
        for name in BAND_NAMES:
            row[f"{name}_rmse"] = model[name]['rmse']
            row[f"{name}_intercept"] = model[name]['intercept']
            row[f"{name}_magnitude"] = model[name]['magnitude']
            for i, coef in enumerate(model[name]['coefficients']):
                row[f"{name}_coef_{i}"] = coef
        rows.append(row)

    pd.DataFrame(rows).to_csv(os.path.join(out, f"{procedure}_reference.csv"), index=False)
    pd.DataFrame({"mask": mask.astype(np.int8)}).to_csv(
        os.path.join(out, f"{procedure}_mask_reference.csv"), index=False)

    print(f"Saved {procedure}_reference.csv")
    print(f"Saved {procedure}_mask_reference.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="pyccd's reference for one procedure on its reference pixel.")
    parser.add_argument("--procedure", required=True, choices=sorted(PROCEDURES))
    parser.add_argument("--out", default=HERE)
    args = parser.parse_args()
    results, mask = run(args.procedure)
    write(args.procedure, results, mask, args.out)


if __name__ == "__main__":
    main()
