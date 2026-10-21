"""make_regression_reference.py --model {lasso,robust} [--out DIR]

Writes the reference of one regression on twelve observations of pyccd's reference
pixel: <model>_reference.csv, one row per band (its coefficients, the count of
nonzero ones, the RMSE and each residual; lasso also its intercept and iteration
count). lasso is scikit-learn's Lasso on pyccd's 4-coefficient matrix, robust
pyccd's RLM on its Tmask matrix; test_lasso.cpp and test_robust.cpp compare
ccdcpp's solvers with them. Needs pyccd (lcmap-pyccd 2021.7.19) and scikit-learn;
DIR is this script's directory by default.
"""
import argparse
import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn import linear_model  # pyright: ignore[reportMissingImports]  # scikit-learn, the reference

from ccd.models import robust_fit  # pyright: ignore[reportMissingImports]  # pyccd, the reference
from ccd.models.lasso import coefficient_matrix  # pyright: ignore[reportMissingImports]  # pyccd, the reference
from ccd.models.tmask import tmask_coefficient_matrix  # pyright: ignore[reportMissingImports]  # pyccd, the reference
from ccd.math_utils import calc_rmse  # pyright: ignore[reportMissingImports]  # pyccd, the reference

HERE = os.path.dirname(os.path.abspath(__file__))

# num params / coef: 4
DATES = np.array([
        724387, 724419, 724451, 724483,
        724547, 724739, 724867, 724915,
        724931, 724947, 725075, 725091
    ])

Y = np.array([
    [ 432,  447,  289,  310,  493,  757,  508,  429,  577,  633, 1323,  637],
    [ 514,  602,  484,  549,  731,  954,  793,  652,  803,  968, 1514,  840],
    [ 608,  595,  406,  427,  850, 1131,  853,  745, 1034, 1305, 1662, 1052],
    [ 937, 1891, 2526, 3103, 2355, 1716, 2389, 2168, 2217, 2322, 2240, 1602],
    [1073, 1585, 1547, 1777, 2817, 2838, 2779, 2622, 2974, 3195, 1786, 2740],
    [ 683, 1094,  813,  855, 1752, 2237, 1708, 1595, 1770, 2134, 1157, 1994],
    [1935, 2325, 2665, 2705, 1105,  765, 2415, 1345,  865,  665,  -55,  315]
])


def lasso_rows() -> list[dict[str, Any]]:
    """scikit-learn's Lasso of each band on pyccd's 4-coefficient matrix."""
    X = coefficient_matrix(DATES, 365.2425, 4)
    rows: list[dict[str, Any]] = []
    for band in range(7):
        lasso = linear_model.Lasso(max_iter=1000)
        model = lasso.fit(X, Y[band])

        predictions = model.predict(X)
        rmse, residuals = calc_rmse(Y[band], predictions, num_pm=4)

        row: dict[str, Any] = {
            "band": band,
            "intercept": model.intercept_,
            "coef_0": model.coef_[0],
            "coef_1": model.coef_[1],
            "coef_2": model.coef_[2],
            "coef_3": model.coef_[3],
            "iterations": model.n_iter_,
            "nonzero_coefficients": np.count_nonzero(model.coef_),
            "rmse": rmse,
        }
        # Store each residual as its own column
        for i, r in enumerate(residuals):
            row[f"residual_{i}"] = r
        rows.append(row)

        print(f"Band {band}")
        print(f"  Intercept : {model.intercept_}")
        print(f"  Coefficients : {model.coef_}")
        print(f"  Iterations : {model.n_iter_}")
        print(f"  RMSE : {rmse}")
        print("-" * 50)
    return rows


def robust_rows() -> list[dict[str, Any]]:
    """pyccd's RLM of each band on its Tmask matrix."""
    X = tmask_coefficient_matrix(DATES, 365.2425)
    robust = robust_fit.RLM(maxiter=5)
    rows: list[dict[str, Any]] = []
    for band in range(7):
        model = robust.fit(X, Y[band, :])

        predictions = model.predict(X)
        rmse, residuals = calc_rmse(Y[band, :], predictions, num_pm=5)

        row: dict[str, Any] = {
            "band": band,
            "coef_0": model.coef_[0],
            "coef_1": model.coef_[1],
            "coef_2": model.coef_[2],
            "coef_3": model.coef_[3],
            "coef_4": model.coef_[4],
            "nonzero_coefficients": np.count_nonzero(model.coef_),
            "rmse": rmse,
        }
        # Store each residual as its own column
        for i, r in enumerate(residuals):
            row[f"residual_{i}"] = r
        rows.append(row)

        print(f"Band {band}")
        print(f"  Coefficients : {model.coef_}")
        print(f"  RMSE : {rmse}")
        print("-" * 50)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="The reference of one regression on twelve observations of pyccd's reference pixel.")
    parser.add_argument("--model", required=True, choices=["lasso", "robust"])
    parser.add_argument("--out", default=HERE)
    args = parser.parse_args()

    rows = lasso_rows() if args.model == "lasso" else robust_rows()
    name = f"{args.model}_reference.csv"
    pd.DataFrame(rows).to_csv(os.path.join(args.out, name), index=False)
    print(f"\nSaved results to {name}")


if __name__ == "__main__":
    main()
