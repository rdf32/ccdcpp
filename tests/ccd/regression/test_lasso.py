import numpy as np
import pandas as pd
from sklearn import linear_model

from ccd.models.lasso import coefficient_matrix
from ccd.math_utils import calc_rmse

# num params / coef: 4
dates = np.array([
        724387, 724419, 724451, 724483,
        724547, 724739, 724867, 724915,
        724931, 724947, 725075, 725091
    ])

X = coefficient_matrix(
    dates,
    365.2425,
    4
)

y = np.array([
    [ 432,  447,  289,  310,  493,  757,  508,  429,  577,  633, 1323,  637],
    [ 514,  602,  484,  549,  731,  954,  793,  652,  803,  968, 1514,  840],
    [ 608,  595,  406,  427,  850, 1131,  853,  745, 1034, 1305, 1662, 1052],
    [ 937, 1891, 2526, 3103, 2355, 1716, 2389, 2168, 2217, 2322, 2240, 1602],
    [1073, 1585, 1547, 1777, 2817, 2838, 2779, 2622, 2974, 3195, 1786, 2740],
    [ 683, 1094,  813,  855, 1752, 2237, 1708, 1595, 1770, 2134, 1157, 1994],
    [1935, 2325, 2665, 2705, 1105,  765, 2415, 1345,  865,  665,  -55,  315]
])

rows = []

for band in range(7):
    lasso = linear_model.Lasso(max_iter=1000)
    model = lasso.fit(X, y[band])

    predictions = model.predict(X)
    rmse, residuals = calc_rmse(y[band], predictions, num_pm=4)

    row = {
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

df = pd.DataFrame(rows)
df.to_csv("lasso_reference.csv", index=False)

print("\nSaved results to lasso_reference.csv")
