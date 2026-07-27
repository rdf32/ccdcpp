import numpy as np
import pandas as pd

from ccd.procedures import insufficient_clear_procedure

from ccd import (
    app,
    attr_from_str,
    __check_inputs,
    __sort_dates
)


params = {
    'QA_BITPACKED': False,
    'QA_FILL': 255,
    'QA_CLEAR': 0,
    'QA_WATER': 1,
    'QA_SHADOW': 2,
    'QA_SNOW': 3,
    'QA_CLOUD': 4
}

def read_data(path):
    """Load a sample file containing acquisition days and spectral values.

    The first column is assumed to be the day number, subsequent columns
    correspond to the day number. This improves readability of large datasets.

    Args:
        path: location of CSV containing test data

    Returns:
        A 2D numpy array.
    """
    return np.genfromtxt(path, delimiter=',', dtype=np.int64).T

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

data = read_data(
    'test_3657_3610_observations.csv'
)

(
    dates,
    blues,
    greens,
    reds,
    nirs,
    swir1s,
    swir2s,
    thermals,
    qas
) = data


# ---------------------------------------------------------
# Parameters
# ---------------------------------------------------------

proc_params = app.get_default_params()

if params:
    proc_params.update(params)


# ---------------------------------------------------------
# Prepare arrays
# ---------------------------------------------------------

dates = np.asarray(dates)
qas = np.asarray(qas)

spectra = np.stack(
    (
        blues,
        greens,
        reds,
        nirs,
        swir1s,
        swir2s,
        thermals
    )
)


fitter_fn = attr_from_str(
    proc_params.FITTER_FN
)


__check_inputs(
    dates,
    qas,
    spectra
)


# Sort chronologically
indices = __sort_dates(dates)

dates = dates[indices]
spectra = spectra[:, indices]
qas = qas[indices]


# ---------------------------------------------------------
# Run procedure
# ---------------------------------------------------------

prev_results = None

results, mask = insufficient_clear_procedure(
    dates,
    spectra,
    fitter_fn,
    qas,
    prev_results,
    proc_params
)


# ---------------------------------------------------------
# Write results
# ---------------------------------------------------------

rows = []


for model in results:

    row = {
        "start_day": model['start_day'],
        "end_day": model['end_day'],
        "break_day": model['break_day'],
        "observation_count": model['observation_count'],
        "change_probability": model['change_probability'],
        "curve_qa": model['curve_qa'],
    }


    band_names = [
        "blue",
        "green",
        "red",
        "nir",
        "swir1",
        "swir2",
        "thermal"
    ]


    for name in band_names:

        row[f"{name}_rmse"] = model[name]['rmse']
        row[f"{name}_intercept"] = model[name]['intercept']
        row[f"{name}_magnitude"] = model[name]['magnitude']

        for i, coef in enumerate(
            model[name]['coefficients']
        ):
            row[
                f"{name}_coef_{i}"
            ] = coef


    rows.append(row)



df = pd.DataFrame(rows)

df.to_csv(
    "clear_reference.csv",
    index=False
)


# ---------------------------------------------------------
# Store mask separately
# ---------------------------------------------------------

mask_df = pd.DataFrame(
    {
        "mask": mask.astype(np.int8)
    }
)

mask_df.to_csv(
    "clear_mask_reference.csv",
    index=False
)


print("Saved clear_reference.csv")
print("Saved clear_mask_reference.csv")
