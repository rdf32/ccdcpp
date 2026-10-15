# ccdcpp

A high-performance C++17 implementation of the Continuous Change Detection and Classification (CCD) algorithm with Python bindings via **pybind11**.

`ccdcpp` was developed as a native C++ implementation of the USGS PyCCD algorithm while preserving algorithm behavior and output compatibility. The library can be called directly from Python while benefiting from compiled C++ performance and efficient memory management.

> **Performance**
>
> On the benchmark example included in the original PyCCD repository, `ccdcpp` achieved approximately **60× faster execution** than PyCCD. Actual performance improvements will vary depending on hardware, dataset size, and workload, but the implementation is designed to substantially reduce both execution time and memory allocations compared to the original Python implementation.

---

# Features

- Native C++17 implementation of the CCD algorithm
- Python bindings through **pybind11**
- Significant reductions in execution time and memory usage
- OpenMP parallel processing for image cubes
- Zero-copy NumPy interface
- Raw QA input decoded through optional lookup tables (Landsat C1/C2; HLS and Sentinel-2 building blocks)
- Sensor profiles: band roles, units and options in one object, with Landsat presets and HLS and Sentinel-2 presets validated on a real sample (`ccdcpp.presets`)
- Designed to reproduce PyCCD algorithm behavior and outputs
- Cross-platform CMake build system

---


# Why ccdcpp?

Compared to the original Python implementation, `ccdcpp` is designed for production-scale processing of Landsat imagery.

Key advantages include:

- Native C++ execution for substantially lower runtime
- Greatly reduced memory allocations through reusable workspaces
- Zero-copy NumPy interoperability
- Automatic multi-core execution with OpenMP
- Easy integration into existing Python workflows
- Algorithm behavior designed to closely match PyCCD

# Performance Optimizations

The mathematical algorithm remains equivalent to PyCCD, but the implementation was redesigned from the ground up to improve computational efficiency while maintaining output compatibility.

---

# Data Layout

For best performance, input arrays should be **C-contiguous** and use the expected memory layout. The library is designed around cache-friendly access patterns and avoids unnecessary data copies by operating directly on NumPy memory whenever possible.

`dates` are proleptic Gregorian ordinals in non-decreasing order. Every entry point raises `ValueError` (`std::invalid_argument` in C++, `ccd::validate_dates`) on a date earlier than the one before it, once per call and before any pixel runs; equal dates are allowed, and the filters keep the first observation of each date.

Input arrays are never modified. `detect_cube` and `detect_cube_columns` read the spectral cube as `float64`, `float32` or `uint16` in any layout with non-negative strides, converting one pixel at a time inside each worker thread, so a `uint16` cube is never widened to `float64` as a whole; the results are those of the same values passed as `float64`. Other dtypes are converted to a C-contiguous `float64` array first, as long as numpy can do so safely.

## Single Pixel

The `detect()` API expects a single pixel time series in the following layout:

| Array | Shape | Description |
|-------|--------|-------------|
| `dates` | `(timesteps,)` | Acquisition dates, non-decreasing |
| `spectral` | `(bands, timesteps)` | Spectral observations for one pixel |
| `qas` | `(timesteps,)` | QA codes, or raw QA with `qa_table=` (see [QA Input](#qa-input)) |

For example:

```python
dates.shape      # (T,)
spectral.shape   # (7, T)
qas.shape        # (T,)
```

The spectral array stores each band as a contiguous time series, allowing the regression routines to efficiently access observations for each band with minimal cache misses.

---

## Image Cubes

The `detect_cube()` API expects image cubes in the following layout:

| Array | Shape | Description |
|-------|--------|-------------|
| `dates` | `(timesteps,)` | Shared acquisition dates, non-decreasing |
| `spectral` | `(height, width, bands, timesteps)` | Spectral observations |
| `qas` | `(height, width, timesteps)` | QA codes, or raw QA with `qa_table=` (see [QA Input](#qa-input)) |

For example:

```python
dates.shape      # (T,)
spectral.shape   # (H, W, 7, T)
qas.shape        # (H, W, T)
```

Each pixel's complete time series is stored contiguously in memory. This layout provides several advantages:

- Improved CPU cache locality during time-series processing
- Efficient sequential access by the CCD algorithm
- Minimal pointer arithmetic and memory indirection
- Zero-copy interoperability with NumPy
- Efficient OpenMP parallelization across pixels

Using C-contiguous arrays with these layouts maximizes performance and minimizes unnecessary memory copies.

---

## Band Layout

`HarmonicOptions` says which row of `spectral` plays which role. The defaults are the Landsat layout, seven bands in the order blue, green, red, nir, swir1, swir2, thermal:

| Field | Default | Read by |
|-------|---------|---------|
| `NUM_BANDS` | `7` | every entry point: the band count `spectral` must have |
| `DETECTION_BANDS` | `[1, 2, 3, 4, 5]` | change magnitude and the stability check, in list order |
| `TMASK_BANDS` | `[1, 4]` | the Tmask outlier screen |
| `SATURATION_BANDS` | `[0, 1, 2, 3, 4, 5]` | the saturation filter (`REFL_MIN`, `REFL_MAX`) |
| `GREEN_IDX` | `1` | the InsufficientClear green median filter |
| `THERMAL_IDX` | `6` | the thermal transform and filter; `ccdcpp.NO_BAND` for none |

Every row is fitted and reported, so results have `NUM_BANDS` bands in row order: `detect` returns a `bands` list of that length per model and `num_bands`, and `detect_cube_columns` returns `[n, B]` and `[n, B, 7]` arrays. `CHANGE_THRESHOLD` and `OUTLIER_THRESHOLD` default to chi-squared quantiles with 5 degrees of freedom, the default detection band count; a layout with a different count sets them to match. `BLUE_IDX`, `RED_IDX`, `NIR_IDX`, `SWIR1_IDX`, `SWIR2_IDX` and `QA_IDX` are informational and not read. A layout that does not fit the input (wrong band count, an index off the band axis, a repeated index, an empty detection or Tmask list, no green band, thermal in the saturation list, more than `ccdcpp.CCD_MAX_BANDS` bands) raises `ValueError` before any pixel runs.

## Input Unit Convention

The observation filters compare `spectral` against absolute thresholds, so each threshold is a statement about the unit of the input. The defaults are Collection 1 DN, as pyccd consumes it; the physical-unit preset (`presets.landsat_c1_physical()`) is every one of them divided by 10000. See the input unit convention in `harmonic.hpp` for the measured equivalence. The filters' defaults are pyccd's, as pyccd intends them: its exclusive saturation and thermal bounds, and the green median filter scope its InsufficientClear filter computes (see [Differences from pyccd](#differences-from-pyccd) for where pyccd's code does not do what it intends).

| Field | Default | Read by |
|-------|---------|---------|
| `THERMAL_SCALE`, `THERMAL_OFFSET` | `10.0`, `-27315.0` | the Standard procedure's thermal transform, `thermal * scale + offset` |
| `REFL_MIN`, `REFL_MAX` | `0.0`, `10000.0` | the saturation filter, on `SATURATION_BANDS` |
| `REFL_MIN_INCLUSIVE`, `REFL_MAX_INCLUSIVE` | `False`, `False` | whether a value equal to `REFL_MIN` / `REFL_MAX` is kept |
| `THERM_MIN`, `THERM_MAX` | `-9320.0`, `7070.0` | the thermal filter, on `THERMAL_IDX`, after the transform |
| `THERM_MIN_INCLUSIVE`, `THERM_MAX_INCLUSIVE` | `False`, `False` | whether a value equal to `THERM_MIN` / `THERM_MAX` is kept |
| `MEDIAN_GREEN_FILTER` | `400.0` | the InsufficientClear green median filter's offset |

The `*_INCLUSIVE` flags are not unit constants: a bound means the same in DN and in reflectance, so the preset leaves them as they are. The defaults are exclusive on both ends of both ranges, which is pyccd's form, `0 < v < 10000` for the saturation bound and `-9320 < v < 7070` for the thermal one: an observation with a saturation band at exactly `0` or `10000`, or NaN, is dropped. With either end exclusive NaN is rejected; with both ends of a range inclusive the test is `!(v < lo || v > hi)`, under which `0` and `10000` are kept and NaN passes. A NaN the filter keeps reaches the fit: on a synthetic pixel with one NaN in a saturation band, `detect` returns no segments and an empty processing mask, and `detect_cube_columns` reports no failure for it.

**Green median filter scope.** The InsufficientClear procedure drops cloud-like observations by their green value: the threshold is the median green of the masked observations dated up to `STAT_ORD`, plus `MEDIAN_GREEN_FILTER`, and an observation at or above it is dropped. `GREEN_FILTER_SCOPE`, a `ccdcpp.GreenFilterScope`, says where: `StatWindow` (the default) drops it only up to `STAT_ORD`, which is the scope pyccd's `insufficient_clear_filter` computes its mask for, `All` at any date, and `Off` never. When no masked observation is dated up to `STAT_ORD` there is no median and nothing is dropped. It is not a unit constant. pyccd's own filter drops nothing (below): on an InsufficientClear pixel with an observation up to `STAT_ORD` at or above the threshold, the default gives what pyccd's filter computes and `Off` gives what pyccd's code does.

**Inclusive bounds and the all-dates filter.** Set `REFL_MIN_INCLUSIVE` and `REFL_MAX_INCLUSIVE` to `True` to keep observations on a saturation bound (and NaN), and `GREEN_FILTER_SCOPE` to `All` to drop green outliers at every date.

### Differences from pyccd

ccdcpp recreates pyccd. Where pyccd's code does not do what it evidently intends, ccdcpp does what it intends, so the two differ on the inputs that reach it:

- **RLM convergence.** pyccd's robust fit stops when `not numpy.any(numpy.fabs(x0 - x > tol))` (`ccd/models/robust_fit.py`, `_check_converge`): the parenthesis takes the absolute value of the comparison, not of the difference, so a coefficient that fell by more than `tol` counts as converged. ccdcpp stops when every coefficient moved by less than `tol` in either direction. A fit that pyccd stops early gets different weights and coefficients.
- **Date order.** pyccd orders observations with `np.argsort(dates)`, which is not a stable sort, so which of two observations on the same date survives the duplicate-date filter depends on the sort. ccdcpp takes dates in non-decreasing order (unsorted dates raise `ValueError`) and keeps the first observation of each date.
- **Green median filter.** pyccd's `insufficient_clear_filter` computes the green mask for the observations up to `STAT_ORD` and applies it with `standard_mask[standard_mask][:n] &= green_mask` (`ccd/qa.py`), which writes into a copy, so the filter drops nothing. ccdcpp applies it, with the `StatWindow` scope by default.
- **Seasonal RMSE ties.** pyccd's `find_closest_doy` (`ccd/change.py`) takes the 24 fit-window observations closest in day of year to the last peek date with `np.argsort(d_yr)[:num]`, which is not a stable sort. Observations a multiple of 1461 days apart are exactly equally distant, so which of the tied observations are kept depends on the sort. ccdcpp breaks ties in window order, so the seasonal RMSE, and with it the change magnitudes, can differ on dense series.

On the synthetic corpus, with pyccd patched for these four (the convergence test as `|x0 - x| > tol`, a stable date sort, the green mask written through `standard_mask[np.flatnonzero(standard_mask)[:n]] &= green_mask`, and `np.argsort(d_yr, kind="stable")` in `find_closest_doy`), ccdcpp's defaults match pyccd to 1e-6 on every InsufficientClear pixel (175 in DN) and on every pixel where a saturation-band value on a bound reaches the saturation filter (106, and the golden test's bounds pixel); with the green filter left unpatched, `Off` matches on the same 175. One more difference is kept on purpose: an undefined raw QA value decodes to fill where pyccd raises (see [QA Input](#qa-input)). One pyccd quirk is kept for parity: the PermanentSnow and InsufficientClear procedures test the untransformed thermal band against the Celsius bounds (see `harmonic.hpp`).

## QA Input

`qas` reaches the procedures as one class code per observation. It can be passed two ways.

**Pre-classified codes (uint8).** `qas` already holds the codes, and is read as it is. The codes are the `HarmonicOptions` `QA_*` fields:

| Field | Default | Read by |
|-------|---------|---------|
| `QA_FILL` | `255` | the quality statistics: not counted as a valid observation |
| `QA_CLEAR` | `0` | the fit (kept), the clear percentage |
| `QA_WATER` | `1` | the fit (kept), the clear percentage |
| `QA_SNOW` | `3` | the snow percentage and the PermanentSnow procedure |
| `QA_CLOUD` | `4` | the quality statistics' cloud count |
| `QA_SHADOW` | `2` | not read by the procedures; written by the QA tables |
| `QA_CIRRUS1`, `QA_CIRRUS2`, `QA_OCCLUSION` | `8`, `9`, `10` | informational; not read by ccdcpp |

Any code other than `QA_CLEAR` and `QA_WATER` keeps the observation out of the fit; a code that is none of the above counts only toward the total.

**Raw QA plus a table (uint8 or uint16).** Pass the archive's own QA band and a `ccdcpp.QaTable` as `qa_table=`. Each pixel's values are decoded through the table, per thread, before the procedures run, so the run is exactly the run of the decoded codes passed as uint8, without a decoded copy of the cube. A uint16 `qas` without a table raises `TypeError`.

```python
table = ccdcpp.qa_tables.landsat_c2_qa_pixel(hoptions)

columns = ccdcpp.detect_cube_columns(
    dates, spectra, qa_pixel, hoptions, loptions, threads=8, qa_table=table)
```

| Builder | Product | Validated for production |
|---------|---------|--------------------------|
| `landsat_c2_qa_pixel(options)` | Landsat Collection 2 `QA_PIXEL` (uint16) | yes |
| `landsat_c1_pixel_qa(options)` | Landsat Collection 1 `pixel_qa` (uint16) | yes |
| `hls_fmask(options, exclude_high_aerosol=False)` | HLS v2.0 `Fmask` (uint8) | decode checked on real data: all 43 values present (38 in L30, 43 in S30) match the HLS v2.0 User Guide (T16SGC, 2020–2023) |
| `sentinel2_scl(options)` | Sentinel-2 L2A `SCL` (uint8) | decode checked on real data: every class present (0, 2–11) matches SentiWiki (T16SGC, 2020–2023) |

Each builder writes the `QA_*` codes of the `HarmonicOptions` it is given (default `HarmonicOptions()`), so build the table from the options the run uses. Each decodes most-excluding first: fill, then cloud, then shadow, then snow, then water, then clear.

- **Landsat C2 `QA_PIXEL`:** bit 0 fill; bits 3, 1, 2 (cloud, dilated cloud, cirrus) cloud; bit 4 shadow; bit 5 snow; bit 7 water; otherwise clear. Bit 6 (clear) and the confidence bits 8-15 are not read. This is the table the millworks nlcd plugin builds, entry for entry.
- **Landsat C1 `pixel_qa`:** pyccd's `qabitval`: bit 0 fill; bit 5 cloud; bit 3 shadow; bit 4 snow; bit 2 water; bit 1 clear; bits 8 and 9 together, or bit 10, clear. A value pyccd rejects decodes to fill.
- **HLS `Fmask`:** 255 fill; bits 1, 2, 0 (cloud, adjacent to cloud or shadow, reserved) cloud; bit 3 shadow; bit 4 snow; bit 5 water; otherwise clear. The aerosol level (bits 6-7) is ignored unless `exclude_high_aerosol=True`, which makes high aerosol cloud. Values above 255 are fill. The HLS v2.0 User Guide lists bit 0 as reserved and not used in v2.0 (Fmask folds cirrus into cloud), so it decodes to cloud only should it ever be set.
- **Sentinel-2 `SCL`:** 0, 1 fill; 2, 3 (cast shadows, cloud shadows) shadow; 4, 5 clear; 6 water; 7-10 (unclassified, cloud medium and high probability, thin cirrus) cloud; 11 snow; anything else fill.

The HLS and Sentinel-2 tables follow the products' definitions, are tested, and were checked against that documentation on every value present in the real sample the presets were validated on, one cell of MGRS tile T16SGC, 2020–2023 (Fmask bit 0 was never set; SCL class 1 was absent). What the decode costs on that sample, Fmask over bright surfaces and SCL 7 over water, is under [Validation on real data](#validation-on-real-data). For any other product, `ccdcpp.QaTable(codes)` takes a `(65536,)` uint8 array whose entry `v` is the code of raw value `v`.

## Cadence Parameters

The Standard procedure has constants that assume something about the revisit interval of the input or the false-alarm rate of the change test. They are `HarmonicOptions` fields. Each one defaults to pyccd's value, which was tuned for 16-day Landsat:

| Field | Default | Unit | Read by |
|-------|---------|------|---------|
| `PEEK_REFERENCE_DAYS` | `16.0` | days | the peek size: `round(PEEK_SIZE * PEEK_REFERENCE_DAYS / median gap)`, never below `PEEK_SIZE` |
| `VARIOGRAM_MIN_LAG_DAYS` | `30` | days | the variogram: the smallest lag whose most common separation exceeds this, over the pairs further apart than this |
| `RETRAIN_MIN_OBS` | `24` | observations | lookforward: refit while the window holds fewer than this; a window of at most this many compares against the model RMSE, a longer one against the seasonal RMSE |
| `RETRAIN_SPAN_FACTOR` | `1.33` | ratio | lookforward: refit once the window spans at least this times the span of the last fit |
| `SEASONAL_RMSE_COUNT` | `24` | observations | the seasonal RMSE: residuals of this many fit observations closest in day of year to the last peek observation |
| `SEASONAL_RMSE_DIVISOR` | `4.0` | none | the seasonal RMSE: their norm divided by this. pyccd's 4 is `sqrt(24 - 8)`; it is not derived from `SEASONAL_RMSE_COUNT` |
| `CHANGE_PROBABILITY` | `0.99` | probability | the change threshold of a peek window longer than the default: the chi-squared quantile at `1 - (1 - CHANGE_PROBABILITY) ** (PEEK_SIZE / peek)` |

`ccdcpp.chi2_quantile(p, df)` is the chi-squared quantile the procedure computes, from the same boost call. `CHANGE_THRESHOLD` is `chi2_quantile(CHANGE_PROBABILITY, 5)` to the last bit, so a caller changing the probability or the detection band count sets the threshold to match:

```python
h.CHANGE_PROBABILITY = 0.995
h.CHANGE_THRESHOLD = ccdcpp.chi2_quantile(h.CHANGE_PROBABILITY, len(h.DETECTION_BANDS))
```

A value that makes no sense raises `ValueError` before any pixel runs: zero or negative days, a count below 1, a probability outside (0, 1), a factor or divisor of zero or less.

**Denser cadences.** The peek size follows the cadence through `PEEK_REFERENCE_DAYS`: it is `round(PEEK_SIZE * PEEK_REFERENCE_DAYS / (median gap + 0.001))`, never below `PEEK_SIZE`, where the gap is between consecutive observations the processing mask keeps, up to `STAT_ORD`. At 8 days it is 12 and at 5 days 19 (`6 * 16 / 5 ≈ 19`). On the cell the HLS and Sentinel-2 presets were validated on (T16SGC, 2020–2023) `hls_s30()` and `sentinel2_l2a()` ran at a median gap of 5 days, peek 19, and the combined `hls()` at 4–5 days, peek 19–24 (`6 * 16 / 4 = 24`); `hls_l30()` at 16 days, keeping `PEEK_SIZE`. The change threshold falls as the window grows, so that a longer window keeps the default window's false-alarm rate: it is the chi-squared quantile, with one degree of freedom per detection band, at `1 - (1 - CHANGE_PROBABILITY) ** (PEEK_SIZE / peek)`. With the defaults that is 9.24 at a peek of 12 (8 days), 6.83 at 19 (5 days) and 5.90 at 24 (4 days). Both are computed per pixel from the `PEEK_SIZE` and `CHANGE_THRESHOLD` the caller passes, as pyccd's `adjustpeek` and `adjustchgthresh` do. A series whose median gap is about 14.8 days or more keeps `PEEK_SIZE` and `CHANGE_THRESHOLD` as given.

`MEOW_SIZE` and the `RETRAIN_*` constants are observation counts and are deliberately not scaled with the cadence, as pyccd does not scale them: at the 4–5 day cadence of `hls()` on the validation cell, 12 observations are about 50–60 days and 24 about 100–120, where at 16 days they are about half a year and a year. A caller who wants them in time rather than in observations sets them from the cadence.

## Sensor Profiles

A `ccdcpp.SensorProfile` describes one sensor's input in one immutable object: the band names, the role each band plays, the detection, Tmask and saturation lists by role name, the unit-coupled thresholds, the Lasso constants, the cadence options, `STAT_ORD` and the QA table. `detect`, `detect_cube` and `detect_cube_columns` take one as `profile=`:

```python
from ccdcpp import presets

columns = ccdcpp.detect_cube_columns(
    dates, spectra, qa_pixel, threads=8, profile=presets.landsat_c2())
```

The profile supplies `hoptions`, `loptions`, `qa_table` and `preprocess` (see [Preprocessing](#preprocessing)). Any of them passed explicitly is used instead, whole: `hoptions=` replaces the profile's `HarmonicOptions` rather than being merged into it, `qa_table=None` reads `qas` as pre-classified codes, and `preprocess=None` runs the default plan. The profile's table and plan are built with the `HarmonicOptions` the run uses. `spectral` is checked against the profile's band count first (`ValueError`). Only the count: a stack with the right number of rows in another order runs as if it were in the profile's order. `band_names=`, keyword only and only with `profile=` (`TypeError` otherwise), names the rows as the caller holds them, a store's band list say; given, it must be the profile's `band_names` in the profile's order, and a misordered stack raises `ValueError` naming the first row that differs, before anything runs. A Sentinel-2 store that keeps B08 with the 10 m bands (`B02, B03, B04, B08, B05, ...`) is refused by `sentinel2_l2a()` this way rather than having B08 read as red edge 1. Without it the call is unchanged. The outputs of `detect` and `detect_cube_columns` gain `band_names`, a tuple in row order; `detect_cube`'s per-pixel dicts do not. Without `profile=`, `hoptions` and `loptions` are required and a call, positional or by keyword, is the call of the extension function of the same name (`ccdcpp.ccdcpp.detect`, ...), output included.

`profile.options()` returns new `(HarmonicOptions, LassoOptions)`, `profile.qa_table()` the table (or `None` for pre-classified input), built by the `qa_tables` builder `qa_table_builder` names with the profile's `qa_table_options` as keyword arguments (`{"exclude_high_aerosol": True}` for `hls_fmask`; the builder's defaults where none is given), and `profile.check(spectral, band_names=None)` raises `ValueError` when the band axis (second from last) does not match, or the names given are not the profile's in its order. Profiles are hashable and picklable, where the option objects are not, so a profile can be sent to worker processes.

### Presets

`ccdcpp.presets` holds the inputs ccdcpp is validated on, Landsat, and presets for HLS v2.0 and Sentinel-2 L2A. Their values follow the products' documentation, are tested on synthetic series, and were validated on a real sample: one cell of MGRS tile T16SGC near Atlanta, 2020–2023, four years (see [Validation on real data](#validation-on-real-data)).

The three Landsat presets take seven bands in the order blue, green, red, nir, swir1, swir2, thermal (TM/ETM+ bands 1, 2, 3, 4, 5, 7, 6; OLI/TIRS bands 2, 3, 4, 5, 6, 7, 10).

| Preset | Optical input | Thermal input | QA input | Validated for production |
|--------|---------------|---------------|----------|--------------------------|
| `landsat_c1_dn()` | reflectance x 10000 | Kelvin x 10 | pre-classified codes | yes |
| `landsat_c1_physical()` | reflectance 0..1 (DN / 10000) | degrees Celsius / 100 | pre-classified codes | yes |
| `landsat_c2()` | reflectance 0..1 (`SR_B* * 2.75e-5 - 0.2`) | degrees Celsius / 100 | raw `QA_PIXEL` (uint16) | yes |
| `hls_l30()` | reflectance 0..1 (`B* * 0.0001`) | B10 degrees Celsius / 100 (`B10 * 0.0001`) | raw `Fmask` (uint8) | yes (T16SGC, 2020–2023) |
| `hls_s30()` | reflectance 0..1 (`B* * 0.0001`) | none | raw `Fmask` (uint8) | yes, with a caveat: Fmask over bright surfaces (T16SGC, 2020–2023) |
| `hls()` | reflectance 0..1 (`B* * 0.0001`), L30 and S30 | none | raw `Fmask` (uint8) | yes, with a caveat: Fmask over bright surfaces (T16SGC, 2020–2023) |
| `sentinel2_l2a()` | reflectance 0..1 (`(DN + BOA_ADD_OFFSET) / 10000`) | none | raw `SCL` (uint8) | validated at median level on Planetary Computer's Esri/MSFT/S2RP L2A, 2020–23; ESA's own processing not sampled |

`landsat_c1_dn().options()` is `HarmonicOptions()` and `LassoOptions()`, field for field. `landsat_c1_physical()` is the physical-unit preset (see the input unit convention in `harmonic.hpp`): `REFL_MIN`/`REFL_MAX` 0 and 1, `THERM_MIN`/`THERM_MAX` -0.932 and 0.707, `MEDIAN_GREEN_FILTER` 0.04, `alpha` and `coef_floor` 1e-4, which is every absolute threshold and both Lasso constants of the DN preset divided by 10000, and the identity thermal transform. It is the same fit as the DN preset, with coefficients in the input's units. `landsat_c2()` has the same thresholds (Collection 2's saturation range in DN, 7273..43636, is reflectance 0..1) and decodes `QA_PIXEL` with `qa_tables.landsat_c2_qa_pixel`.

The HLS and Sentinel-2 presets have the thresholds and Lasso constants of `landsat_c1_physical()`, detect change on the five Landsat-equivalent bands green, red, nir, swir1 and swir2 with the default chi-squared(5) thresholds, use green and swir1 for Tmask, saturation-screen every optical row, have no statistics cutoff (`stat_ord=None`) and keep pyccd's cadence values (see [Cadence Parameters](#cadence-parameters)). The single-product presets name their rows by product band ID; `hls()` names them by role, since its rows mix two numberings.

| Preset | Rows (role) | Left out |
|--------|-------------|----------|
| `hls_l30()` | B01 (coastal), B02 (blue), B03 (green), B04 (red), B05 (nir), B06 (swir1), B07 (swir2), B10 (thermal) | B09 cirrus, B11 TIRS 2 |
| `hls_s30()` | B01 (coastal), B02 (blue), B03 (green), B04 (red), B05, B06, B07 (red_edge_1..3), B08 (nir_broad), B8A (nir), B11 (swir1), B12 (swir2) | B09 water vapour, B10 cirrus |
| `hls()` | coastal, blue, green, red, nir (L30 B05, S30 B8A), swir1, swir2 | thermal, which S30 lacks |
| `sentinel2_l2a()` | B02 (blue), B03 (green), B04 (red), B05, B06, B07 (red_edge_1..3), B08 (nir_broad), B8A (nir), B11 (swir1), B12 (swir2) | B01, B09 (60 m, atmospheric); L2A has no B10 |

- **HLS scaling.** Every HLS layer is `int16` times 0.0001. For the reflective bands that is the product's scale factor; L30 B10 is brightness temperature in degrees Celsius with scale factor 0.01, so `B10 * 0.0001` is degrees Celsius / 100, `CELSIUS_HUNDREDTHS`, and one divisor serves every row. The fill, -9999, becomes -0.9999 and is dropped by the reflectance and thermal bounds.
- **HLS QA.** `Fmask` decodes through `qa_tables.hls_fmask` with the aerosol level ignored; `dataclasses.replace(preset, qa_table_options={"exclude_high_aerosol": True})` treats high aerosol as cloud.
- **S30 and Sentinel-2 bands.** The red edges and B08 are fitted, reported and saturation-screened but not detected on; HLS does not adjust them to the OLI bandpasses.
- **Sentinel-2 offset.** From processing baseline 04.00 (25 January 2022) L2A stores `DN = reflectance * 10000 - BOA_ADD_OFFSET`, with a negative, band-dependent `BOA_ADD_OFFSET` in each product's metadata (0 before). The offset varies by scene and ccdcpp never sees the metadata, so the caller applies it; a series that mixes corrected and uncorrected scenes jumps at the baseline change and detects a false break. Some archives distribute corrected bands while their metadata still declares the offset (earth-search's `sentinel-2-l2a` COGs), and there it must not be applied a second time. DN 0 (no data) becomes 0 or negative and is dropped.
- **Sentinel-2 QA.** `SCL` decodes through `qa_tables.sentinel2_scl`; class 7 (unclassified) is cloud, which costs water pixels (see [Validation on real data](#validation-on-real-data)). A caller who wants them passes `qa_table=ccdcpp.QaTable(codes)`, with `codes` the preset table's codes and entry 7 changed.
- **L30 and S30 in one series.** An L30 and an S30 observation on one date are deduplicated like any other pair, and the first in array order is kept: put L30 before S30 on a shared date.
- **Dense cadences.** The peek window follows the cadence (about 12 observations at 8 days, 19 at 5, 24 at 4; on the validation cell `hls()` ran at 4–5 days, peek 19–24), with the change threshold that goes with it. `MEOW_SIZE` and the `RETRAIN_*` counts stay pyccd's observation counts, as described under Denser cadences.

### Validation on real data

The HLS and Sentinel-2 presets were run on one cell: Landsat ARD tile h024v013 chunk 00030008 near Atlanta, cut to MGRS tile T16SGC, 2020-01-01 to 2023-12-31, read from Microsoft Planetary Computer (`hls2-l30` and `hls2-s30`, which are LP DAAC's HLS v2.0, and `sentinel-2-l2a` at 20 m). The reference was Landsat Collection 2 ARD of the same cell and years run with `landsat_c2()` without its statistics cutoff. Four years of one cell is the whole record; nothing here was measured elsewhere. The quarry and SCL 7 counts are recomputed in the validation's `results/qa_costs.json` (`scripts/qa_costs.py`), kept with the validation outside this repository.

| Preset | Pixels, dates | Status | What was measured |
|--------|---------------|--------|-------------------|
| `hls_l30()` | 300,288 at 30 m, 238 | validated | Break agreement per pixel with Landsat 8+9 ARD kappa 0.44, where adding Landsat 7 to the ARD gives 0.43; matched breaks a median 8 days apart, 85% within 96 days, no date bias; model levels within 0.004 reflectance. With a thin 2020 (37 scenes, half partial) no break is detected before the second half of 2021. |
| `hls_s30()` | 300,288, 461 | validated, with a caveat: Fmask over bright surfaces | 41% more break pixels than L30, a median 16 days earlier, in classes that change. Fmask on Sentinel-2 (no thermal band) calls bright barren land cloud: over all 1,862 quarry (land cover 31) pixels on the HLS grid it flags 63% of observations as cloud, and 1,145 of them go to InsufficientClear (68% cloud over those). 170 of ARD's 173 InsufficientClear pixels are quarries, and both of L30's 2. |
| `hls()` | 300,288, 699 | validated, with a caveat: Fmask over bright surfaces | Mostly follows S30 (kappa 0.71 with S30, 0.45 with L30). L30 first on a shared date worked on all 85 same-date pairs (84 dates; on 2021-11-17 L30 has two granules). 1,741 pixels go to InsufficientClear (S30: 2,101; L30: 2). |
| `sentinel2_l2a()` | 675,648 at 20 m, 565 | validated at median level on Planetary Computer's Esri/MSFT/S2RP L2A, 2020–23; ESA's own processing not sampled | With the per-scene offset the 2022 baseline change looks like any other window (0.34% of pixels break within 60 days of it; 92.3% without), apart from 24 water pixels (0.60% of water pixels against a 0.08% baseline), a seasonal winter-to-spring step. Against S30, levels about 0.01 higher and about twice the RMSE (no BRDF or bandpass adjustment), kappa 0.48, fewer break pixels (2.35% against 3.62%). |

- **SCL 7 stays cloud, at a cost in water pixels.** SCL 7 (unclassified) is 2.13% of the cell's non-fill observations. As cloud, together with SCL 2 (cast shadows) on small water bodies, it sends 1,355 of the cell's 3,983 water pixels, and 3,426 pixels in all, to the InsufficientClear procedure. Decoded as fill it would send 2,545, and as clear 587.
- **QA decode.** `hls_fmask` matches the HLS v2.0 User Guide on all 43 values present (38 in L30, 43 in S30) and `sentinel2_scl` matches SentiWiki on every class present (0, 2–11). High aerosol is flagged on about half the observations; treating it as cloud would remove another 16.5% (L30) and 16.8% (S30) of what QA keeps, which is why the presets ignore it.
- **Coastal B01.** Its saturation screen drops 0.1–0.8% of observations, all at or below 0, none at 1.0; without it 99.0% (L30) and 99.4% (S30) of breaks are unchanged. It stays in the screen.
- **Band order.** Only the band count of a stack is checked unless `band_names=` is passed; the sample's Sentinel-2 store kept B08 with the 10 m bands, which `sentinel2_l2a()` would have read as red edge 1 without complaint (see [Sensor Profiles](#sensor-profiles)).
- **Fixtures.** Four of the sample's pixels (an evergreen-forest pixel, open water, high-intensity developed and a pixel with a 2022 break) in all three products, and a fifth in the two HLS products whose result depends on the coastal low bound, are pinned in `tests/python/fixtures/t16sgc_2020_2023`, raw values and raw QA with each scene's Sentinel-2 offset, with every preset's output on them (`tests/python/test_real_fixtures.py`). The forest pixel, `481_393`, is named 1-based (column, row) as the production CCD table names it.

### Roles and thresholds

The roles are `blue`, `green`, `red`, `nir`, `swir1`, `swir2`, `thermal`, `coastal`, `red_edge_1`, `red_edge_2`, `red_edge_3`, `nir_broad` and `other` (`ccdcpp.ROLES`). `nir` is the Landsat near infrared, TM/ETM+ band 4 and OLI band 5 (865 nm), and on Sentinel-2 the 865 nm B8A; the broad 842 nm Sentinel-2 B08 is `nir_broad`. Each role but `other` names at most one band; `other` bands are fitted and reported but no list can name them. A role resolves to its band's position in `band_names`, and each list keeps the order the profile gives, which is the order the procedures accumulate in. So a profile whose bands are listed in another order, with `spectral` in that order, gives the same output with its band axis in that order.

`change_threshold` and `outlier_threshold` default to `None`, which computes `chi2_quantile(CHANGE_PROBABILITY, n)` and `chi2_quantile(ccdcpp.OUTLIER_PROBABILITY, n)` for `n` detection bands; for five bands these are the `HarmonicOptions` defaults to the last bit. A profile whose detection count is not five but whose threshold is set to the chi2(5) default gets a `UserWarning` from `options()`.

### Adding a sensor

A sensor is one `SensorProfile` call; `ccdcpp.presets` writes its profiles that way. **Not shipped:** the Landsat 8/9 Collection 2 profile below, with the coastal band, shows the shape; its values are not validated and it is not part of `ccdcpp.presets`.

```python
from ccdcpp import CELSIUS_HUNDREDTHS, REFLECTANCE, SensorProfile

def landsat_oli_c2():
    return SensorProfile(
        name="landsat_oli_c2",
        band_names=("SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "ST_B10"),
        roles=("coastal", "blue", "green", "red", "nir", "swir1", "swir2", "thermal"),
        detection=("green", "red", "nir", "swir1", "swir2"),
        tmask=("green", "swir1"),
        saturation=("coastal", "blue", "green", "red", "nir", "swir1", "swir2"),
        units=REFLECTANCE,
        alpha=1e-4,
        coef_floor=1e-4,
        thermal=CELSIUS_HUNDREDTHS,
        stat_ord=736694,  # 2017-12-31, the Landsat statistics cutoff
        qa_table_builder="landsat_c2_qa_pixel",
    )
```

`units` is a `ccdcpp.Units` (`REFL_MIN`, `REFL_MAX`, `MEDIAN_GREEN_FILTER`, and `refl_inclusive=(lo, hi)` for `REFL_MIN_INCLUSIVE`/`REFL_MAX_INCLUSIVE`, default `(False, False)`; `ccdcpp.DN` and `ccdcpp.REFLECTANCE` are the two Landsat blocks), `thermal` a `ccdcpp.Thermal` (transform and bounds, and `inclusive=(lo, hi)` for `THERM_MIN_INCLUSIVE`/`THERM_MAX_INCLUSIVE`, default `(False, False)`; `ccdcpp.KELVIN_TENTHS`, `ccdcpp.CELSIUS_HUNDREDTHS`) or `None` for an input without one, and `cadence` a mapping of the [Cadence Parameters](#cadence-parameters) and `MEOW_SIZE`/`PEEK_SIZE` (`ccdcpp.CADENCE_FIELDS`). `green_filter_scope` is `"stat_window"` (the default), `"all"` or `"off"` (`ccdcpp.GREEN_FILTER_SCOPES`), for `GREEN_FILTER_SCOPE`. `qa_table_options` is a mapping of keyword arguments for the `qa_table_builder` (none by default), each a name and a value of exactly the type `ccdcpp.QA_TABLE_OPTIONS` lists for the builder (`exclude_high_aerosol` takes `True` or `False`, not `None` or `1`). The profile stores every argument its builder declares, the default written out where none is given, so the HLS presets hold `(("exclude_high_aerosol", False),)` and `()`, `{"exclude_high_aerosol": False}` and the preset are one profile; a profile without a builder stores none. Options without a builder, a repeated name, a name the builder does not take, a value of another type, or input that is not a mapping or `(name, value)` pairs raise `ValueError`. `alpha` and `coef_floor` have no default because both are statements about the input unit and a wrong value does not raise. A profile that breaks a layout rule (an unknown or repeated role, a list naming a role the profile lacks, no green band, thermal in the saturation list, thermal band without `thermal=`) raises `ValueError` when it is constructed.

## Preprocessing

Before a procedure fits a pixel it builds the processing mask, the set of observations it fits: an optional transform of one band of the pixel's private copy, then a chain of filter steps run in order. A `PreprocessPlan` holds one such `ProcedurePlan` (transform and chain) per procedure, `standard`, `snow` (PermanentSnow) and `insufficient_clear`, and each pixel runs the one of the procedure it selects. The steps are:

| Step | Effect |
|------|--------|
| `KeepQa(codes)` | keep only observations whose QA code is in `codes` |
| `AddQa(codes)` | also keep every observation whose code is in `codes`; must be followed by `Dedupe` |
| `ThermalRange(band, range)` | keep only observations whose thermal value is in the range |
| `BandRange(bands, range)` | keep only observations on which every band is in the range, tested in list order |
| `GreenMedian(band, offset, max_date, scope)` | drop observations at or above the median of `band` over the kept observations dated up to `max_date`, plus `offset`, at the dates `scope` names |
| `Dedupe` | keep the first kept observation of each date |

A range keeps `min < x < max`; an inclusive end also keeps a value equal to it, and NaN passes only a range inclusive at both ends. A chain holds at most `CHAIN_CAPACITY` (8) steps.

### The default plan

Without a plan, `detect`, `detect_cube` and `detect_cube_columns` run `default_plan(hoptions)`, the plan the `HarmonicOptions` describe. For `HarmonicOptions()` it is (`ccdcpp.default_plan(ccdcpp.HarmonicOptions()).describe()`):

```text
standard
  transform  band 6 <- x * 10 - 27315
  0  KeepQa codes {0, 1}
  1  ThermalRange band 6, -9320 < x < 7070
  2  BandRange bands [0, 1, 2, 3, 4, 5], 0 < x < 10000
  3  Dedupe
snow
  transform  none
  0  KeepQa codes {0, 1}
  1  ThermalRange band 6, -9320 < x < 7070
  2  BandRange bands [0, 1, 2, 3, 4, 5], 0 < x < 10000
  3  Dedupe
  4  AddQa codes {3}
  5  Dedupe
insufficient_clear
  transform  none
  0  KeepQa codes {0, 1}
  1  ThermalRange band 6, -9320 < x < 7070
  2  BandRange bands [0, 1, 2, 3, 4, 5], 0 < x < 10000
  3  Dedupe
  4  GreenMedian band 1, drop x >= median(dates <= 736694) + 400, scope StatWindow
  5  Dedupe
```

Every procedure keeps `QA_CLEAR` and `QA_WATER`, tests `THERMAL_IDX` against `THERM_MIN`/`THERM_MAX` (omitted when `THERMAL_IDX` is `NO_BAND`), tests `SATURATION_BANDS` against `REFL_MIN`/`REFL_MAX` and dedupes; `snow` then adds `QA_SNOW` back and dedupes, and `insufficient_clear` runs the green median filter (`GREEN_IDX`, `MEDIAN_GREEN_FILTER`, `STAT_ORD`, `GREEN_FILTER_SCOPE`) and dedupes. Only `standard` has a transform, `THERMAL_SCALE`/`THERMAL_OFFSET`, and none when that is the identity, so the other two test the thermal band in the caller's unit, as pyccd does. Passing the default plan explicitly gives the same output, to the bit, as passing none.

### A custom plan in C++

A plan is a value; edit the default or build chains from the named step constructors, and pass it to `detect` or `detect_cube` as the trailing `preprocess` argument, shared by every pixel and thread of the call:

```cpp
#include "ccd/ccd.hpp"

namespace pp = ccd::preprocess;

ccd::HarmonicOptions h;
pp::PreprocessPlan plan = pp::default_plan(h);

// The thermal transform on every procedure, not only Standard.
plan.snow.transform = plan.standard.transform;
plan.insufficient_clear.transform = plan.standard.transform;

// Standard keeps snow observations too.
plan.standard.chain = pp::Chain{
    pp::Step::keep_qa(pp::qa_set({h.QA_CLEAR, h.QA_WATER, h.QA_SNOW})),
    pp::Step::thermal_range(h.THERMAL_IDX, ccd::filter::thermal_range(h)),
    pp::Step::band_range(h.SATURATION_BANDS, ccd::filter::reflectance_range(h)),
    pp::Step::dedupe(),
};

const auto shared = std::make_shared<const pp::PreprocessPlan>(plan);
ccd::FitResult result = ccd::detect(dates, spectral, qas, h, l, nullptr, shared);
```

`pp::validate(plan, bands)` runs once per call, before any pixel, and throws `std::invalid_argument`, naming the procedure and the step, when a band is not below the band count, a `GreenMedian` has no band, a transformed band is also tested by a `BandRange` of its procedure, an `AddQa` is not followed by a `Dedupe`, or a chain is over capacity.

### A custom plan in Python

The same classes are bound: `ccdcpp.StepKind`, `Step` (with the constructors `keep_qa`, `add_qa`, `thermal_range`, `band_range`, `green_median`, `dedupe`), `Chain`, `Affine`, `ProcedurePlan`, `PreprocessPlan`, `default_plan(hoptions)` and `validate_plan(plan, bands)`. Plans compare with `==`, and `describe()` (also the `repr`) prints them as above. Each entry point takes one as `preprocess=`, keyword only, checks it against the band count (`ValueError`) and copies it before the GIL is released:

```python
plan = ccdcpp.default_plan(hoptions)
plan.standard.chain[0] = ccdcpp.Step.keep_qa([hoptions.QA_CLEAR, hoptions.QA_WATER, hoptions.QA_SNOW])
out = ccdcpp.detect(dates, spectra, qas, hoptions, loptions, preprocess=plan)
```

With a sensor profile, a plan is written by role and QA code name instead, as frozen dataclasses: `KeepQa(codes=("clear", "water"))`, `AddQa(codes=("snow",))`, `ThermalRange(min=, max=, inclusive=)`, `BandRange(roles=, min=, max=, inclusive=)`, `GreenMedian(offset=, scope=, max_date=, role="green")`, `Dedupe()` and `Transform(role, scale, offset)`, gathered per procedure as `ProcedureChain(steps=, transform=)` and per plan as `Preprocess(standard=, snow=, insufficient_clear=)`. A field left `None` takes the value the default plan would take from the run's options; a `ProcedureChain` with `steps=None` keeps that procedure's default steps, and its `transform` defaults to `ccdcpp.DEFAULT_TRANSFORM`, the procedure's default transform (`None` for none); a procedure left `None` keeps its default transform and steps, so `Preprocess()` is the default plan. Roles resolve to the profile's rows as its band lists do, and code names (`ccdcpp.QA_CODE_FIELDS`) to the run's `QA_*` codes. A profile holds it as `SensorProfile(preprocess=...)` and stays hashable and picklable; `profile.preprocess_plan(options)` builds the `PreprocessPlan`, and the entry points run it unless `preprocess=` is passed (`None` for the default plan, a `PreprocessPlan`, or another `Preprocess`).

Landsat Collection 2 with its thermal band handed over as the raw `ST_B10` DN, converted to the degrees Celsius / 100 of `landsat_c2()`'s bounds on every procedure, `((DN * 0.00341802 + 149.0) - 273.15) / 100`:

```python
import dataclasses
import ccdcpp
from ccdcpp import Preprocess, ProcedureChain, Transform, presets

# ((DN * 0.00341802 + 149.0) - 273.15) / 100
st_b10 = Transform("thermal", scale=3.41802e-5, offset=-1.2415)
c2_raw_thermal = dataclasses.replace(
    presets.landsat_c2(),
    name="landsat_c2_raw_thermal",
    preprocess=Preprocess(
        standard=ProcedureChain(transform=st_b10),
        snow=ProcedureChain(transform=st_b10),
        insufficient_clear=ProcedureChain(transform=st_b10),
    ),
)

columns = ccdcpp.detect_cube_columns(
    dates, spectra, qa_pixel, threads=8, profile=c2_raw_thermal)
```

A `Preprocess` that does not fit its profile raises `ValueError`: an unknown QA code name or scope when it is constructed, a role the profile lacks when the profile is, and the `validate_plan` checks when the plan is built.

### Remaining differences from pyccd

The default plan is pyccd's preprocessing as pyccd intends it. What still differs is listed in [Differences from pyccd](#differences-from-pyccd): the RLM convergence test, the date order and the first observation of a date kept by `Dedupe`, and the green median filter, which ccdcpp applies (`StatWindow` scope) where pyccd's drops nothing (`GreenMedian(scope="off")` gives pyccd's code). An undefined raw QA value decodes to fill where pyccd raises. The PermanentSnow and InsufficientClear procedures keep pyccd's untransformed thermal test; a plan with the transform on every procedure, as above, changes it.

## Native C++ Implementation

The core CCD algorithm was translated into modern C++17 to eliminate the overhead associated with Python execution.

## Python Bindings

Python bindings were developed using **pybind11**, allowing existing Python workflows to utilize the C++ implementation with minimal code changes.

## Reusable Workspace Buffers

One of the largest performance improvements comes from eliminating repeated memory allocations.

Instead of constructing temporary arrays during each stage of the algorithm, reusable workspace buffers are allocated once and reused throughout processing. This significantly reduces dynamic allocations, improves cache locality, and lowers memory overhead.

## Reduced Redundant Computation

Several portions of the original implementation repeatedly compute values that remain unchanged.

`ccdcpp` caches and reuses these intermediate results whenever possible, including:

- Harmonic basis matrices
- Regression workspaces
- LASSO solver buffers
- Design matrices that would otherwise be rebuilt repeatedly

Reducing redundant computation lowers overall runtime while preserving algorithm behavior.

## Zero-Copy NumPy Interface

Custom `ArrayView` classes provide lightweight views directly into NumPy memory.

Rather than copying arrays into C++, the library operates directly on the existing NumPy buffers whenever possible, dramatically reducing memory traffic for large datasets.

## Python GIL Release

Long-running computations release Python's Global Interpreter Lock (GIL).

This allows multiple OpenMP worker threads to execute concurrently while preventing Python from becoming a serialization bottleneck.

## OpenMP Parallel Processing

The `detect_cube()` API automatically parallelizes processing across CPU cores using OpenMP.

Entire image chunks can therefore be processed efficiently without requiring Python multiprocessing or additional user code.

---

# Installation

The repository includes an `environment.yml` file containing the recommended build environment.

## 1. Create the Conda Environment

```bash
conda env create -f environment.yml
conda activate geo
```

## 2. Configure the Project

```bash
cmake -S . -B build
```

## 3. Build

```bash
cmake --build build --config Release
```

## 4. Install the Python Package

```bash
pip install .
```

---

# Usage

Benchmark notebook - tests/benchmark.ipynb

## Single Pixel Detection

`detect()` runs the CCD algorithm on a single pixel time series.

```python
import time
import numpy as np
import ccdcpp

dates = np.asarray(dates.copy(), dtype=np.int64, order="C")
qas = np.asarray(qas.copy(), dtype=np.uint8, order="C")

spectra = np.asarray(
    np.stack(
        (
            blues,
            greens,
            reds,
            nirs,
            swir1s,
            swir2s,
            thermals,
        )
    ),
    dtype=np.float64,
    order="C",
)

hoptions = ccdcpp.HarmonicOptions()
loptions = ccdcpp.LassoOptions()

hoptions.PEEK_SIZE = 6
hoptions.MEOW_SIZE = 12

loptions.max_iter = 1000
loptions.alpha = 1.0
loptions.tolerance = 1e-4
loptions.fit_intercept = True

start = time.perf_counter()

results = ccdcpp.detect(
    dates,
    spectra,
    qas,
    hoptions,
    loptions,
)

elapsed = time.perf_counter() - start

print(f"CPP runtime: {elapsed:.4f} seconds")
```

---

## Image Cube Processing

`detect_cube()` performs CCD on an entire image cube and automatically parallelizes computation using OpenMP.

```python
import time

start = time.perf_counter()

results = ccdcpp.detect_cube(
    dates,
    spectra,
    qas,
    hoptions,
    loptions,
)

elapsed = time.perf_counter() - start

print(f"CPP runtime: {elapsed:.4f} seconds")
```

No additional multiprocessing code is required—the library automatically distributes pixels across CPU cores.

---

# License and Attribution

`ccdcpp` is released into the public domain under the **Unlicense**.

This project is a C++17 translation of the USGS LCMAP PyCCD software.
The original PyCCD implementation was also released into the public domain.

The goals of this project are to:

- preserve PyCCD algorithm behavior
- reproduce PyCCD outputs
- provide a high-performance native C++ implementation
- enable Python interoperability through pybind11

Minor implementation changes were necessary to accommodate modern NumPy versions and to adapt portions of the Python implementation into efficient C++.

---

# Independent Development

`ccdcpp` is an independent software project developed separately from the U.S. Geological Survey (USGS).

This project was created independently for research, software engineering, and performance optimization of the publicly available CCD algorithm.

The author is not affiliated with, employed by, contracted by, or acting on behalf of the USGS in the development of this project. The USGS did not commission, fund, review, endorse, or provide technical support for this implementation.

The relationship to PyCCD is limited to algorithmic translation and validation against the publicly available implementation. All software architecture, memory management strategies, parallelization, optimizations, and implementation decisions in `ccdcpp` are original work developed independently for this project.