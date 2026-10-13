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

Input arrays are never modified. `detect_cube` and `detect_cube_columns` read the spectral cube as `float64`, `float32` or `uint16` in any layout with non-negative strides, converting one pixel at a time inside each worker thread, so a `uint16` cube is never widened to `float64` as a whole; the results are those of the same values passed as `float64`. Other dtypes are converted to a C-contiguous `float64` array first, as long as numpy can do so safely.

## Single Pixel

The `detect()` API expects a single pixel time series in the following layout:

| Array | Shape | Description |
|-------|--------|-------------|
| `dates` | `(timesteps,)` | Acquisition dates |
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
| `dates` | `(timesteps,)` | Shared acquisition dates |
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
| `hls_fmask(options, exclude_high_aerosol=False)` | HLS v2.0 `Fmask` (uint8) | not yet |
| `sentinel2_scl(options)` | Sentinel-2 L2A `SCL` (uint8) | not yet |

Each builder writes the `QA_*` codes of the `HarmonicOptions` it is given (default `HarmonicOptions()`), so build the table from the options the run uses. Each decodes most-excluding first: fill, then cloud, then shadow, then snow, then water, then clear.

- **Landsat C2 `QA_PIXEL`:** bit 0 fill; bits 3, 1, 2 (cloud, dilated cloud, cirrus) cloud; bit 4 shadow; bit 5 snow; bit 7 water; otherwise clear. Bit 6 (clear) and the confidence bits 8-15 are not read. This is the table the millworks nlcd plugin builds, entry for entry.
- **Landsat C1 `pixel_qa`:** pyccd's `qabitval`: bit 0 fill; bit 5 cloud; bit 3 shadow; bit 4 snow; bit 2 water; bit 1 clear; bits 8 and 9 together, or bit 10, clear. A value pyccd rejects decodes to fill.
- **HLS `Fmask`:** 255 fill; bits 1, 2, 0 (cloud, adjacent to cloud or shadow, cirrus) cloud; bit 3 shadow; bit 4 snow; bit 5 water; otherwise clear. The aerosol level (bits 6-7) is ignored unless `exclude_high_aerosol=True`, which makes high aerosol cloud. Values above 255 are fill.
- **Sentinel-2 `SCL`:** 0, 1 fill; 2, 3 shadow; 4, 5 clear; 6 water; 7-10 (unclassified, cloud medium and high probability, thin cirrus) cloud; 11 snow; anything else fill.

The HLS and Sentinel-2 tables follow the products' definitions and are tested, but no production run has used them yet; only the Landsat tables are blessed for production use so far. For any other product, `ccdcpp.QaTable(codes)` takes a `(65536,)` uint8 array whose entry `v` is the code of raw value `v`.

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