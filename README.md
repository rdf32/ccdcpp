# ccdcpp

ccdcpp is a C++17 implementation of the Continuous Change Detection and Classification (CCDC) algorithm, with Python bindings through pybind11. It reproduces the USGS LCMAP PyCCD algorithm and its outputs, doing what PyCCD's code intends where the code does otherwise (every such difference is listed under [Differences from pyccd](#differences-from-pyccd)), and runs one pixel about 200 times faster than PyCCD in about half the process memory. A run takes one pixel or a whole image cube, in parallel with OpenMP, and can be continued when new observations arrive instead of rerun.

A run is two things: an **input**, which says how to read the data (`ccdcpp.Input`, or a sensor profile that builds one), and a **configuration**, which says what to do with it, in physical units (`ccdcpp.Config`; the `ccdc` preset by default). Upgrading from 0.10? See [docs/migration-1.0.md](docs/migration-1.0.md).

## Contents

- [Features](#features)
- [Quick start](#quick-start): [Python](#one-pixel), [a cube](#a-cube), [C++](#from-c)
- [The two calls](#the-two-calls)
- [Inputs and profiles](#inputs-and-profiles): [the input](#the-input), [band layout](#band-layout), [QA](#qa-input), [sensor profiles](#sensor-profiles), [presets](#presets), [adding a sensor](#adding-a-sensor)
- [Configurations and presets](#configurations-and-presets): [the rule](#the-rule), [changing a configuration](#changing-a-configuration), [shared values](#shared-values), [stages](#stages-and-strategies), [TOML presets](#toml-presets), [cadence](#cadence-parameters), [preprocessing](#preprocessing)
- [Resuming a run](#resuming-a-run)
- [Diagnostics](#diagnostics)
- [Build and test](#build-and-test)
- [The C++ API](#the-c-api)
- [How it works](#how-it-works)
- [Performance](#performance)
- [Differences from pyccd](#differences-from-pyccd)
- [License and Attribution](#license-and-attribution)

# Features

- **PyCCD's algorithm as it intends**, every difference from its code listed and checked against Zhu's MATLAB reference ([Differences from pyccd](#differences-from-pyccd)).
- **Fast**: about 200 times PyCCD's speed per pixel in about half its process memory; a 500 x 500 Landsat cell over 40 years in 23 s on 18 threads ([Performance](#performance)).
- **Two calls**: `detect` for a pixel, `detect_cube` for an image cube in parallel, returning columns (OpenMP, the GIL released), reading float64, float32 or uint16 cubes in any layout without copying them.
- **Inputs and sensor profiles**: band roles, units and QA decoding of an input in one object, with presets for Landsat Collection 1 and 2, HLS v2.0 and Sentinel-2 L2A (`ccdcpp.presets`).
- **Raw QA input**: Landsat `QA_PIXEL` and `pixel_qa`, HLS `Fmask` and Sentinel-2 `SCL` decoded through lookup tables per pixel.
- **One configuration for every input**: the algorithm in physical units (reflectance, degrees Celsius, probabilities, bands by role), its decisions as stages decided by named strategies, its preprocessing as a chain of steps; immutable values, TOML presets, part of a run's record ([Configurations and presets](#configurations-and-presets)).
- **Resume**: a stored run continued on new observations, equal bit for bit to a full rerun, at about a third of its time ([Resuming a run](#resuming-a-run)).
- **Diagnostics**: every transition, change test, dropped observation and segment of the pixels you name, as rows for pandas, without changing the output ([Diagnostics](#diagnostics)).
- **Typed**: the Python package ships its types (PEP 561); the results are `TypedDict`s.
- **A C++ API** of the same two calls, `ccd::detect` and `ccd::detect_cube`, and a CMake build ([The C++ API](#the-c-api)).

# Quick start

## Install

ccdcpp builds with CMake through scikit-build-core. It needs Python 3.11 or later, numpy, a C++17 compiler, Boost, Eigen 3 and pybind11; OpenMP is optional (without it a cube runs on one thread). `environment.yml` is the conda environment it is developed in (a Windows export; elsewhere install the same packages from conda-forge).

```bash
conda env create -f environment.yml
conda activate geo
pip install .
```

## One pixel

`detect` runs one pixel. The arrays are `dates` (T,), proleptic Gregorian ordinals in non-decreasing order, `spectral` (bands, T) and `qas` (T,). The input says what the rows are and in which units; here the sensor profile `presets.landsat_c2()`, Landsat Collection 2 surface reflectance with its thermal band in degrees Celsius / 100 and the raw `QA_PIXEL` band. The pixel is synthetic: twenty years at a 16-day cadence with an annual cycle, noise, 15% of observations cloudy and a land-cover change in June 2012.

```python
from datetime import date

import numpy as np
import ccdcpp
from ccdcpp import presets

rng = np.random.default_rng(42)
dates = np.arange(date(2000, 1, 1).toordinal(), date(2020, 1, 1).toordinal(), 16, dtype=np.int64)
T = dates.size
season = np.sin(2 * np.pi * dates / 365.25)

# Rows: blue, green, red, nir, swir1, swir2 (reflectance 0..1), thermal (degrees C / 100).
level = np.array([0.04, 0.07, 0.06, 0.30, 0.18, 0.09])
spectral = np.empty((7, T))
spectral[:6] = level[:, None] * (1 + 0.3 * season) + rng.normal(0, 0.005, (6, T))
spectral[:6, dates >= date(2012, 6, 1).toordinal()] *= [[1.3], [1.4], [1.6], [0.6], [1.5], [1.7]]
spectral[6] = 0.20 + 0.10 * season

qas = np.full(T, 21824, dtype=np.uint16)   # QA_PIXEL clear
qas[rng.random(T) < 0.15] = 22280          # QA_PIXEL cloud

out = ccdcpp.detect(dates, spectral, qas, input=presets.landsat_c2())

for m in out["models"]:
    print(date.fromordinal(m["start_day"]), date.fromordinal(m["end_day"]),
          "break", date.fromordinal(m["break_day"]),
          "change probability", m["change_probability"], "observations", m["observation_count"])
print(out["band_names"])
print(sorted(out["models"][0]["bands"][0]))
```

```text
2000-01-17 2012-05-25 break 2012-06-10 change probability 1.0 observations 239
2012-06-10 2019-07-16 break 2019-08-01 change probability 0.0 observations 131
('blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal')
['coefficients', 'intercept', 'magnitude', 'rmse']
```

Two segments: the change on the first clear observation after 2012-06-01 (change probability 1), and the last curve, which ends at the series end (change probability 0). Each segment has one fit per row (`bands`): intercept, seven harmonic coefficients, RMSE and change magnitude, in the input's units. The result also holds the `processing_mask`, the observations the fit used. The run uses the `ccdc` configuration, as no `config=` names another.

The examples below continue from this one.

## A cube

`detect_cube` runs every pixel of an (H, W, bands, T) cube in parallel and returns the segments as columns, one entry per segment; `ccdcpp.to_pixels(cols)` reads them as one dict per pixel, as `detect` returns them.

```python
H, W = 3, 4
cube = np.ascontiguousarray(np.broadcast_to(spectral, (H, W, 7, T)) + rng.normal(0, 0.002, (H, W, 7, T)))
cube_qa = np.ascontiguousarray(np.broadcast_to(qas, (H, W, T)))

cols = ccdcpp.detect_cube(dates, cube, cube_qa, input=presets.landsat_c2(), threads=4)

print(cols["pixels_total"], cols["pixels_with_segments"], len(cols["row"]), cols["failures"])
print(cols["coefficients"].shape)
print(sorted(k for k in cols if not k.startswith("pixels")))
```

```text
12 12 24 []
(24, 7, 7)
['band_names', 'break_day', 'change_probability', 'coefficients', 'col', 'curve_qa', 'end_day', 'failures', 'intercept', 'magnitude', 'num_bands', 'observation_count', 'rmse', 'row', 'start_day', 'threads']
```

## From C++

The same two calls are `ccd::detect` and `ccd::detect_cube` (`include/ccd/run/run.hpp`). What a call runs is a `ccd::RunSpec`: the input (`ccd::Input`), the compiled configuration (`ccd::Algorithm`, the `ccdc` preset unless you give one), and whether to return resume records and processing masks. The arrays are `ccd::ArrayView`s over your memory. Link the CMake target `ccd_core`.

```cpp
#include <cstdint>
#include <iostream>
#include <vector>

#include "ccd/ccd.hpp"

// One pixel: T dates, 7 rows of T values (row after row), T QA codes (0 clear).
void one_pixel(const std::vector<std::int64_t>& dates,
               const std::vector<double>& spectral,
               const std::vector<std::uint8_t>& qas)
{
    const std::size_t T = dates.size();

    ccd::RunSpec spec;   // Collection-1 DN input, the "ccdc" configuration
    ccd::PixelOutput run = ccd::detect(spec,
        ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {T}),
        ccd::ArrayView<const double, 2>::contiguous(spectral.data(), {7, T}),
        ccd::ArrayView<const std::uint8_t, 1>::contiguous(qas.data(), {T}));

    for (const ccd::ChangeModel& m : run.result.models)
        std::cout << m.start_day << " " << m.end_day << " " << m.break_day << " "
                  << m.bands[3].rmse << "\n";
}

// A cube: (H, W, 7, T) values and (H, W, T) QA, on 8 threads.
void a_cube(const std::vector<std::int64_t>& dates, const std::vector<double>& cube,
            const std::vector<std::uint8_t>& qas, std::size_t H, std::size_t W)
{
    const std::size_t T = dates.size();

    ccd::CubeOutput r = ccd::detect_cube(ccd::RunSpec{},
        ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {T}),
        ccd::ArrayView<const double, 4>::contiguous(cube.data(), {H, W, 7, T}),
        ccd::ArrayView<const std::uint8_t, 3>::contiguous(qas.data(), {H, W, T}),
        /*threads=*/8);

    for (const ccd::CubePixel& p : r.pixels)
        std::cout << p.row << "," << p.col << ": " << p.result.models.size() << " segments\n";
    for (const ccd::CubeFailure& f : r.failures)
        std::cerr << f.row << "," << f.col << ": " << f.message << "\n";
}
```

[The C++ API](#the-c-api) has the rest: configurations, resume, observers and what each call returns.

# The two calls

```text
ccdcpp.detect(dates, spectral, qas, *, input, config="ccdc", record=False, resume=None,
              band_names=None, trace=False)
ccdcpp.detect_cube(dates, spectral, qas, *, input, config="ccdc", threads=0, record=False,
                   resume=None, band_names=None, trace_pixels=None, masks=False)
ccdcpp.to_pixels(cols)
```

| Argument | What it is |
|---|---|
| `input=` | how to read the data: a `ccdcpp.Input`, or a `ccdcpp.SensorProfile` whose `input()` is used ([Inputs and profiles](#inputs-and-profiles)) |
| `config=` | what to do with it: a `ccdcpp.Config` or a preset's name, `"ccdc"` by default ([Configurations and presets](#configurations-and-presets)) |
| `record=` | also return the run's resume record (`"resume"`, `"resume_header"`) |
| `resume=` | an earlier call's output made with `record=True`: continue that run on this longer series ([Resuming a run](#resuming-a-run)) |
| `masks=` | `detect_cube` only: also return `"processing_mask"`, every pixel's processing mask in one `(H, W, T)` uint8 array (`detect` always returns its pixel's) |
| `band_names=` | with a profile, the names of the rows as the caller holds them, checked against the profile's before anything runs |
| `trace=`, `trace_pixels=` | the run step by step, for one pixel or the cube pixels named ([Diagnostics](#diagnostics)) |
| `threads=` | `detect_cube`'s OpenMP threads; 0, the default, is the runtime's count |

| Array | `detect` | `detect_cube` |
|---|---|---|
| `dates` | `(T,)` int64 | `(T,)` int64, shared by every pixel |
| `spectral` | `(B, T)` float64 | `(H, W, B, T)` float64, float32 or uint16 |
| `qas` | `(T,)` | `(H, W, T)` |

`qas` is uint8 QA codes, or raw uint8 or uint16 QA when the input has a QA table ([QA input](#qa-input)). `dates` are proleptic Gregorian ordinals (`datetime.date.toordinal()`) in non-decreasing order: both calls raise `ValueError` on a date earlier than the one before it, once per call and before any pixel runs. Equal dates are allowed, and the filters keep the first observation of each date.

Input arrays are never modified. `detect_cube` reads the spectral cube in any layout with non-negative strides, converting one pixel at a time inside each worker thread, so a uint16 cube is never widened to float64 as a whole; the results are those of the same values passed as float64. Other dtypes are converted to a C-contiguous float64 array first, as long as numpy can do so safely. C-contiguous arrays, each pixel's series contiguous, are the fastest.

`detect` returns a dict: `models` (one dict per segment), `processing_mask` and `num_bands`, plus `band_names` with a profile. `detect_cube` returns columns, one entry per segment (`row`, `col`, `start_day`, `end_day`, `break_day`, `curve_qa`, `observation_count`, `change_probability`, `intercept` `[n, B]`, `coefficients` `[n, B, 7]`, `rmse` `[n, B]`, `magnitude` `[n, B]`) with `num_bands`, `failures`, `threads`, `pixels_total` and `pixels_with_segments`. A pixel with no segment has no row. A pixel that raises does not stop the cube: it is listed in `failures` with its row, column and message. With `masks=True`, `to_pixels` gives each pixel its mask:

```python
cols = ccdcpp.detect_cube(dates, cube, cube_qa, input=presets.landsat_c2(), threads=4, masks=True)
pixels = ccdcpp.to_pixels(cols)
p = pixels[6]
single = ccdcpp.detect(dates, cube[p["row"], p["col"]], cube_qa[p["row"], p["col"]],
                       input=presets.landsat_c2())
print(cols["processing_mask"].shape, len(pixels), p["row"], p["col"])
print(p["models"] == single["models"], np.array_equal(p["processing_mask"], single["processing_mask"]))
```

```text
(3, 4, 457) 12 1 2
True True
```

**Types.** The package is typed (PEP 561, `py.typed`): every function carries parameter and return types, and `ccdcpp.pyi` and `qa_tables.pyi` type the compiled extension, so pyright, mypy and editors check calls against them. The results are `TypedDict`s in `ccdcpp.results`: `detect` returns a `PixelResult`, `detect_cube` a `CubeColumns` and `to_pixels` a list of `Pixel`; at run time they are plain dicts.

# Inputs and profiles

## The input

A `ccdcpp.Input` (`ccd::Input` in C++) says how to read the data, and nothing else: the QA codes and the table that decodes raw QA, the role of each row, the reflectance encoding, the thermal transform and the stored saturation codes. It is immutable: `ccdcpp.Input(**fields)` by keyword, each field defaulting to Collection-1 DN with pre-classified QA codes, and `replace(**fields)` for a changed copy; inputs are equal and hashed by value.

```python
dn = ccdcpp.Input()
reflectance = dn.replace(reflectance_scale=1.0, thermal_scale=1.0, thermal_offset=0.0,
                         thermal_at_100c=1.0)
print(dn.reflectance_scale, dn.thermal_scale, dn.thermal_offset, dn.roles)
print(reflectance == presets.landsat_c1_physical().input())
```

```text
10000.0 10.0 -27315.0 ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal']
True
```

| Field | Default | What it says |
|-------|---------|---------|
| `reflectance_scale` | `10000.0` | reflectance `r` is held as `r * reflectance_scale + reflectance_offset` (Collection-1 DN 10000, reflectance 1) |
| `reflectance_offset` | `0.0` | the data's value of reflectance 0; Collection-2 DN (`r = DN * 2.75e-5 - 0.2`) is `reflectance_scale = 1 / 2.75e-5`, `reflectance_offset = 0.2 / 2.75e-5` |
| `thermal_scale`, `thermal_offset` | `10.0`, `-27315.0` | the thermal transform, `thermal * scale + offset`, applied by a procedure whose chain reads `thermal = "celsius"` (Standard in `ccdc`); the default takes Kelvin x 10 to Celsius x 100 |
| `thermal_at_100c` | `10000.0` | the value 100 °C has after the transform (Celsius x 100: 10000; Celsius / 100: 1) |
| `saturation_code`, `thermal_saturation_code` | `None`, `None` | the value a saturated observation holds in the reflective rows and in the thermal row, as the data stores it (Collection-1 DN: 20000 and 20000), or `None`; no step of the `ccdc` configuration reads them |
| `roles` | the Landsat seven | one role per row ([Band layout](#band-layout)) |
| `qa_fill` ... `qa_occlusion`, `qa_table` | the codes below, `None` | the QA codes, and the table that decodes raw QA ([QA input](#qa-input)) |

The outputs are in the input's units: DN in, DN out. [docs/units.md](docs/units.md) states the boundary between the data (the storage domain) and the configuration (the physical domain), with a worked Collection-1 DN example, the physical-unit input and its measured equivalence with DN, and which products store a saturation code.

## Band layout

`roles` says which row of `spectral` plays which role (`ccdcpp.ROLES` lists them). The default is the Landsat layout, seven bands in the order blue, green, red, nir, swir1, swir2, thermal. Every row is fitted and reported, so results have one band per row, in row order. `"green"` is required (the green median filter reads it); without `"thermal"` the thermal transform and range are skipped.

The configuration names its bands by role (the shared `detection_bands`; `window_screen__bands`; `band_range`'s `bands`, blue, green, red, nir, swir1 and swir2 in `ccdc`), resolved to the input's rows when a run resolves the configuration, so one configuration fits any band order. An input that does not fit (wrong band count, an unknown or repeated role, no green row, more than `ccdcpp.CCD_MAX_BANDS` bands) raises `ValueError` before any pixel runs, as does a configuration naming a role the input lacks.

## QA input

`qas` reaches the procedures as one class code per observation. It can be passed two ways.

**Pre-classified codes (uint8).** `qas` already holds the codes, the input's `qa_*` fields:

| Field | Default | Read by |
|-------|---------|---------|
| `qa_fill` | `255` | the quality statistics: not counted as a valid observation |
| `qa_clear` | `0` | the fit (kept), the clear percentage |
| `qa_water` | `1` | the fit (kept), the clear percentage |
| `qa_snow` | `3` | the snow percentage and the PermanentSnow procedure |
| `qa_cloud` | `4` | the quality statistics' cloud count |
| `qa_shadow` | `2` | not read by the procedures; written by the QA tables |
| `qa_cirrus1`, `qa_cirrus2`, `qa_occlusion` | `8`, `9`, `10` | informational; not read by ccdcpp |

The configuration names the codes it keeps (`keep_qa`'s `codes`, `["clear", "water"]` in `ccdc`), resolved to the input's. Any code other than those keeps the observation out of the fit; a code that is none of the above counts only toward the total.

**Raw QA plus a table (uint8 or uint16).** Pass the archive's own QA band, with a `ccdcpp.QaTable` as the input's `qa_table` (a sensor profile's `input()` carries its own). Each pixel's values are decoded through the table, per thread, before the procedures run, so the run is exactly the run of the decoded codes passed as uint8, without a decoded copy of the cube. A uint16 `qas` without a table raises `TypeError`.

```python
c2 = presets.landsat_c2()
h = c2.input().replace(qa_table=ccdcpp.qa_tables.landsat_c2_qa_pixel(c2.input()))
explicit = ccdcpp.detect(dates, spectral, qas, input=h)
print(explicit["models"] == out["models"])
```

```text
True
```

| Builder | Product | Validated for production |
|---------|---------|--------------------------|
| `landsat_c2_qa_pixel(input)` | Landsat Collection 2 `QA_PIXEL` (uint16) | yes |
| `landsat_c1_pixel_qa(input)` | Landsat Collection 1 `pixel_qa` (uint16) | yes |
| `hls_fmask(input, exclude_high_aerosol=False)` | HLS v2.0 `Fmask` (uint8) | decode checked on real data: all 43 values present (38 in L30, 43 in S30) match the HLS v2.0 User Guide (T16SGC, 2020–2023) |
| `sentinel2_scl(input)` | Sentinel-2 L2A `SCL` (uint8) | decode checked on real data: every class present (0, 2–11) matches SentiWiki (T16SGC, 2020–2023) |

Each builder writes the `qa_*` codes of the `Input` it is given, so build the table from the input the run uses. Each decodes most-excluding first: fill, then cloud, then shadow, then snow, then water, then clear.

- **Landsat C2 `QA_PIXEL`:** bit 0 fill; bits 3, 1, 2 (cloud, dilated cloud, cirrus) cloud; bit 4 shadow; bit 5 snow; bit 7 water; otherwise clear. Bit 6 (clear) and the confidence bits 8-15 are not read.
- **Landsat C1 `pixel_qa`:** pyccd's `qabitval`: bit 0 fill; bit 5 cloud; bit 3 shadow; bit 4 snow; bit 2 water; bit 1 clear; bits 8 and 9 together, or bit 10, clear. A value pyccd rejects decodes to fill.
- **HLS `Fmask`:** 255 fill; bits 1, 2, 0 (cloud, adjacent to cloud or shadow, reserved) cloud; bit 3 shadow; bit 4 snow; bit 5 water; otherwise clear. The aerosol level (bits 6-7) is ignored unless `exclude_high_aerosol=True`, which makes high aerosol cloud. Values above 255 are fill. Bit 0 is reserved and unused in v2.0, so it decodes to cloud only should it ever be set.
- **Sentinel-2 `SCL`:** 0, 1 fill; 2, 3 (cast shadows, cloud shadows) shadow; 4, 5 clear; 6 water; 7-10 (unclassified, cloud medium and high probability, thin cirrus) cloud; 11 snow; anything else fill.

For any other product, `ccdcpp.QaTable(codes)` takes a `(65536,)` uint8 array whose entry `v` is the code of raw value `v`.

## Sensor profiles

A `ccdcpp.SensorProfile` describes one sensor's data in one immutable object: the band names, the role each band plays, the units (`Units(scale, offset)`), the thermal transform (`Thermal(scale, offset, at_100c)`), the stored saturation codes and the QA table builder. It holds no algorithm value: the configuration is the same for every profile. `input=` takes one wherever it takes a `ccdcpp.Input`.

- `profile.input()` is its `ccdcpp.Input`, the QA table included, built once per profile and shared.
- `spectral` is checked against the profile's band count (`ValueError`), and only the count. `band_names=` (keyword only, only with a profile) names the rows as the caller holds them; a stack in another order then raises `ValueError` naming the first row that differs, before anything runs. A Sentinel-2 store that keeps B08 with the 10 m bands is refused by `sentinel2_l2a()` this way rather than having B08 read as red edge 1.
- The outputs of `detect` and `detect_cube` gain `band_names` with a profile.
- `profile.check(spectral, band_names=None)` is the band check. Profiles are hashable and picklable, so a profile can be sent to worker processes.

## Presets

`ccdcpp.presets` holds the inputs ccdcpp is validated on, Landsat, and presets for HLS v2.0 and Sentinel-2 L2A, which follow the products' documentation, are tested on synthetic series and were validated on a real sample: one cell of MGRS tile T16SGC near Atlanta, 2020–2023. The three Landsat presets take seven bands in the order blue, green, red, nir, swir1, swir2, thermal (TM/ETM+ bands 1, 2, 3, 4, 5, 7, 6; OLI/TIRS bands 2, 3, 4, 5, 6, 7, 10).

| Preset | Optical input | Thermal input | QA input | Validated for production |
|--------|---------------|---------------|----------|--------------------------|
| `landsat_c1_dn()` | reflectance x 10000 | Kelvin x 10 | pre-classified codes | yes |
| `landsat_c1_physical()` | reflectance 0..1 (DN / 10000) | degrees Celsius / 100 | pre-classified codes | yes |
| `landsat_c2()` | reflectance 0..1 (`SR_B* * 2.75e-5 - 0.2`) | degrees Celsius / 100 | raw `QA_PIXEL` (uint16) | yes |
| `hls_l30()` | reflectance 0..1 (`B* * 0.0001`) | B10 degrees Celsius / 100 (`B10 * 0.0001`) | raw `Fmask` (uint8) | yes (T16SGC, 2020–2023) |
| `hls_s30()` | reflectance 0..1 (`B* * 0.0001`) | none | raw `Fmask` (uint8) | yes, with a caveat: Fmask over bright surfaces (T16SGC, 2020–2023) |
| `hls()` | reflectance 0..1 (`B* * 0.0001`), L30 and S30 | none | raw `Fmask` (uint8) | yes, with a caveat: Fmask over bright surfaces (T16SGC, 2020–2023) |
| `sentinel2_l2a()` | reflectance 0..1 (`(DN + BOA_ADD_OFFSET) / 10000`) | none | raw `SCL` (uint8) | validated at median level on Planetary Computer's Esri/MSFT/S2RP L2A, 2020–23; ESA's own processing not sampled |

`landsat_c1_dn().input()` is `ccdcpp.Input(saturation_code=20000.0, thermal_saturation_code=20000.0)`, field for field: the defaults with the values a Collection-1 saturated observation holds, which the other presets, whose data is in reflectance, do not declare: a code does not survive the conversion ([docs/units.md](docs/units.md#saturation-codes) lists which products store one). `landsat_c1_physical()` has `reflectance_scale` 1 and the identity thermal transform, so the configuration's values resolve to the DN ones divided by 10000: the same fit, with coefficients in the input's units. `landsat_c2()` reads the same way (reflectance 0..1 is Collection 2's DN 7273..43636, which `band_range` under `ccdc` keeps) and decodes `QA_PIXEL`.

The HLS and Sentinel-2 presets read reflectance and degrees Celsius / 100 as `landsat_c1_physical()` does. Under `ccdc`, `band_range` tests blue, green, red, nir, swir1 and swir2 (not coastal, the red edges or B08), change is detected on green, red, nir, swir1 and swir2 with the chi-squared(5) thresholds, Tmask reads green and swir1, and the cadence values are pyccd's.

| Preset | Rows (role) | Left out |
|--------|-------------|----------|
| `hls_l30()` | B01 (coastal), B02 (blue), B03 (green), B04 (red), B05 (nir), B06 (swir1), B07 (swir2), B10 (thermal) | B09 cirrus, B11 TIRS 2 |
| `hls_s30()` | B01 (coastal), B02 (blue), B03 (green), B04 (red), B05, B06, B07 (red_edge_1..3), B08 (nir_broad), B8A (nir), B11 (swir1), B12 (swir2) | B09 water vapour, B10 cirrus |
| `hls()` | coastal, blue, green, red, nir (L30 B05, S30 B8A), swir1, swir2 | thermal, which S30 lacks |
| `sentinel2_l2a()` | B02 (blue), B03 (green), B04 (red), B05, B06, B07 (red_edge_1..3), B08 (nir_broad), B8A (nir), B11 (swir1), B12 (swir2) | B01, B09 (60 m, atmospheric); L2A has no B10 |

- **HLS scaling.** Every HLS layer is `int16` times 0.0001; for L30 B10 (degrees Celsius, scale 0.01) that is degrees Celsius / 100, so one divisor serves every row. The fill, -9999, becomes -0.9999, which `band_range` and `thermal_range` under `ccdc` drop.
- **HLS QA.** `Fmask` decodes with the aerosol level ignored; `dataclasses.replace(preset, qa_table_options={"exclude_high_aerosol": True})` treats high aerosol as cloud.
- **S30 and Sentinel-2 bands.** Under `ccdc` the red edges and B08 are fitted and reported, but neither tested by `band_range` nor detected on; a configuration that lists them in `band_range`'s `bands` tests them.
- **Sentinel-2 offset.** From processing baseline 04.00 (25 January 2022) L2A stores `DN = reflectance * 10000 - BOA_ADD_OFFSET`, with a band-dependent offset in each product's metadata. ccdcpp never sees the metadata, so the caller applies it; a series that mixes corrected and uncorrected scenes jumps at the baseline change and detects a false break. Some archives distribute corrected bands while their metadata still declares the offset (earth-search's `sentinel-2-l2a` COGs); there it must not be applied twice.
- **Sentinel-2 QA.** `SCL` class 7 (unclassified) is cloud, which costs water pixels (below). A caller who wants them runs a copy of `input()` whose `qa_table` is `ccdcpp.QaTable(codes)` with entry 7 changed.
- **L30 and S30 in one series.** An L30 and an S30 observation on one date are deduplicated like any other pair, the first in array order kept: put L30 before S30 on a shared date.

**Validation on real data.** The HLS and Sentinel-2 presets were run on Landsat ARD tile h024v013 chunk 00030008 near Atlanta, cut to MGRS tile T16SGC, 2020-01-01 to 2023-12-31, from Microsoft Planetary Computer (`hls2-l30`, `hls2-s30`, `sentinel-2-l2a` at 20 m), against Landsat Collection 2 ARD of the same cell run with `landsat_c2()`, all without a statistics cutoff. Four years of one cell is the whole record.

| Preset | Pixels, dates | What was measured |
|--------|---------------|-------------------|
| `hls_l30()` | 300,288 at 30 m, 238 | Break agreement per pixel with Landsat 8+9 ARD kappa 0.44 (0.43 with Landsat 7 added); matched breaks a median 8 days apart, 85% within 96 days, no date bias; model levels within 0.004 reflectance. No break before the second half of 2021 (a thin 2020). |
| `hls_s30()` | 300,288, 461 | 41% more break pixels than L30, a median 16 days earlier. Fmask on Sentinel-2 calls bright barren land cloud: over the 1,862 quarry pixels it flags 63% of observations as cloud and sends 1,145 of them to InsufficientClear. |
| `hls()` | 300,288, 699 | Mostly follows S30 (kappa 0.71 with S30, 0.45 with L30); L30 first on a shared date worked on all 85 same-date pairs. 1,741 pixels go to InsufficientClear (S30: 2,101; L30: 2). |
| `sentinel2_l2a()` | 675,648 at 20 m, 565 | With the per-scene offset the 2022 baseline change looks like any other window (0.34% of pixels break within 60 days of it; 92.3% without), apart from 24 water pixels. Against S30, levels about 0.01 higher and about twice the RMSE, kappa 0.48, fewer break pixels (2.35% against 3.62%). SCL 7 (2.13% of observations), with SCL 2 on small water bodies, sends 1,355 of the cell's 3,983 water pixels to InsufficientClear (2,545 decoded as fill, 587 as clear). |

High aerosol is flagged on about half the HLS observations; treating it as cloud would remove another 16.5% (L30) and 16.8% (S30) of what QA keeps, which is why the presets ignore it. Five of the sample's pixels (four in all three products, a fifth in the two HLS products) are pinned with every preset's output in `tests/python/fixtures/t16sgc_2020_2023` (`tests/python/test_real_fixtures.py`).

## Adding a sensor

A sensor is one `SensorProfile` call; `python/ccdcpp/presets.py` writes its profiles that way. **Not shipped:** this Landsat 8/9 Collection 2 profile with the coastal band shows the shape; its values are not validated.

```python
from ccdcpp import CELSIUS_OVER_100, REFLECTANCE, SensorProfile

def landsat_oli_c2():
    return SensorProfile(
        name="landsat_oli_c2",
        band_names=("SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "ST_B10"),
        roles=("coastal", "blue", "green", "red", "nir", "swir1", "swir2", "thermal"),
        units=REFLECTANCE,
        thermal=CELSIUS_OVER_100,
        qa_table_builder="landsat_c2_qa_pixel",
    )

oli = landsat_oli_c2()
h = oli.input()
shared = ccdcpp.config("ccdc").resolve(h)["shared"]
print(h.bands(), shared["detection_bands"], h.index("thermal"))
```

```text
8 [2, 3, 4, 5, 6] 7
```

- **Roles**: `blue`, `green`, `red`, `nir`, `swir1`, `swir2`, `thermal`, `coastal`, `red_edge_1`, `red_edge_2`, `red_edge_3`, `nir_broad` and `other` (`ccdcpp.ROLES`). `nir` is the 865 nm band (TM/ETM+ band 4, OLI band 5, Sentinel-2 B8A); the broad 842 nm B08 is `nir_broad`. Each role but `other` names at most one band; `other` bands are fitted and reported but no band list can name them.
- **Units and thermal**: `units` is a `ccdcpp.Units` (`scale`, `offset`; `ccdcpp.DN` and `ccdcpp.REFLECTANCE`), `thermal` a `ccdcpp.Thermal` (`scale`, `offset`, `at_100c`; `ccdcpp.KELVIN_TENTHS`, `ccdcpp.CELSIUS_OVER_100`) or `None`, `saturation_code` and `thermal_saturation_code` the values a saturated observation holds in the reflective and the thermal rows, as stored, or `None`.
- **QA**: `qa_table_builder`, a `ccdcpp.qa_tables` builder name or `None` for codes; `qa_table_options`, keyword arguments for it of exactly the types `ccdcpp.QA_TABLE_OPTIONS` lists.
- A profile that breaks a layout rule (an unknown or repeated role, no green band, a thermal band without `thermal=`) raises `ValueError` when it is constructed.

# Configurations and presets

ccdcpp runs one fixed machine, the CCDC phases and routes, whose decisions are made by **stages**: the procedure a pixel takes, its noise, where the search for a stable start begins, the start window's outlier screen, fit and stability rule, Lookback, the monitoring model and when it is refitted, the change test, the outlier rule, and the fits a segment reports. Each stage is decided by a **strategy**, chosen by name, with parameters. A **configuration** holds the **shared** values several stages read, for every stage its strategy and that strategy's parameters, and for every procedure its **preprocessing** chain. A run without one uses the preset `ccdc`: PyCCD's algorithm with ccdcpp's corrections, each strategy and step at its defaults, on any input.

## The rule

| | Example |
|---|---|
| The input says only how to read the data. | `ccdcpp.Input()` is Collection-1 DN: reflectance x 10000 (`reflectance_scale` 10000), thermal Kelvin x 10 (`thermal_scale` 10, `thermal_offset` -27315). |
| The configuration says what to do with it, in physical units. | `ccdc`'s `band_range` keeps reflectance in (0, 1), its `thermal_range` -93.2 to 70.7 °C, its `lasso.alpha` is 1e-4 reflectance. |
| The run translates once: 0–1 reflectance × 10000 = the 0–10000 DN check, exactly as today. | `plan_of("ccdc", ccdcpp.Input())` tests `0 < x < 10000` on DN, and on `presets.landsat_c1_physical()` `0 < x < 1`. |

```python
dn = ccdcpp.Input()
print(dn.reflectance_scale, dn.thermal_scale, dn.thermal_offset)

band_range = ccdcpp.config("ccdc").preprocess("standard")["steps"][2]
print(band_range["step"], band_range["reflectance"])

print(ccdcpp.plan_of("ccdc", dn).standard.chain[2])
print(ccdcpp.plan_of("ccdc", presets.landsat_c1_physical()).standard.chain[2])
```

```text
10000.0 10.0 -27315.0
band_range [0.0, 1.0]
<Step BandRange bands [0, 1, 2, 3, 4, 5], 0 < x < 10000>
<Step BandRange bands [0, 1, 2, 3, 4, 5], 0 < x < 1>
```

A configuration's values are physical, so one configuration means the same on every input: reflectance levels (`band_range`'s `reflectance` range) resolve as `r * reflectance_scale + reflectance_offset`, reflectance differences (the shared `lasso.alpha` and `lasso.coef_floor`, `green_median`'s offset) as `r * reflectance_scale`, degrees Celsius `c` as `c * 100 * thermal_at_100c / 10000`, probabilities to chi-squared thresholds for the detection-band count, roles to rows and QA code names to codes. A run resolves its configuration against its input once: `ccd::Algorithm`, which a `Config` holds, caches the resolution of each input it meets (the 16 used last). `cfg.resolve(input)` shows every resolved value and `ccdcpp.plan_of(cfg, input)` the resolved preprocessing.

```python
cfg = ccdcpp.config("ccdc")
print(cfg.resolve(ccdcpp.Input())["model"]["lasso.alpha"])            # 1e-4 x 10000
print(cfg.resolve(presets.landsat_c2().input())["model"]["lasso.alpha"])
```

```text
1.0
0.0001
```

## Changing a configuration

A `Config` is immutable and checked and compiled when it is built. `replace(shared__name=value)` and `replace(stage__parameter=value)` return a changed copy, and `replace(stage="strategy")` swaps a stage's strategy, its parameters starting from the new strategy's defaults. `detect` and `detect_cube` take it as `config=`:

```python
c2 = presets.landsat_c2()
base = ccdcpp.config("ccdc")
stricter = base.replace(shared__change_probability=0.999, refit__factor=2.0)
out2 = ccdcpp.detect(dates, spectral, qas, input=c2, config=stricter)
print(base.hash() != stricter.hash(), stricter.shared()["change_probability"], stricter.params("refit"))
```

```text
True 0.999 {'min_obs': 24, 'factor': 2.0}
```

Mistakes are refused with `ValueError` naming the stage, strategy and parameter, before any pixel runs:

```python
for bad in (lambda: base.replace(refit__fctor=2.0),
            lambda: base.replace(refit__factor=0.0),
            lambda: base.replace(change_test__change_threshold=10.0),
            lambda: base.replace(shared__min_observations=0)):
    try:
        bad()
    except ValueError as e:
        print(e)
```

```text
refit 'span-factor' has no parameter 'fctor'; it has min_obs, factor
refit 'span-factor' parameter 'factor' is 0; it must be greater than 0
change_test 'chi2' parameter 'change_threshold' is the shared value 'change_probability'; set shared__change_probability instead
shared parameter 'min_observations' is 0; it must be at least 1
```

`ccdcpp.strategies()` lists every stage's strategies and their parameters, with unit, bounds, default and meaning; `ccdcpp.shared_values()` the shared values and `ccdcpp.steps()` the preprocessing steps:

```python
for p in ccdcpp.strategies()["refit"]["span-factor"]:
    print(p["name"], "|", p["unit"], "|", p["bounds"], "|", p["default"], "|", p["doc"])
```

```text
min_obs | observations | at least 1 | 24 | observations up to which every test refits
factor | none | greater than 0 | 1.33 | refit once the window spans this many times the last fit's span
```

`cfg.hash()` is a digest equal for equal configurations on any platform, `cfg.to_toml()` writes the configuration as a preset file, and a `Config` pickles, so it can go to worker processes. A preset name passed as `config=` is built once and shared; a `Config` you build once and pass to every call costs nothing per call. In C++ a configuration is a `config::Config`, compiled into a `ccd::Algorithm` ([The C++ API](#the-c-api)). [docs/config.md](docs/config.md) covers configurations from Python and C++, experiments and writing a strategy.

## Shared values

A value several stages read is written once, in `shared`, and the stages that read it hold no copy of it: a stage that sets one is refused, naming the shared value to set instead.

| Shared value | `ccdc` | Unit | Read by |
|---|---|---|---|
| `detection_bands` | `["green", "red", "nir", "swir1", "swir2"]` | roles | the bands every test reads; their count is the chi-squared degrees of freedom |
| `change_probability` | `0.99` | probability | `change_test`, `stability_rule` and `lookback`: their change threshold is its chi-squared quantile |
| `outlier_probability` | `0.999999` | probability | `outlier`'s and `lookback`'s outlier threshold, its chi-squared quantile |
| `lasso.alpha`, `lasso.coef_floor` | `1e-4`, `1e-4` | reflectance | the Lasso of `model`, `stability_fit` and `edge_segments` |
| `lasso.max_iter`, `lasso.tolerance` | `1000`, `1e-4` | none | the same three |
| `coefficients.min`, `.mid`, `.max`, `.per` | `4`, `6`, `8`, `3` | none | the coefficient rule of `model` and `stability_fit`; `edge_segments` fits `coefficients.min`, and writes a Start or End segment only on more observations than the peek and than `coefficients.min` |
| `min_observations` | `12` | observations | the fewest observations a curve is fitted on: `start_search`'s start window, `edge_segments`' whole-series curve; more than `coefficients.min` |
| `statistics_end` | `"none"` | a date | the last date the QA statistics, the cadence, the variogram and the green median read: `"none"`, `"YYYY-MM-DD"` or an ordinal day |

**`ccdc` has no statistics cutoff**: the statistics read the whole series. PyCCD and LCMAP's 2018 production baseline stop them at 2017-12-31, ordinal 736694; `ccdcpp.config("ccdc").replace(shared__statistics_end=736694)` runs that cutoff. `shared__statistics_end="2017-12-31"` is the same date, and `"none"` no cutoff.

## Stages and strategies

| Stage | Decides | `ccdc`'s strategy |
|---|---|---|
| `mode` | the pixel's procedure from its QA statistics | `pyccd` (`clear_pct_threshold` 0.25, `snow_pct_threshold` 0.75) |
| `noise` | each band's noise | `variogram` (`min_lag_days` 30) |
| `start_search` | where a stable start is searched for | `pyccd` (`day_delta` 365; the shared `min_observations`) |
| `window_screen` | the outliers of a candidate start window | `tmask` (`bands` green and swir1, `t_const` 4.89) |
| `stability_fit`, `stability_rule` | the start window's fit and whether it is stable | `lasso`, `pyccd` |
| `lookback` | whether an observation before a stable start joins it | `pyccd` |
| `model`, `refit` | the monitoring model and when it is refitted | `lasso`, `span-factor` (`min_obs` 24, `factor` 1.33) |
| `change_test` | whether a peek is a change | `chi2` (`peek_size` 6, `peek_reference_days` 16, `model_rmse_up_to` 24, `seasonal_count` 24, `seasonal_divisor` 4) |
| `outlier` | whether a peek's first observation is an outlier | `floored` |
| `segment_report`, `edge_segments` | the fits a segment reports; the Start, End and whole-series curves | `model-fit`, `lasso` |

[docs/ccdc-state-machine.md](docs/ccdc-state-machine.md#stages-and-strategies) has what each stage decides and each strategy's rule. The public library has one strategy per stage; a C++ build extension registers more ([docs/config.md](docs/config.md#experiments)).

## TOML presets

A preset is a TOML file: a `[shared]` table (with its tables `[shared.coefficients]` and `[shared.lasso]`), one `[preprocess.<procedure>]` table per procedure, and one table per stage, in the order a pixel meets them, holding the stage's `strategy` and beside it the strategy's parameters. Each table says what it decides and each value its unit, as comments. `python/ccdcpp/configs/ccdc.toml` is `ccdc` written out (`Config.to_toml()`); [docs/config.md](docs/config.md#presets) has the layout. `ccdcpp.config(name)` returns a preset by name (built once); `ccdcpp.config(path)` reads a file, and a file may `extends = "ccdc"` and list only what differs:

```toml
# denser.toml
extends = "ccdc"

[shared]
min_observations = 24

[refit]
min_obs = 48
```

```python
denser = ccdcpp.config("denser.toml")
print(denser.shared()["min_observations"], denser.params("refit"), denser.strategy("refit"))
print(denser == ccdcpp.config("ccdc").replace(shared__min_observations=24, refit__min_obs=48))
```

```text
24 {'factor': 1.33, 'min_obs': 48} span-factor
True
```

## Cadence parameters

The Standard procedure has parameters that assume something about the revisit interval of the input or the false-alarm rate of the change test. They are values of the configuration, each defaulting to pyccd's value, tuned for 16-day Landsat. A denser sensor's cadence is a configuration. Where pyccd has one constant that two stages hold ("Also in"), the `ccdc` preset sets them alike, and a caller changing one sets the other to match, or apart on purpose.

| Parameter | Default | Unit | Also in | Read by |
|---|---|---|---|---|
| `change_test__peek_size` | `6` | observations | | the peek at the reference cadence; the stability rule and Lookback use the test's peek (their `peek` is `"change_test"`) |
| `change_test__peek_reference_days` | `16.0` | days | | the peek size: `round(peek_size * peek_reference_days / median gap)`, never below `peek_size` |
| `shared__change_probability` | `0.99` | probability | | the change threshold at a peek of `peek_size` is its chi-squared quantile for the detection-band count (15.0863 for five), and a longer peek's the quantile at `1 - (1 - change_probability) ** (peek_size / peek)` |
| `shared__outlier_probability` | `0.999999` | probability | | the outlier threshold is its chi-squared quantile (35.8882 for five bands) |
| `noise__min_lag_days` | `30` | days | | the variogram: the smallest lag whose most common separation exceeds this, over the pairs further apart than this |
| `refit__min_obs` | `24` | observations | `change_test__model_rmse_up_to` | Extend: refit while the window holds at most this many; such a window compares against the model RMSE, a longer one against the seasonal RMSE |
| `refit__factor` | `1.33` | ratio | | Extend: refit once the window spans at least this times the span of the last fit |
| `change_test__seasonal_count` | `24` | observations | | the seasonal RMSE: residuals of this many fit observations closest in day of year to the last peek observation |
| `change_test__seasonal_divisor` | `4.0` | none | | the seasonal RMSE: their norm divided by this. pyccd's 4 is `sqrt(24 - 8)`; it is not derived from `seasonal_count` |
| `shared__min_observations` | `12` | observations | | the start window, and the observations a start needs after it; the observations a whole-series curve needs |
| `start_search__day_delta` | `365` | days | | the least span of a start window |

`ccdcpp.chi2_quantile(p, df)` is the chi-squared quantile the procedure computes. The change threshold a run resolves is `chi2_quantile(change_probability, 5)` to the last bit for the default five detection bands:

```python
r = ccdcpp.config("ccdc").resolve(ccdcpp.Input())
print(r["change_test"]["change_threshold"] == ccdcpp.chi2_quantile(0.99, 5))
```

```text
True
```

A value outside its parameter's bounds (`ccdcpp.strategies()` lists them) raises `ValueError` before any pixel runs: `peek_reference_days` or `min_lag_days` of zero or less, a count below 1, a `peek_size` below 2, a `min_observations` not above `coefficients.min`, a probability outside (0, 1) or NaN, a factor or divisor of zero or less, a negative or NaN Lasso constant, and coefficient counts that decrease from `coefficients.min` to `coefficients.max` or exceed 8 (the harmonic basis's width).

**Denser cadences.** The peek size follows the cadence: it is `round(peek_size * peek_reference_days / (median gap + 0.001))`, never below `peek_size`, where the gap is between consecutive observations the processing mask keeps, up to the shared `statistics_end`; at 8 days it is 12, at 5 days 19, at 4 days 24. The change threshold falls as the window grows, so that a longer window keeps the default window's false-alarm rate: with the defaults it is 9.24 at a peek of 12, 6.83 at 19 and 5.90 at 24. Both are computed per pixel, as pyccd's `adjustpeek` and `adjustchgthresh` do; a series whose median gap is about 14.8 days or more keeps `peek_size` and its threshold as given. The shared `min_observations` and the refit's `min_obs` are observation counts and are deliberately not scaled with the cadence, as pyccd does not scale them: at a 4–5 day cadence 12 observations are about 50–60 days, where at 16 days they are about half a year. A caller who wants them in time sets them from the cadence, as `denser.toml` above does.

## Preprocessing

Before a procedure fits a pixel it builds the processing mask, the observations it fits: the input's thermal transform of the pixel's private copy where the procedure's chain asks for it, then a chain of filter steps run in order. The configuration's `preprocess` section holds one chain per procedure, `standard`, `snow` (PermanentSnow) and `insufficient_clear`, its steps by name with parameters in physical units; each pixel runs the one of the procedure it takes. `ccdcpp.plan_of(config, input)` is the resolved plan, in the input's codes, rows and units. The steps (`ccdcpp.steps()` lists their parameters):

| Step | Effect |
|------|--------|
| `keep_qa` (`KeepQa(codes)`) | keep only observations whose QA code is in `codes` (names: `ccdcpp.QA_CODE_NAMES`, or ints) |
| `add_qa` (`AddQa(codes)`) | also keep every observation whose code is in `codes`; must be followed by `one_per_date` |
| `thermal_range` (`ThermalRange(celsius, inclusive)`) | keep only observations whose thermal value is in the range `celsius = [min, max]`, degrees Celsius (an end `inclusive = [min, max]` keeps a value equal to it); skipped without a thermal row |
| `band_range` (`BandRange(bands, reflectance, inclusive)`) | keep only observations on which every band (roles, tested in order; by default blue, green, red, nir, swir1 and swir2) is in the range `reflectance = [min, max]`, reflectance (`inclusive` as above); an empty list is refused |
| `green_median` (`GreenMedian(offset, scope)`) | drop observations at or above the median green over the kept observations dated up to the shared `statistics_end`, plus `offset` (reflectance), at the dates `scope` names |
| `one_per_date` (`OnePerDate()`) | keep the first kept observation of each date |

A range keeps `min < x < max`; an inclusive end also keeps a value equal to it, and NaN passes only a range inclusive at both ends. The defaults are exclusive at both ends of both ranges, pyccd's `0 < v < 10000` and `-9320 < v < 7070` on DN: an observation with a reflective band at exactly a bound, or NaN, is dropped. A chain holds at most `CHAIN_CAPACITY` (8) steps.

The green median filter's scope says where it drops: `"stat_window"` (the default, the scope pyccd's filter computes its mask for) only up to `statistics_end`, `"all"` at any date, `"off"` never (what pyccd's code does; see [Differences from pyccd](#differences-from-pyccd)). With no kept observation up to `statistics_end` nothing is dropped.

### The default plan

The `ccdc` configuration holds pyccd's preprocessing as pyccd intends it. On `ccdcpp.Input()` it resolves to:

```python
print(ccdcpp.plan_of("ccdc", ccdcpp.Input()))
```

```text
standard
  transform  band 6 <- x * 10 - 27315
  0  KeepQa codes {0, 1}
  1  ThermalRange band 6, -9320 < x < 7070
  2  BandRange bands [0, 1, 2, 3, 4, 5], 0 < x < 10000
  3  OnePerDate
snow
  transform  none
  0  KeepQa codes {0, 1}
  1  ThermalRange band 6, 1799.5 < x < 3438.5
  2  BandRange bands [0, 1, 2, 3, 4, 5], 0 < x < 10000
  3  OnePerDate
  4  AddQa codes {3}
  5  OnePerDate
  6  ThermalRange band 6, 1799.5 < x < 3438.5
insufficient_clear
  transform  none
  0  KeepQa codes {0, 1}
  1  ThermalRange band 6, 1799.5 < x < 3438.5
  2  BandRange bands [0, 1, 2, 3, 4, 5], 0 < x < 10000
  3  OnePerDate
  4  GreenMedian band 1, drop x >= median(every date) + 400, scope StatWindow
  5  OnePerDate
```

Only `standard` reads `thermal = "celsius"` (through the input's `thermal_scale`/`thermal_offset`, none when that is the identity) and tests the thermal range on the transformed row; the other two read the thermal band in the caller's unit and test it against the same degrees Celsius taken back through the transform: 1799.5 to 3438.5, Kelvin x 10, here. Each chain's thermal range covers every observation its fit uses: `snow` tests it again as its last step, after adding the snow observations back ([docs/units.md](docs/units.md#the-thermal-range)).

### A custom chain

A chain is part of the configuration: `Config.replace()` edits one step's parameter (`preprocess__<procedure>__<step>__<param>`), every step of a type (`preprocess__<step>__<param>`), how a procedure reads the thermal row (`preprocess__<procedure>__thermal="celsius"` or `"stored"`) or a whole chain (`preprocess__<procedure>=ccdcpp.ProcedureChain(...)`); a TOML preset holds `[preprocess.<procedure>]` tables. Here the Standard procedure keeps snow observations too:

```python
from ccdcpp import BandRange, OnePerDate, KeepQa, ProcedureChain, ThermalRange

snowy_cfg = ccdcpp.config("ccdc").replace(preprocess__standard=ProcedureChain(
    steps=(KeepQa(("clear", "water", "snow")), ThermalRange(), BandRange(), OnePerDate()),
    thermal="celsius"))
print(ccdcpp.plan_of(snowy_cfg, ccdcpp.Input()).standard.chain[0])
snowy = ccdcpp.detect(dates, spectral, qas, input=presets.landsat_c2(), config=snowy_cfg)
```

```text
<Step KeepQa codes {0, 1, 3}>
```

Landsat Collection 2 with its thermal band as the raw `ST_B10` DN is an input whose thermal transform converts it, with a configuration that applies the transform on every procedure:

```python
import dataclasses
from ccdcpp import Thermal

# ((DN * 0.00341802 + 149.0) - 273.15) / 100
c2_raw_thermal = dataclasses.replace(presets.landsat_c2(), name="landsat_c2_raw_thermal",
                                     thermal=Thermal(scale=3.41802e-5, offset=-1.2415,
                                                     at_100c=1.0))
everywhere = ccdcpp.config("ccdc").replace(preprocess__snow__thermal="celsius",
                                           preprocess__insufficient_clear__thermal="celsius")

raw = spectral.copy()
raw[6] = (spectral[6] * 100 + 273.15 - 149.0) / 0.00341802   # ST_B10 DN
from_raw = ccdcpp.detect(dates, raw, qas, input=c2_raw_thermal, config=everywhere)
print([m["break_day"] for m in from_raw["models"]] == [m["break_day"] for m in out["models"]])
```

```text
True
```

A configuration is checked when it is built (an unknown step, QA code name or scope, a missing procedure, a chain over capacity) and its plan when it is resolved against an input (a band not below the band count, a role the input lacks, a transformed band also tested by a `band_range` of its procedure, an `add_qa` not followed by a `one_per_date`), with `ValueError` (`std::invalid_argument` in C++) naming the procedure and the step, before any pixel runs.

# Resuming a run

When new observations arrive, a stored run can be continued instead of rerun. The first run asks for a resume record (`record=True`); the update passes the whole series again (the stored observations and the new ones) and the first run's output as `resume=`, and gets the complete result for the whole series:

```python
n = int(np.searchsorted(dates, date(2019, 1, 1).toordinal()))   # the first run: up to 2018

first = ccdcpp.detect_cube(dates[:n], cube[..., :n], cube_qa[..., :n],
                           input=c2, threads=4, record=True)
# Store first's segment columns, first["resume"] (one entry per pixel) and first["resume_header"].

SEGMENT_COLUMNS = ("row", "col", "start_day", "end_day", "break_day", "curve_qa",
                   "observation_count", "change_probability", "intercept", "coefficients",
                   "rmse", "magnitude")
update = ccdcpp.detect_cube(dates, cube, cube_qa, input=c2, threads=4, resume=first)

print(update["counts"])
capped = ccdcpp.config("ccdc").replace(shared__statistics_end=int(dates[n - 1]))
full = ccdcpp.detect_cube(dates, cube, cube_qa, input=c2, config=capped, threads=4)
print(all(np.array_equal(update[k], full[k]) for k in SEGMENT_COLUMNS))
```

```text
{'resumed': 12, 'rerun_input_changed': 0, 'failed': 0}
True
```

- **The result is the full run's, bit for bit**: `detect_cube(resume=...)` gives what `detect_cube` gives on the whole series with the statistics cutoff (the shared `statistics_end`) capped at the first run's series end. With `ccdc`'s no cutoff, the statistics (the procedure choice, the cadence, the variogram, the green median) stay those of the first run until a full rerun re-baselines them; with a cutoff before the first run's end (PyCCD's 2017-12-31), the update is the plain full rerun.
- **What it saves is compute**: each pixel the Standard machine was running continues from where its stored run stopped, its earlier segments kept as stored and its open segment refitted once. The pixels with nothing to continue (PermanentSnow, InsufficientClear and the early exits) are rerun whole and count as resumed, since their result is the rerun's. On a 500 x 500 Landsat cell the annual update takes 0.36 of a full run's time ([Performance](#performance)).
- **A pixel whose stored observations changed** (a reprocessed value, a back-filled scene) is rerun in full, and its status says so (`update["resume"]["status"]`: 0 resumed, 1 rerun because its input changed, 2 failed). A scene the pixel's QA marks fill does not count as a change.
- **A record that does not match the call is refused** (`ValueError`): another record format (`ccdcpp.RESUME_FORMAT`, 2; it changes with the record's layout or meaning) or results version (`ccdcpp.RESULTS_VERSION`, 3; it changes whenever a release changes an output for the same input and configuration, or the default configuration's outputs), another input (its QA table included) or configuration (`resume_header["digest"]` covers both), a series that ends before the stored one, columns that do not fit. Nothing falls back to a full rerun silently.
- The record is plain columns (`ccdcpp.resume_schema()` gives their dtypes); the caller stores them beside its segments.

One pixel is the same with `detect`:

```python
stored = ccdcpp.detect(dates[:n], spectral[:, :n], qas[:n], input=c2, record=True)
more = ccdcpp.detect(dates, spectral, qas, input=c2, resume=stored)
print(more["status"], more["models"] == ccdcpp.detect(dates, spectral, qas, input=c2, config=capped)["models"])
try:
    ccdcpp.detect(dates, spectral, qas, input=c2, config=stricter, resume=stored)
except ValueError as e:
    print(e)
```

```text
resumed True
the resume record was made with another input or configuration
```

# Diagnostics

`trace=True` on `detect`, or `trace_pixels=[(row, col), ...]` on `detect_cube`, also returns the run step by step: every phase transition with the machine's cursor, every Extend change test with its outcome and statistics, every observation a test drops, every segment written and how the pixel ended, one dict per event, ready for `pandas.DataFrame`. The results are the untraced run's, bit for bit.

```python
import pandas as pd

traced = ccdcpp.detect(dates, spectral, qas, input=presets.landsat_c2(), trace=True)
trace = pd.DataFrame(traced["trace"])
print(trace.kind.value_counts().to_dict())
print(trace[trace.kind == "segment"][["day", "day_to", "outcome", "observations"]].to_string())
print(trace[trace.kind == "exit"]["outcome"].item())

cube_traced = ccdcpp.detect_cube(dates, cube, cube_qa, input=presets.landsat_c2(), threads=4,
                                 trace_pixels=[(1, 2)])
print(len(cube_traced["traces"][(1, 2)]))
```

```text
{'transition': 356, 'test': 327, 'segment': 2, 'exit': 1}
        day  day_to                                  outcome  observations
449  730136  734648  curve_qa 8 break 734664 change 1.000000           239
682  734664  737256  curve_qa 8 break 737272 change 0.000000           131
SeriesEnd
686
```

In C++ an observer is `RunSpec::observer` (`ccd::TextObserver` prints the events), and `RunSpec::observe` names a cube's pixels. The hooks compile out with CMake's `-DCCD_OBSERVE=OFF`. See [docs/diagnostics.md](docs/diagnostics.md) for the event fields and the C++ side.

# Build and test

```bash
cmake -S . -B build                    # C++ library, tests and the Python module
cmake --build build --config Release -j8
pip install .                          # the Python package
```

CMake options: `CCD_ENABLE_OPENMP` (ON), `CCD_BUILD_PYTHON` (ON), `CCD_BUILD_TESTS` (ON), `CCD_OBSERVE` (ON; the diagnostics hooks), `CCD_EXTENSIONS` (strategy extension directories; none by default) and `CCD_INSTALL_CPP` (OFF; installs `ccd_core` and the headers). On macOS with Apple clang, CMake finds Homebrew's libomp only when told where it is: add `-DOpenMP_CXX_FLAGS="-Xclang -fopenmp" -DOpenMP_CXX_LIB_NAMES=libomp -DOpenMP_CXX_INCLUDE_DIR=$(brew --prefix libomp)/include -DOpenMP_libomp_LIBRARY=$(brew --prefix libomp)/lib/libomp.dylib` to the configure line.

The tests run in three tiers, from the cheapest:

| Tier | Command | What it runs | When |
|---|---|---|---|
| Quick | `ctest --test-dir build -j8 -LE corpus` | every test, the sweeps over cuts (`PrefixStability.*`, `ResumeEqualsTheFullRun.*`) on their sample shard: every eighth cut, each series' last cut and the cuts through the peek after each break | while changing code |
| Full | `ctest --test-dir build -j8` | every test, every cut | before a change is merged, and after any change to the machine, the fits, the record or resume |
| Release | the full tier on a sanitizer build, below, and on Linux (`tools/linux-compare`) | memory errors, undefined behaviour and asserts; another compiler's arithmetic | before a release, or a change that can behave differently by platform |

The sanitizer build is Debug at `-O2` (asserts on; at `-O0` the sweeps run several hundred times slower than Release):

```bash
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS_DEBUG="-O2 -g" \
  -DCCD_BUILD_PYTHON=OFF \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build build-asan -j8
ctest --test-dir build-asan -j8
```

`Golden.BitIdenticalToFixture`, `Bands.BitIdenticalToFixture` and `Transitions.BitIdenticalToFixture` compare against fixtures written by a Release build on one platform (clang, arm64, macOS), so they fail on the sanitizer build and skip on Linux; every other test passes on both. The labels need CMake 3.22 or later; before it every test runs unlabelled.

# The C++ API

The library is the CMake target `ccd_core`; `#include "ccd/ccd.hpp"` includes every public header, `ccd/run/run.hpp` the two calls. Both calls throw `std::invalid_argument` before any pixel runs on an input, dates or a configuration they refuse; a pixel that throws in `detect_cube` is one of its `failures`.

```cpp
namespace ccd {
PixelOutput detect(const RunSpec& spec, ArrayView<const std::int64_t, 1> dates,
                   const PixelSpectral& spectral, const PixelQa& qas,
                   const Resume* resume = nullptr);
CubeOutput detect_cube(const RunSpec& spec, ArrayView<const std::int64_t, 1> dates,
                       const CubeSpectral& spectral, const CubeQa& qas,
                       int threads = 0, const Resume* resume = nullptr);
}
```

**`ccd::RunSpec`** is what a call runs:

| Field | Default | What it is |
|---|---|---|
| `input` | `ccd::Input{}` (Collection-1 DN, QA codes) | how to read the data, its QA table (`Input::qa_table`) included |
| `algorithm` | `Algorithm::preset("ccdc")` | the compiled configuration |
| `record` | `false` | also return each pixel's resume record and the call's header |
| `masks` | `false` | also return each pixel's processing mask: `detect`'s `result.mask`, `detect_cube`'s `CubeOutput::masks` |
| `observer`, `observe` | none, every pixel | where the observed pixels report each step ([docs/diagnostics.md](docs/diagnostics.md)) |

The values are views of any strides: `PixelSpectral` is a `(B, T)` view of `double`, `float` or `std::uint16_t`, `PixelQa` a `(T)` view of `std::uint8_t` codes or `std::uint16_t` raw QA, and `CubeSpectral`, `CubeQa` the `(H, W, B, T)` and `(H, W, T)` cube forms. `detect` returns a `ccd::PixelOutput`: `result` (a `FitResult`: `models`, and `mask` with `RunSpec::masks`), and with `RunSpec::record` or a resume, `record`, `status` and `header`. `detect_cube` returns a `ccd::CubeOutput`: `pixels` in raster order (each a `ccd::CubePixel`: `row`, `col`, `result`, `record`, `status`), `failures`, `threads`, `header`, the cube's `height`, `width` and `timesteps`, and with `RunSpec::masks` every pixel's mask in one (H, W, T) array, `masks`, read per pixel by `mask(row, col)`.

**A configuration** is a `ccd::config::Config` (`include/ccd/config/config.hpp`): `shared` values, `stages` (a strategy and its parameters per stage) and `preprocess` (a chain per procedure), a value you edit. A `ccd::Algorithm` (`include/ccd/run/algorithm.hpp`) compiles one: checked when it is built, immutable, shared by every call and thread, and resolved and built once per input it meets. `ccd::plan_of(config, input)` is the preprocessing plan it runs on an input, and `ccd::config::resolve(registry, config, input)` (`include/ccd/config/resolve.hpp`) everything it runs there.

```cpp
#include "ccd/ccd.hpp"

namespace cfg = ccd::config;

ccd::RunSpec a_spec(const ccd::Input& input)
{
    cfg::Config config = ccd::strategies::library_registry().preset("ccdc");
    config.shared["statistics_end"] = cfg::integer(736694);         // pyccd's cutoff
    config.shared["change_probability"] = cfg::real(0.995);
    config.stages["refit"].params["factor"] = cfg::real(2.0);
    config.preprocess["standard"].steps[0].params["codes"] = cfg::names({"clear", "water", "snow"});

    ccd::RunSpec spec;
    spec.input = input;
    spec.algorithm = std::make_shared<const ccd::Algorithm>(config);   // checked here
    spec.record = true;
    return spec;
}
```

**Resume.** A stored run is the call's last argument, a `ccd::Resume{header, records, segments}` (`include/ccd/results/record.hpp`): the stored call's header and each pixel's record and segments, in raster order. `detect(spec, dates, spectral, qas, ccd::Resume{header, {record}, {segments}})` resumes a pixel, `detect_cube(spec, dates, cube, qas, threads, &from)` a cube from H × W of each. A resume is refused unless its header's digest is that of the call's input and configuration.

**Extensions.** A new decision rule is a strategy class for its stage's interface (`include/ccd/strategies/stages.hpp`), and a new preprocessing step a step type (`Registry::add_step()`), each registered by name in `ccd::strategies::library_registry()` through a build extension (CMake's `CCD_EXTENSIONS`). [docs/config.md](docs/config.md) shows one, and [docs/ccdc-state-machine.md](docs/ccdc-state-machine.md) the machine they plug into.

# How it works

Per pixel, a run decodes the QA, computes the QA statistics, chooses the pixel's procedure, builds the processing mask (the observations it fits) and fits them. A pixel with enough clear observations takes the **Standard** procedure, CCDC's change detection; a mostly snowy pixel (**PermanentSnow**) and one with too few clear observations (**InsufficientClear**) get one curve over the whole series.

The Standard procedure is a small state machine: a per-pixel record, one transition function per phase, each making one decision and returning the next phase, and one loop that steps it. Bracketed names are why the machine stopped:

```
                              ┌─ size <= meow_size ───────────────────────► Done  [TooFewObservations]
                              ├─ a NaN in the variogram ──────────────────► Done  [VariogramNaN]
   Prepare ───────────────────┼─ no room for a start window ──────────────► EndCatch [NoRoomToInitialize]
                              ▼
       ┌───────────────► Initialize ◄──┐  ShortSpan, AllOutliers, RejectedOutliers: grow
       │                  │    │   └───┘  Unstable, DroppedThenUnstable: shift
       │                  │    └── Exhausted ─────────────────────────────► EndCatch [InitializeExhausted]
       │                  │ Stable, DroppedThenStable
       │                  ▼
       │   window.start > prev_break ?──── no ─────────────┐
       │                  │ yes                            │
       │                  ▼                                │
       │      ┌──► Lookback  Change / Dropped / Included   │
       │      └───────┘  (while window.start > prev_break) │
       │                  │                                │
       │                  ▼                                ▼
       │   no segment yet and window.start - prev_break > peek ?
       │                  │ yes                            │ no
       │                  ▼                                │
       │              StartCatch (the Start segment)       │
       │                  ▼                                ▼
       │   a full peek after the window ? ── no ──────────────────────────► EndCatch [PeekBeyondEnd]
       │                  │ yes
       │                  ▼
       │      ┌──► Extend  Change / Dropped / SeriesEnd / Grown
       │      └───────┘  (while a full peek fits)
       │                  │ series end: segment, change probability 0 ───────────────► EndCatch [SeriesEnd]
       │                  │ Change: segment, change probability 1
       │                  ▼
       └─── room ──── Restart: prev_break = window.stop, window = (stop, stop + meow_size)
                          │ no room
                          ▼
                      EndCatch [NoRoomToInitialize] ── the End segment, if prev_break + peek < size ──► Done
```

`meow_size` is the start window, the shared `min_observations`. The phases and routes are fixed; what a phase decides is a stage, decided by the strategy the configuration names for it ([Configurations and presets](#configurations-and-presets)). Because the record holds everything that survives from one step to the next, a run can stop at a size guard and continue on a longer series ([Resuming a run](#resuming-a-run)), and an observer can report each step ([Diagnostics](#diagnostics)).

- [CCDC as a state machine](docs/ccdc-state-machine.md): the phases, guards and routes, the record, the stages and strategies, resuming, and why the code is shaped this way.
- [Floating-point sums](docs/floating-point-sums.md): why the order of the Lasso solver's sums is part of the output, and how they are made fast.

# Performance

Measured for ccdcpp 0.10.0 on an Apple M5 Max (18 cores: 6 performance, 12 efficiency; 64 GB; macOS 26.7), Release build, Python 3.11, numpy 1.26.4, Collection-1 DN at pyccd's statistics cutoff (`presets.landsat_c2()` for the real cell), on a machine that was not otherwise idle. 1.0 runs a pixel in the same time as 0.10.0 (0.403 ms against 0.404 ms per pixel, 129 MB against 128 MB peak; [docs/benchmarks.md](docs/benchmarks.md#10-against-0100)). Memory is the peak resident set size of the process, including the interpreter and the input it holds. [docs/benchmarks.md](docs/benchmarks.md) has every table, the load during each run, and how to measure again.

**Against PyCCD**, one pixel at a time on one thread, 404 pixels (the synthetic corpus and PyCCD's own test pixels), two rounds:

| | PyCCD with ccdcpp's corrections | PyCCD as released | ccdcpp |
|---|---|---|---|
| median per pixel | 54.6, 52.4 ms | 51.8, 52.5 ms | 0.27 ms |
| total, 404 pixels | 27.4, 27.1 s | 26.7, 26.9 s | 0.15 s |
| same segments as ccdcpp | 404 of 404 | 155 of 404 | |
| peak memory (own process) | 243 MB | | 131 MB |

About 200 times faster per pixel (median), 174-178 times on the total, and 68-73 times on PyCCD's 443-observation reference pixel (192-205 ms against 2.8 ms). Both memory figures are almost all the interpreter, the libraries and the loaded corpus: after the first pixel, the other 403 raise the peak by under 3 MB in either.

**A real cell**: a 500 x 500 Landsat Collection 2 ARD cell (h024v013, 1984-2023, 2473 dates, 250,000 pixels, 455,231 segments) through `detect_cube` in strips of 50 rows:

| Threads | Detect | ms per pixel | Speed-up | Peak memory |
|---|---|---|---|---|
| 1 | 292.4 s | 1.170 | 1.00 | 17.6 GB |
| 8 | 42.4 s | 0.169 | 6.90 | 19.8 GB |
| 18 | 22.8 s | 0.091 | 12.8 | 19.9 GB |

The memory is mostly the input as the bench holds it: the cell read whole (9.9 GB of uint16), plus each strip converted to float64 (3.5 GB); it barely moves with the thread count, and a caller that reads per strip needs less.

**The annual update** of that cell (recorded to 2022-12-31, 89 dates of 2023 added, 8 threads, two rounds): `detect_cube(resume=...)` 14.6 s and 13.9 s against a full run's 40.1 s and 38.7 s, 0.36 of the time, equal to the full run on all 250,000 pixels; 19.9 GB peak for the whole bench (stored run, update and full run in one process).

# Differences from pyccd

ccdcpp recreates pyccd. Where pyccd's code does not do what it evidently intends, ccdcpp does what it intends, so the two differ on the inputs that reach it:

- **RLM convergence.** pyccd's robust fit stops when `not numpy.any(numpy.fabs(x0 - x > tol))` (`ccd/models/robust_fit.py`, `_check_converge`): the parenthesis takes the absolute value of the comparison, not of the difference, so a coefficient that fell by more than `tol` counts as converged. ccdcpp stops when every coefficient moved by less than `tol` in either direction. A fit that pyccd stops early gets different weights and coefficients.
- **Date order.** pyccd orders observations with `np.argsort(dates)`, which is not a stable sort, so which of two observations on the same date survives the duplicate-date filter depends on the sort. ccdcpp takes dates in non-decreasing order (unsorted dates raise `ValueError`) and keeps the first observation of each date.
- **Green median filter.** pyccd's `insufficient_clear_filter` computes the green mask for the observations up to its statistics cutoff and applies it with `standard_mask[standard_mask][:n] &= green_mask` (`ccd/qa.py`), which writes into a copy, so the filter drops nothing. ccdcpp applies it, with the `stat_window` scope by default.
- **Seasonal RMSE ties.** pyccd's `find_closest_doy` (`ccd/change.py`) takes the 24 fit-window observations closest in day of year to the last peek date with `np.argsort(d_yr)[:num]`, which is not a stable sort. Observations a multiple of 1461 days apart are exactly equally distant, so which of the tied observations are kept depends on the sort. ccdcpp breaks ties in window order, so the seasonal RMSE, and with it the change magnitudes, can differ on dense series.

On the synthetic corpus, with pyccd patched for these four (the convergence test as `|x0 - x| > tol`, a stable date sort, the green mask written through `standard_mask[np.flatnonzero(standard_mask)[:n]] &= green_mask`, and `np.argsort(d_yr, kind="stable")` in `find_closest_doy`), ccdcpp at pyccd's statistics cutoff, with the Standard procedure as pyccd's, matches pyccd to 1e-6 on every InsufficientClear pixel (175 in DN) and on every pixel where a reflective band value on a bound reaches the saturation filter (106, and the golden test's bounds pixel); with the green filter left unpatched, the `off` scope matches on the same 175. One more difference is kept on purpose: an undefined raw QA value decodes to fill where pyccd raises (see [QA input](#qa-input)).

Two defaults of `ccdc` differ from pyccd by choice:

- **The thermal range outside Standard.** pyccd's PermanentSnow and InsufficientClear filters test their untransformed thermal band against the Celsius bounds: on Collection-1 DN (Kelvin x 10 against -9320 and 7070) that drops saturated and NaN values but keeps implausible temperatures, and on any other input it tests the wrong unit. `ccdc` tests those chains' stored thermal band against the bounds taken back through the input's thermal transform, so the range is -93.2 to 70.7 °C in every procedure on every input, over every observation the procedure's fit uses (the snow chain's snow observations too, which pyccd does not test), and a physical-unit run segments as the DN run does ([docs/units.md](docs/units.md#the-thermal-range)).
- **The statistics cutoff.** pyccd stops the QA statistics, the cadence, the variogram and the green median at 2017-12-31 (`STAT_ORD`), LCMAP's 2018 production baseline; `ccdc` reads the whole series. `shared__statistics_end=736694` is pyccd's cutoff.

The Standard procedure (`ccd/procedures.py`) is pyccd's with five such corrections, each checked against Zhu's MATLAB reference, TrendSeasonalFit, and each in one function of the state machine (see [CCDC as a state machine](docs/ccdc-state-machine.md)):

- **Lookback's peek far from the break.** `slice(start - 1, start - peek, -1)` (line 753) reads one observation fewer than the peek; ccdcpp reads the peek, as Extend does after the window and TrendSeasonalFit does both ways.
- **Lookback's peek near the break.** The guard against reading past the start of the series (`start - peek <= 0`, line 755) also fires after a break, where it reads the previous segment's observations; ccdcpp reads down to the previous break and never below, as TrendSeasonalFit does. It takes an adjusted peek of 13 or more, a median gap of about 7.7 days or less (HLS, Sentinel-2), never Landsat's 16 days.
- **Coefficient counts.** Initialize fits a literal 4 coefficients and the peeks predict with a literal 8-term basis; ccdcpp reads the shared `coefficients.min` and `coefficients.max` (TrendSeasonalFit's `min_num_c`, `max_num_c`). The same at the defaults.
- **The series end.** The check before Extend's grow (line 691) tests `>` where its comment, and pyccd's own `>=` before it changed its loop guard, mean the window to stop before the grow, so the check never holds and the last curve grows once more: its end day is its break day. ccdcpp stops at the last full peek, so a series-end segment ends one observation earlier, before its break day (as after an outlier drop). That curve is the last: pyccd restarts after it, and at peeks of 26 or more runs Initialize on the last observations and drops observations no segment uses; ccdcpp goes to the End segment's test, as TrendSeasonalFit ends.
- **Refit at 24 observations.** Extend refits while the window holds fewer than 24 observations (line 632) but compares against the model RMSE while it holds at most 24 (line 646), so at exactly 24 it compares an older fit's RMSE; ccdcpp refits at most 24, as TrendSeasonalFit does with one `<=` for both.

The measured agreement with PyCCD, as released and with exactly these corrections applied, is in [docs/benchmarks.md](docs/benchmarks.md#against-pyccd).

---

# License and Attribution

`ccdcpp` is released into the public domain under the **Unlicense**.

This project is a C++17 translation of the USGS LCMAP PyCCD software.
The original PyCCD implementation was also released into the public domain.

The goals of this project are to:

- preserve PyCCD's algorithm, doing what its code intends where the two differ
- reproduce PyCCD's outputs except where [Differences from pyccd](#differences-from-pyccd) says otherwise
- provide a high-performance native C++ implementation
- enable Python interoperability through pybind11

Minor implementation changes were necessary to accommodate modern NumPy versions and to adapt portions of the Python implementation into efficient C++.

---

# Independent Development

`ccdcpp` is an independent software project developed separately from the U.S. Geological Survey (USGS).

This project was created independently for research, software engineering, and performance optimization of the publicly available CCD algorithm.

The author is not affiliated with, employed by, contracted by, or acting on behalf of the USGS in the development of this project. The USGS did not commission, fund, review, endorse, or provide technical support for this implementation.

The relationship to PyCCD is limited to algorithmic translation and validation against the publicly available implementation. All software architecture, memory management strategies, parallelization, optimizations, and implementation decisions in `ccdcpp` are original work developed independently for this project.
