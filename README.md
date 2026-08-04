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