"""What the bench scripts share: the build under test, their inputs, the timing and the peak memory.

Each script puts its own directory first on sys.path before `import _common`, since `python -I`
leaves it off.
"""
import datetime as dt
import os
import resource
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

#: pyccd's statistics cutoff, 2017-12-31, as an ordinal day.
PYCCD_STATISTICS_END = 736694

#: The bench cell's corner in the cube of bench_cell.py: rows 4000-4499, cols 1500-1999.
CELL_ROW, CELL_COL, CELL_SIZE = 4000, 1500, 500


def import_build(module_dir: str) -> None:
    """Imports ccdcpp from `module_dir`, and exits if it loaded from anywhere else: a wrong
    directory would silently time whatever ccdcpp the interpreter has installed. The caller's
    `import ccdcpp` then gets this module."""
    sys.path.insert(0, module_dir)
    import ccdcpp
    if not os.path.realpath(ccdcpp.__file__).startswith(os.path.realpath(module_dir) + os.sep):
        sys.exit(f"{sys.argv[0]}: ccdcpp loaded from {ccdcpp.__file__}, not from {module_dir}")


@dataclass
class Corpus:
    """The synthetic corpus as `golden_driver gen` writes it."""
    dates: NDArray[np.int64]       # (T,)
    spectral: NDArray[np.float64]  # (H, W, 7, T), read-only
    qas: NDArray[np.uint8]         # (H, W, T), read-only


def read_corpus(path: str) -> Corpus:
    """Reads a corpus file: little-endian uint64 H, W, T; int64 dates[T]; float64
    spectral[H][W][7][T]; uint8 qas[H][W][T]."""
    with open(path, "rb") as f:
        H, W, T = (int(v) for v in np.frombuffer(f.read(24), dtype=np.uint64))
        dates = np.frombuffer(f.read(8 * T), dtype=np.int64).copy()
        spectral = np.frombuffer(f.read(8 * H * W * 7 * T), dtype=np.float64).reshape(H, W, 7, T)
        qas = np.frombuffer(f.read(H * W * T), dtype=np.uint8).reshape(H, W, T)
    return Corpus(dates, spectral, qas)


@dataclass
class Cell:
    """The bench cell as stored: uint16 Landsat C2 ARD."""
    dates: NDArray[np.int64]  # (T,) ordinal days
    sr: NDArray[np.uint16]    # (N, N, 7, T): SR_B1..B6 and QA_PIXEL
    st: NDArray[np.uint16]    # (N, N, T): ST


def read_cell(cube: str, size: int) -> Cell:
    """Reads the `size` x `size` corner of the bench cell from `cube`, a zarr 3 group holding
    `landsat-c2ard-sr`, (rows, cols, 7, T) uint16 SR_B1..B6 and QA_PIXEL, and
    `landsat-c2ard-bt`, (rows, cols, 1, T) uint16 ST, with the ISO dates in the `time`
    attribute of `landsat-c2ard-sr`."""
    import zarr  # pyright: ignore[reportMissingImports]  # the cell benches' interpreter has it

    g = zarr.open_group(cube, mode="r")
    sr, bt = g["landsat-c2ard-sr"], g["landsat-c2ard-bt"]
    dates = np.array([dt.date.fromisoformat(t).toordinal() for t in sr.attrs["time"]], np.int64)
    r1, c1 = CELL_ROW + size, CELL_COL + size
    return Cell(dates, np.asarray(sr[CELL_ROW:r1, CELL_COL:c1, :, :]),
                np.asarray(bt[CELL_ROW:r1, CELL_COL:c1, 0, :]))


def c2_strip(cell: Cell, r0: int, r1: int) -> tuple[NDArray[np.float64], NDArray[np.uint16]]:
    """Rows r0..r1-1 of the cell in presets.landsat_c2()'s units: (rows, N, 7, T) float64 spectral
    (nodata out of range) and the (rows, N, T) QA_PIXEL."""
    sp = np.empty((r1 - r0, cell.sr.shape[1], 7, cell.dates.size), np.float64)
    opt = cell.sr[r0:r1, :, :6]
    sp[:, :, :6] = opt * 2.75e-5 - 0.2
    sp[:, :, :6][opt == 0] = -1.0
    t = cell.st[r0:r1]
    sp[:, :, 6] = (t * 0.00341802 + 149.0 - 273.15) / 100.0
    sp[:, :, 6][t == 0] = -9.0
    return sp, np.ascontiguousarray(cell.sr[r0:r1, :, 6])


def best_of(reps: int, call: Callable[[], object]) -> float:
    """The shortest wall time of `reps` calls, in seconds."""
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        call()
        best = min(best, time.perf_counter() - t0)
    return best


def peak_mb() -> float:
    """The process's peak resident set size, in MB of 10^6 bytes as /usr/bin/time reports it.
    ru_maxrss is bytes on macOS and kilobytes on Linux."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r * 1024 / 1e6
