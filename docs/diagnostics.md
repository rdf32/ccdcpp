# Diagnostics: what a run does, step by step

A run can report, beside its output, why each observation is where the run left it (its
reason code), and each step of the pixels it observes: every phase transition of the machine
with where its cursor is, every Extend change test with its outcome and statistics, every
observation a stage drops, every segment it writes, and how the pixel ends. Nothing about the
run's output changes: a run with either has the results of the run without, bit for bit.

## Reason codes

`detect(..., reasons=True)` also returns `"mask_reasons"`, one uint8 code per observation;
`detect_cube(..., reasons=True)` returns every pixel's in one `(H, W, T)` array, as `masks=True`
does the processing masks, and `ccdcpp.to_pixels()` gives each pixel its row. `ccdcpp.REASONS`
names the codes; those below `ccdcpp.FIRST_DROP` are the observations the processing mask keeps,
so `processing_mask == (mask_reasons < ccdcpp.FIRST_DROP)`.

| code | name | the observation |
|---|---|---|
| 0 | `kept` | is kept and in a segment |
| 1 | `no_fit` | is kept, after the last segment of a pixel whose model could not be fitted on its window (exit `NoFit`) |
| 2 | `no_segment` | is kept and in no segment otherwise: the last peek after a curve that reaches the series end, an edge segment that is not due, the gap a Lookback stopped by a change leaves after a break, a pixel with too few observations or a NaN variogram |
| 3 | `qa` | was removed by `keep_qa`: its QA code is not kept |
| 4 | `thermal` | was removed by `thermal_range` |
| 5 | `band_range` | was removed by `band_range` |
| 6 | `green` | was removed by `green_median` (the insufficient-clear procedure, or a mode's extra step) |
| 7 | `duplicate_date` | was removed by `one_per_date`: a later observation of its date |
| 8 | `custom_step` | was removed by a step a registry adds |
| 9 | `window_screen` | was dropped by Initialize's window screen (Tmask) |
| 10 | `lookback_outlier` | was dropped by Lookback as an outlier before the stable window |
| 11 | `outlier` | was dropped by Extend's outlier rule |
| 12 | `before_change` | was dropped at a change, before the change test's break (a test that breaks past the peek's first observation) |
| 13 | `stored` | was dropped by the run a resume continues: the record keeps its day, not why |

A preprocessing code is the last step that removed the observation (an `add_qa` can take one
back). The README's [quick-start pixel](../README.md#one-pixel) (its arrays and imports): the
clouds, and the last peek after the curve that reaches the series end, which no segment holds:

```python
out = ccdcpp.detect(dates, spectral, qas, input=presets.landsat_c2(), reasons=True)
codes = out["mask_reasons"]
names, counts = np.unique(np.array(ccdcpp.REASONS)[codes], return_counts=True)
print(dict(zip(names.tolist(), counts.tolist())))
print(np.array_equal(codes < ccdcpp.FIRST_DROP, out["processing_mask"] == 1))
print([date.fromordinal(int(d)) for d in dates[codes == ccdcpp.REASONS.index("no_segment")]])
```

```text
{'kept': 370, 'no_segment': 6, 'qa': 81}
True
[datetime.date(2019, 8, 1), datetime.date(2019, 8, 17), datetime.date(2019, 9, 2), datetime.date(2019, 10, 4), datetime.date(2019, 11, 5), datetime.date(2019, 11, 21)]
```

A cube pixel that failed has `len(ccdcpp.REASONS)` (no code) throughout. In C++ the codes are
`ccd::Reason` (`ccd/input/processing_mask.hpp`), asked for with `RunSpec::reasons`:
`FitResult::reasons` for `detect`, `CubeOutput::reasons` and `reasons_of(row, col)` for
`detect_cube`. Without them the run writes none: the preprocessing notes each step's removals
and the machine its drops only when asked.

## The trace from Python

`detect(..., trace=True)` returns `"trace"`, one dict per event, ready for `pandas.DataFrame`.
The example is the README's [quick-start pixel](../README.md#one-pixel):

```python
import pandas as pd
import ccdcpp

out = ccdcpp.detect(dates, spectral, qas, input=ccdcpp.presets.landsat_c2(), trace=True)
trace = pd.DataFrame(out["trace"])
print(trace[["kind", "phase", "next", "window_start", "window_stop"]].head(3).to_string())

# Where the change tests fired, and their statistics (one per peek observation):
change = trace[(trace.kind == "test") & (trace.outcome == "change")]
print(change.day.tolist(), [round(v, 1) for v in change["values"].iloc[0]])

# The segments, as the run wrote them:
print(trace[trace.kind == "segment"][["day", "day_to", "observations"]].to_string())
print(out["models"] == ccdcpp.detect(dates, spectral, qas, input=ccdcpp.presets.landsat_c2())["models"])
```

```text
         kind       phase        next  window_start  window_stop
0  transition     Prepare  Initialize             0           12
1  transition  Initialize  Initialize             0           13
2  transition  Initialize  Initialize             0           14
[734664] [132.2, 95.0, 69.5, 71.1, 63.7, 53.6]
        day  day_to  observations
449  730136  734648           239
682  734664  737256           131
True
```

For a cube, name the pixels; `"traces"` holds each one's events:

```python
cols = ccdcpp.detect_cube(dates, cube, cube_qa, input=ccdcpp.presets.landsat_c2(), threads=4,
                          trace_pixels=[(1, 2), (2, 3)])
print(sorted(cols["traces"]), pd.DataFrame(cols["traces"][(2, 3)]).kind.iloc[-1])
```

```text
[(1, 2), (2, 3)] exit
```

Each event is a dict:

| field | for |
|---|---|
| `kind` | `transition`, `test`, `drop`, `segment` or `exit` |
| `machine`, `phase`, `next` | a transition: the machine and the phase it leaves and enters |
| `window_start`, `window_stop`, `prev_break`, `observations` | a transition: the cursor (positions in the masked series) and the series length |
| `day`, `outcome`, `values` | a test: the peek's first day, `change` or `no change`, each peek observation's value of the change test's first statistic (chi2's `change_magnitude`); a drop: the dropped day and why, its reason code's name (`window_screen`, `lookback_outlier`, `outlier`, `before_change`) |
| `day`, `day_to`, `outcome`, `observations` | a segment: its first and last day, its curve_qa, break day and change probability, its observation count |
| `outcome`, `observations` | the exit: why the pixel ended (`SeriesEnd`, `InitializeExhausted`, ...), and its segment count |

The whole-series pixels (insufficient clear, permanent snow) take no machine and report
nothing. A resumed pixel reports from its Prepare on, then from where the stored run stopped.

## The trace from C++

An observer is `RunSpec::observer`; `ccd::TextObserver` prints each event as one line.
`RunSpec::observe` names a cube's observed pixels by flat index (`row * W + col`; empty, the
default, observes every pixel); `detect` observes its one pixel whatever it holds.

```cpp
#include <iostream>

#include "ccd/results/diagnostics.hpp"
#include "ccd/run/run.hpp"

void observed(const ccd::Input& input, ccd::ArrayView<const std::int64_t, 1> dates,
              ccd::ArrayView<const double, 2> spectral, ccd::ArrayView<const std::uint8_t, 1> qas,
              ccd::ArrayView<const double, 4> cube, ccd::ArrayView<const std::uint8_t, 3> cube_qa)
{
    ccd::RunSpec spec;
    spec.input = input;                // ccd::Input, how to read the data
    spec.observer = std::make_shared<ccd::TextObserver>(std::cerr);
    ccd::detect(spec, dates, spectral, qas);

    const ccd::index_t W = cube.shape()[1];
    spec.observe = {1 * W + 2};        // a cube run: only pixel (1, 2)
    ccd::detect_cube(spec, dates, cube, cube_qa, 8);
}
```

Its lines read like these:

```text
pixel 0: transition Prepare -> Initialize window (0, 12) prev_break 0 n 392
pixel 0: transition Initialize -> Initialize window (0, 13) prev_break 0 n 392
```

An `Observer` implements one call, `event(pixel, const ObserverEvent&)`; in a cube run it is
called from several threads at once.

## Building without it

The hooks are compiled in with CMake's `CCD_OBSERVE` (ON by default). A pixel without an
observer then costs one pointer test per hook. `-DCCD_OBSERVE=OFF` removes them: an observer is
never called and `trace` comes back empty.
