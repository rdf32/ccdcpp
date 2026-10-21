# Diagnostics: what a run does, step by step

A run can report each step of the pixels it observes: every phase transition of the machine
with where its cursor is, every Extend change test with its outcome and statistics, every
observation a test drops, every segment it writes, and how the pixel ends. Nothing about the
run's output changes: an observed run's results are the unobserved run's, bit for bit.

## From Python

`detect(..., trace=True)` returns `"trace"`, one dict per event, ready for `pandas.DataFrame`.
The example is the README's [quick-start pixel](../README.md#one-pixel):

```python
import pandas as pd
import ccdcpp

out = ccdcpp.detect(dates, spectral, qas, encoding=ccdcpp.presets.landsat_c2(), trace=True)
trace = pd.DataFrame(out["trace"])
print(trace[["kind", "phase", "next", "window_start", "window_stop"]].head(3).to_string())

# Where the change tests fired, and their statistics (one per peek observation):
change = trace[(trace.kind == "test") & (trace.outcome == "change")]
print(change.day.tolist(), [round(v, 1) for v in change["values"].iloc[0]])

# The segments, as the run wrote them:
print(trace[trace.kind == "segment"][["day", "day_to", "observations"]].to_string())
print(out["models"] == ccdcpp.detect(dates, spectral, qas, encoding=ccdcpp.presets.landsat_c2())["models"])
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
cols = ccdcpp.detect_cube(dates, cube, cube_qa, encoding=ccdcpp.presets.landsat_c2(), threads=4,
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
| `day`, `outcome`, `values` | a test: the peek's first day, `change` or `no change`, each peek observation's value of the change test's first statistic (chi2's `change_magnitude`); a drop: the dropped day and why (`outlier`, `lookback outlier`) |
| `day`, `day_to`, `outcome`, `observations` | a segment: its first and last day, its curve_qa, break day and change probability, its observation count |
| `outcome`, `observations` | the exit: why the pixel ended (`SeriesEnd`, `InitializeExhausted`, ...), and its segment count |

The whole-series pixels (insufficient clear, permanent snow) take no machine and report
nothing. A resumed pixel reports from its Prepare on, then from where the stored run stopped.

## From C++

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
    spec.encoding = input;             // ccd::Input, the data's encoding
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
