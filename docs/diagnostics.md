# Diagnostics: what a run does, step by step

A run can report each step of the pixels it observes: every phase transition of the machine
with where its cursor is, every Extend change test with its outcome and statistics, every
observation a test drops, every segment it writes, and how the pixel ends. Nothing about the
run's output changes: an observed run's results are the unobserved run's, bit for bit.

## From Python

```python
import pandas as pd
import ccdcpp

out = ccdcpp.detect(dates, spectral, qas, encoding=ccdcpp.presets.landsat_c2(), trace=True)
trace = pd.DataFrame(out["trace"])
print(trace[["kind", "phase", "next", "window_start", "window_stop", "day", "outcome"]].head())
#          kind       phase        next  window_start  window_stop     day     outcome
# 0  transition     Prepare  Initialize             0           12       0
# 1  transition  Initialize      Extend             0           12       0
# 2        test                                     0            0  724929   no change
# 3  transition      Extend      Extend             0           13       0
# ...

# Where the change tests fired, and their statistics (one per peek observation):
trace[trace.kind == "test"].query("outcome == 'change'")[["day", "values"]]

# The segments, as the run wrote them:
trace[trace.kind == "segment"][["day", "day_to", "outcome", "observations"]]
```

For a cube, name the pixels:

```python
out = ccdcpp.detect_cube(dates, stack, qas, encoding=..., threads=8,
                         trace_pixels=[(120, 37), (400, 12)])
pd.DataFrame(out["traces"][(120, 37)])
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
`detect` observes its one pixel whatever `RunSpec::observe` holds.

## From C++

```cpp
#include "ccd/results/diagnostics.hpp"
#include "ccd/run/run.hpp"

ccd::RunSpec spec;
spec.encoding = input;             // ccd::Input, the data's encoding
spec.observer = std::make_shared<ccd::TextObserver>(std::cerr);
ccd::detect(spec, dates, spectral, qas);
// pixel 0: transition Prepare -> Initialize window (0, 12) prev_break 0 n 302
// pixel 0: transition Initialize -> Extend window (0, 12) prev_break 0 n 302
// pixel 0: test 724929 no change [0.689656, 3.20607, 0.835624, 2.40568, 0.562177, 1.86732]
// ...

spec.observe = {120 * W + 37};     // a cube run: only these pixels (row * W + col)
ccd::detect_cube(spec, dates, cube, qas, 8);
```

An `Observer` implements one call, `event(pixel, const ObserverEvent&)`; in a cube run it is
called from several threads at once.

## Building without it

The hooks are compiled in with CMake's `CCD_OBSERVE` (ON by default). A pixel without an
observer then costs one pointer test per hook. `-DCCD_OBSERVE=OFF` removes them: an observer is
never called and `trace` comes back empty.
