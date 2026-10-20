# Input units

The input says only how to read the data. The configuration says what to do with it, in
physical units. The run translates once: 0–1 reflectance × 10000 = the 0–10000 DN check,
exactly as today.

## The boundary

- **The storage domain** is the data as given: `spectral` in the input's encoding, its QA, and
  its codes. Fill and saturation codes (`qa_fill`, `saturated`, `thermal_saturated`) belong
  here and are compared on the raw stored values, before any transform. The outputs are in it
  too: coefficients, intercepts, RMSEs and magnitudes are in the input's units (DN in, DN out).
- **The physical domain** is the configuration only: reflectance levels (0–1), reflectance
  differences (`alpha`, `coef_floor`, the green median's offset), degrees Celsius, days,
  observations, probabilities, bands by role and QA codes by name.
- **The conversion.** The run converts the configuration into the storage domain once, when
  it resolves it against the input: levels as `r * reflectance_unit + reflectance_offset`,
  differences as `r * reflectance_unit`, degrees Celsius as `c * 100 * thermal_unit / 10000`.
  The one conversion of the data is the thermal transform, `thermal * thermal_scale +
  thermal_offset`, applied to the thermal row of the procedures whose chain sets
  `thermal_transform`, with its output unit stated by `thermal_unit` (the value 100 °C has
  after it).

On Collection-1 DN (`ccdcpp.Input()`: `reflectance_unit` 10000, `reflectance_offset` 0, the
transform (10, -27315) to Celsius x 100, `thermal_unit` 10000), "ccdc" resolves to:

| Configuration (physical) | Resolved (storage) | Conversion |
|---|---|---|
| `band_range` 0 to 1 | 0 to 10000 | level: `r * 10000 + 0` |
| `thermal_range` -93.2 to 70.7 °C | -9320 to 7070 | `c * 100 * 10000 / 10000` |
| `alpha` 1e-4 | 1.0 | difference: `r * 10000` |
| `green_median` offset +0.04 | +400 | difference: `r * 10000` |

`ccdcpp.plan_of(config, input)` shows the resolved preprocessing and `Config.resolve(input)`
every resolved value.

## The encoding

`ccd::Input` (`include/ccd/input/input.hpp`, `ccdcpp.Input`) states the data's encoding:

- reflectance `r` is held as `r * reflectance_unit + reflectance_offset` (Collection-1 DN:
  10000 and 0, the defaults; reflectance: 1 and 0);
- the thermal transform `thermal * thermal_scale + thermal_offset` takes the thermal row to a
  multiple of degrees Celsius, and `thermal_unit` is the value 100 °C has after it (Collection-1
  DN: Kelvin x 10 through (10, -27315) is Celsius x 100, `thermal_unit` 10000, the defaults;
  Celsius / 100 through the identity: `thermal_unit` 1).

A configuration's reflectance levels (`band_range`'s bounds) resolve as `r * reflectance_unit +
reflectance_offset`, its reflectance differences (`alpha`, `coef_floor`, `green_median`'s
offset) as `r * reflectance_unit`, its degrees Celsius `c` as `c * 100 * thermal_unit / 10000`.
1e-4 x 10000, 0.04 x 10000, 0 x 10000 and 1 x 10000 are 1.0, 400.0, 0 and 10000 exactly, and
-93.2 and 70.7 degrees are -9320 and 7070 on DN and -0.932 and 0.707 on Celsius / 100, so a DN
input runs pyccd's DN values bit for bit.

## The physical-unit input

One divisor, 10000, applied to every band:

```text
reflectance_unit = 1.0
thermal_scale = 1.0       thermal_offset = 0.0    (identity)
thermal_unit = 1.0
```

with the caller's array pre-divided by 10000:

```text
optical  DN / 10000                  -> reflectance 0..1
thermal  (K*10 - 27315) / 10000      -> degrees Celsius / 100
```

The coefficients then come out in those same units: no descaling pass on the way out, at any
scale, for any band. `presets.landsat_c1_physical()` is this input.

**Measured**, on the pyccd reference pixel in `tests/ccd/machine/test_3657_3610_observations.csv`
(443 observations, 5 segments): against the DN input, this input reproduces the processing mask
bit for bit, the same 5 segments with identical start, end and break days, observation counts
and curve_qa, change probabilities equal to 0.0e+00 absolute, and every coefficient, intercept,
rmse and magnitude equal after multiplying by 10000 to a worst relative difference of 4.4e-13.
It is the same fit.

## Why degrees Celsius itself is not the physical unit

The Lasso's alpha is one value shared by all the band fits, and it is an absolute
soft-threshold, so the equivalence holds only if every band shares one scale factor. Optical DN
is 10000x reflectance; the DN thermal path lands on Celsius x 100, so plain Celsius would be a
100x factor: two scales, one alpha. Measured on the same pixel, the Celsius variant leaves the
six optical bands exact (1e-13) and breaks thermal alone: 2 extra nonzero coefficients, rmse
off by 1.7e-2 relative, intercept by 4.0e-2. Celsius / 100 is the unit the downstream
change-detection archive already stores.

## Collection 2

Collection-1's saturation range (0, 10000) and Collection-2's mean the same predicate,
reflectance in (0, 1): Collection-2 surface reflectance is `DN * 2.75e-5 - 0.2`. An input in
Collection-2 DN says so with `reflectance_unit = 1 / 2.75e-5` and `reflectance_offset = 0.2 /
2.75e-5`, and `band_range` 0..1 resolves to (7272.7..., 43636.3...), which keeps DN 7273 to
43636 (`tests/ccd/machine/test_units.cpp`, `Collection2DnResolvesReflectanceZeroToOne`). In
physical units there is no collection branch.

## The thermal range on the procedures without the transform

`thermal_range`'s bounds are in the transformed unit, and `ccdc` transforms for the Standard
procedure only (its preprocess section's `thermal_transform`). That asymmetry is pyccd's:
`kelvin_to_celsius()` is called inside `standard_procedure()` while `snow_procedure_filter()`
applies `filter_thermal_celsius()` to values nothing converted. On a DN input the PermanentSnow
and InsufficientClear procedures therefore test Kelvin x 10 against Celsius x 100 bounds, and
since Kelvin x 10 is about 2200..3300, inside [-9320, 7070], the filter rejects nothing a sensor
records there. It is kept for parity: ccdcpp is a recreation of pyccd and the regression
references encode it.

On the physical-unit input thermal is already Celsius / 100 for every procedure, so
PermanentSnow and InsufficientClear test converted values against converted bounds, as
Standard does, and the filter is live there: a clear or water observation outside (-93.20,
70.70) degrees Celsius is dropped (PermanentSnow then adds back every snow observation
regardless). The DN input keeps that observation. On those two procedures the two inputs agree
only while no clear or water observation's thermal value is out of range; the equivalence
measured above is on a Standard pixel.

## The guards

`tests/ccd/machine/test_units.cpp` asserts the equivalence and then, one value at a time, that
leaving it behind is detected. The three shapes of detection there are not interchangeable:

| Value | How leaving it behind is caught |
|---|---|
| `reflectance_unit` | by the equivalence test directly: it scales alpha (which moves the segment count) and coef_floor (which leaves sparsity identical, so only the value comparison catches it) |
| the bounds | not by it. The physical bounds are narrower, so leaving them at the DN value merely loosens a screen, and a screen that rejects nothing is indistinguishable from a correct one. The guard runs them the other way, DN input with the physical bound, where the mask empties and the pixel returns 0 segments instead of 5 |
| the green filter | unreachable through `run_pixel()` on the reference pixel, which takes the Standard procedure; `ccdc` runs the green median filter in the InsufficientClear chain only. Its guard fits the InsufficientClear procedure directly (`ccd::fit_whole_series`) |

A new unit-bearing field gets its guard in that file, with the shape of detection worked out.
