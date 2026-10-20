# Input units

The input says only how to read the data. The configuration says what to do with it, in
physical units. The run translates once: 0–1 reflectance × 10000 = the 0–10000 DN check,
exactly as today.

## The boundary

- **The storage domain** is the data as given: `spectral` in the input's encoding, its QA, and
  its codes. Fill and saturation codes (`qa_fill`, `saturation_code`,
  `thermal_saturation_code`) belong here and are compared on the raw stored values, before any
  transform. The outputs are in it too: coefficients, intercepts, RMSEs and magnitudes are in
  the input's units (DN in, DN out).
- **The physical domain** is the configuration only: reflectance levels (0–1), reflectance
  differences (`alpha`, `coef_floor`, the green median's offset), degrees Celsius, days,
  observations, probabilities, bands by role and QA codes by name.
- **The conversion.** The run converts the configuration into the storage domain once, when
  it resolves it against the input: levels as `r * reflectance_scale + reflectance_offset`,
  differences as `r * reflectance_scale`, degrees Celsius as `c * 100 * thermal_at_100c /
  10000`. The one conversion of the data is the thermal transform, `thermal * thermal_scale +
  thermal_offset`, applied to the thermal row of the procedures whose chain sets
  `thermal_transform`, with its output unit stated by `thermal_at_100c` (the value 100 °C has
  after it).

On Collection-1 DN (`ccdcpp.Input()`: `reflectance_scale` 10000, `reflectance_offset` 0, the
transform (10, -27315) to Celsius x 100, `thermal_at_100c` 10000), "ccdc" resolves to:

| Configuration (physical) | Resolved (storage) | Conversion |
|---|---|---|
| `band_range` 0 to 1 | 0 to 10000 | level: `r * 10000 + 0` |
| `thermal_range` -93.2 to 70.7 °C | -9320 to 7070 | `c * 100 * 10000 / 10000` |
| `alpha` 1e-4 | 1.0 | difference: `r * 10000` |
| `green_median` offset +0.04 | +400 | difference: `r * 10000` |

`ccdcpp.plan_of(config, encoding)` shows the resolved preprocessing and `Config.resolve(encoding)`
every resolved value.

## The encoding

`ccd::Input` (`include/ccd/input/input.hpp`, `ccdcpp.Input`) states the data's encoding:

- reflectance `r` is held as `r * reflectance_scale + reflectance_offset` (Collection-1 DN:
  10000 and 0, the defaults; reflectance: 1 and 0);
- the thermal transform `thermal * thermal_scale + thermal_offset` takes the thermal row to a
  multiple of degrees Celsius, and `thermal_at_100c` is the value 100 °C has after it
  (Collection-1 DN: Kelvin x 10 through (10, -27315) is Celsius x 100, `thermal_at_100c` 10000,
  the defaults; Celsius / 100 through the identity: `thermal_at_100c` 1).

A configuration's reflectance levels (`band_range`'s bounds) resolve as `r *
reflectance_scale + reflectance_offset`, its reflectance differences (`alpha`, `coef_floor`,
`green_median`'s offset) as `r * reflectance_scale`, its degrees Celsius `c` as `c * 100 *
thermal_at_100c / 10000`. 1e-4 x 10000, 0.04 x 10000, 0 x 10000 and 1 x 10000 are 1.0, 400.0, 0
and 10000 exactly, and -93.2 and 70.7 degrees are -9320 and 7070 on DN and -0.932 and 0.707 on
Celsius / 100, so a DN input runs pyccd's DN values bit for bit.

## Saturation codes

`saturation_code` and `thermal_saturation_code` are the values a saturated observation holds in
the reflective rows and in the thermal row, as stored. Which products store one in the band
(from the official product guides):

| Product | Saturation code in the band (stored value) | Source |
|---|---|---|
| Landsat Collection 1 surface reflectance and brightness temperature | 20000 (fill -9999) | LaSRC product guide, LSDS-1368 |
| Landsat Collection 2 Level-2 scenes | none: saturation is flagged in `QA_RADSAT` only | LSDS-1619, section 6.1.2 |
| Landsat Collection 2 U.S. ARD | surface reflectance: 65535 on Landsat 4-7, 65455 on Landsat 8-9; surface temperature: none | DFCB, LSDS-1435 version 4 |
| HLS v2.0 | none (no saturation value or QA) | HLS v2.0 User Guide |
| Sentinel-2 L2A | 65535, in the raw DN, before `BOA_ADD_OFFSET` and the quantification | L2A product format |

A shipped profile declares a code only when its stored encoding is the product's own: today
only `presets.landsat_c1_dn()`, 20000 and 20000. The codes do not survive conversion to
reflectance: once the data is scaled (and offset), the code is a value like any other, so a
profile of converted data, `landsat_c1_physical()` and the reflectance profiles of the other
products, declares none. An input in a product's stored encoding (Collection-2 ARD DN, say)
declares the product's code itself; Collection-2 ARD's differs between Landsat 4-7 and 8-9, so
a stack of both holds no single code.

## The physical-unit input

One divisor, 10000, applied to every band:

```text
reflectance_scale = 1.0
thermal_scale = 1.0       thermal_offset = 0.0    (identity)
thermal_at_100c = 1.0
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
Collection-2 DN says so with `reflectance_scale = 1 / 2.75e-5` and `reflectance_offset = 0.2 /
2.75e-5`, and `band_range` 0..1 resolves to (7272.7..., 43636.3...), which keeps DN 7273 to
43636 (`tests/ccd/machine/test_units.cpp`, `Collection2DnResolvesReflectanceZeroToOne`). In
physical units there is no collection branch.

## The thermal range

`thermal_range`'s bounds are in the transformed unit, and `ccdc` tests them in the Standard
procedure only, the one whose chain applies the thermal transform. The PermanentSnow and
InsufficientClear chains read the thermal row as the input holds it and test no thermal range.
pyccd tests those two procedures' untransformed thermal band against the Celsius bounds
(`snow_procedure_filter()` applies `filter_thermal_celsius()` to values nothing converted):
on Collection-1 DN that compares Kelvin x 10, about 2200..3300, with [-9320, 7070] and rejects
nothing, so leaving the step out changes no DN output; on any other encoding it would drop
clear and water observations by a meaningless test. A configuration that wants the range there
adds the step to those chains with `thermal_transform`.

## The guards

`tests/ccd/machine/test_units.cpp` asserts the equivalence and then, one value at a time, that
leaving it behind is detected. The three shapes of detection there are not interchangeable:

| Value | How leaving it behind is caught |
|---|---|
| `reflectance_scale` | by the equivalence test directly: it scales alpha (which moves the segment count) and coef_floor (which leaves sparsity identical, so only the value comparison catches it) |
| the bounds | not by it. The physical bounds are narrower, so leaving them at the DN value merely loosens a screen, and a screen that rejects nothing is indistinguishable from a correct one. The guard runs them the other way, DN input with the physical bound, where the mask empties and the pixel returns 0 segments instead of 5 |
| the green filter | unreachable through `detect()` on the reference pixel, which takes the Standard procedure; `ccdc` runs the green median filter in the InsufficientClear chain only. Its guard fits the InsufficientClear procedure directly (`ccd::fit_whole_series`) |

A new unit-bearing field gets its guard in that file, with the shape of detection worked out.
