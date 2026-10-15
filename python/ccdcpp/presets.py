"""Sensor profiles for Landsat, HLS and Sentinel-2.

Each function returns a new :class:`ccdcpp.SensorProfile`.

The Landsat presets are the inputs ccdcpp is validated on. All three take the
seven Landsat bands in this order, one row each:

    blue, green, red, nir, swir1, swir2, thermal

which is TM/ETM+ bands 1, 2, 3, 4, 5, 7, 6 and OLI/TIRS bands 2, 3, 4, 5, 6,
7, 10. They differ in the unit of the input and in the QA they expect.

The HLS and Sentinel-2 presets, :func:`hls_l30`, :func:`hls_s30`,
:func:`hls` and :func:`sentinel2_l2a`, are validated on a real sample: one
cell of MGRS tile T16SGC near Atlanta (Landsat ARD h024v013 chunk 00030008),
2020-2023, read from Planetary Computer, four years of one cell.
:func:`hls_l30` is validated; :func:`hls_s30` and :func:`hls` are validated
with a caveat, Fmask over bright surfaces; :func:`sentinel2_l2a` is
validated at median level on Planetary Computer's Esri/MSFT/S2RP L2A,
2020–23; ESA's own processing not sampled. Each docstring gives the measured
numbers, and tests/python/fixtures/t16sgc_2020_2023 pins four of the
sample's pixels. All four take physical units (reflectance 0..1, thermal in
degrees Celsius / 100), detect change on the five Landsat-equivalent bands
(green, red, nir, swir1, swir2) with the default thresholds, have no
statistics cutoff (``stat_ord=None``) and keep pyccd's cadence values. The
single-product presets name their rows by product band ID (``B01``, ``B8A``,
...); :func:`hls`, whose rows mix two numberings, names them by role.

Every preset leaves ``preprocess`` None, so it runs the default plan of its
options (``ccdcpp.default_plan``); ``dataclasses.replace(preset(),
preprocess=ccdcpp.Preprocess(...))`` runs another.

A profile for another sensor is a :class:`ccdcpp.SensorProfile` call; the
module docstring of ``ccdcpp.profiles`` has an example.
"""
from .profiles import (
    CELSIUS_HUNDREDTHS,
    DN,
    KELVIN_TENTHS,
    LANDSAT_STAT_ORD,
    REFLECTANCE,
    SensorProfile,
)

__all__ = [
    "landsat_c1_dn",
    "landsat_c1_physical",
    "landsat_c2",
    "hls_l30",
    "hls_s30",
    "hls",
    "sentinel2_l2a",
]

_BANDS = ("blue", "green", "red", "nir", "swir1", "swir2", "thermal")

_LANDSAT = dict(
    band_names=_BANDS,
    roles=_BANDS,
    detection=("green", "red", "nir", "swir1", "swir2"),
    tmask=("green", "swir1"),
    saturation=("blue", "green", "red", "nir", "swir1", "swir2"),
    stat_ord=LANDSAT_STAT_ORD,
)


def landsat_c1_dn():
    """Landsat Collection 1 in DN, as pyccd consumes it.

    Input: optical surface reflectance x 10000 (0..10000), thermal brightness
    temperature in Kelvin x 10. QA: pre-classified uint8 codes (the
    ``HarmonicOptions`` ``QA_*`` fields); for raw ``pixel_qa`` pass
    ``qa_table=ccdcpp.qa_tables.landsat_c1_pixel_qa()``.

    ``options()`` is ``HarmonicOptions()`` and ``LassoOptions()``, field for
    field.
    """
    return SensorProfile(
        name="landsat_c1_dn",
        units=DN,
        alpha=1.0,
        coef_floor=1.0,
        thermal=KELVIN_TENTHS,
        **_LANDSAT,
    )


def landsat_c1_physical():
    """Landsat Collection 1 in physical units: the DN preset divided by 10000.

    Input: optical surface reflectance 0..1 (DN / 10000), thermal in degrees
    Celsius / 100 (``(K10 * 10 - 27315) / 10000`` from Kelvin x 10). QA:
    pre-classified uint8 codes.

    Every absolute threshold and both Lasso constants are the DN values
    divided by 10000, and the thermal transform is the identity, so the fit
    is the DN fit and the coefficients come out in the input's units.
    """
    return SensorProfile(
        name="landsat_c1_physical",
        units=REFLECTANCE,
        alpha=1e-4,
        coef_floor=1e-4,
        thermal=CELSIUS_HUNDREDTHS,
        **_LANDSAT,
    )


def landsat_c2():
    """Landsat Collection 2 Level-2 in physical units, with raw QA_PIXEL.

    Input: optical surface reflectance 0..1 (``SR_B* * 2.75e-5 - 0.2``),
    thermal in degrees Celsius / 100 (``(ST_B* * 0.00341802 + 149.0 -
    273.15) / 100``). QA: the raw uint16 ``QA_PIXEL`` band, decoded by
    ``ccdcpp.qa_tables.landsat_c2_qa_pixel``.

    The thresholds are those of :func:`landsat_c1_physical`: Collection 2's
    saturation range in DN, 7273..43636, is reflectance 0..1.
    """
    return SensorProfile(
        name="landsat_c2",
        units=REFLECTANCE,
        alpha=1e-4,
        coef_floor=1e-4,
        thermal=CELSIUS_HUNDREDTHS,
        qa_table_builder="landsat_c2_qa_pixel",
        **_LANDSAT,
    )


#------------------------------------------------------------------------------
# HLS v2.0 and Sentinel-2 L2A. Validated on one cell of T16SGC, 2020-2023.
#------------------------------------------------------------------------------

#: What the HLS and Sentinel-2 presets share: physical units with the
#: thresholds of landsat_c1_physical, the Landsat-equivalent detection and
#: Tmask bands, no statistics cutoff.
_PHYSICAL = dict(
    units=REFLECTANCE,
    alpha=1e-4,
    coef_floor=1e-4,
    detection=("green", "red", "nir", "swir1", "swir2"),
    tmask=("green", "swir1"),
    stat_ord=None,
)

#: Sentinel-2 MSI bands 2-8, 8A, 11, 12 and their roles.
_MSI_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
_MSI_ROLES = ("blue", "green", "red", "red_edge_1", "red_edge_2", "red_edge_3",
              "nir_broad", "nir", "swir1", "swir2")


def hls_l30():
    """HLS v2.0 L30 (Landsat 8/9 OLI/TIRS at 30 m).

    Input: every HLS ``int16`` layer times 0.0001. For B01-B07 that is the
    product's scale factor, giving reflectance; B10 is brightness
    temperature in degrees Celsius with scale factor 0.01, so ``B10 *
    0.0001`` is degrees Celsius / 100. One divisor for every row keeps one
    Lasso alpha right for every row. Fill -9999 becomes -0.9999, outside both
    the reflectance and the thermal bounds, and is dropped. Rows:

        B01 coastal, B02 blue, B03 green, B04 red, B05 nir (865 nm),
        B06 swir1, B07 swir2, B10 thermal (TIRS 1)

    B09 (cirrus) is left out; B11 (TIRS 2) can be added as an ``"other"``
    row. QA: the raw ``Fmask`` layer (uint8), decoded by
    ``ccdcpp.qa_tables.hls_fmask`` with the aerosol bits ignored;
    ``dataclasses.replace(hls_l30(), qa_table_options={"exclude_high_aerosol":
    True})`` treats high aerosol as cloud.

    Detection: B03, B04, B05, B06, B07, the default thresholds. Every
    optical row, coastal included, is saturation-screened. No statistics
    cutoff. The cadence is 16 days with Landsat 8 alone (from 2013) and 8
    with Landsat 9 (from late 2021); the peek window follows it.

    Validation: one cell of MGRS tile T16SGC near Atlanta (Landsat ARD
    h024v013 chunk 00030008), 2020-2023, read from Planetary Computer (LP
    DAAC's HLS v2.0), against Landsat Collection 2 ARD of the same cell and
    years run with :func:`landsat_c2`. Break agreement per pixel, kappa 0.44
    against Landsat 8+9 ARD, where adding Landsat 7 to the ARD gives 0.43;
    matched breaks a median 8 days apart, 85% within 96 days, with no date
    bias; model levels within 0.004 reflectance. Fmask decodes as the HLS
    v2.0 User Guide documents on all 38 values present. L30's 2020 is thin
    on this cell (37 scenes, half of them partial) and Fmask keeps 25-30%
    fewer clear observations than ARD QA_PIXEL, so over the four years no
    break is detected before the second half of 2021.
    """
    return SensorProfile(
        name="hls_l30",
        band_names=("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B10"),
        roles=("coastal", "blue", "green", "red", "nir", "swir1", "swir2", "thermal"),
        saturation=("coastal", "blue", "green", "red", "nir", "swir1", "swir2"),
        thermal=CELSIUS_HUNDREDTHS,
        qa_table_builder="hls_fmask",
        **_PHYSICAL,
    )


def hls_s30():
    """HLS v2.0 S30 (Sentinel-2 MSI at 30 m). Validated, with a caveat: Fmask
    over bright surfaces.

    Input: the HLS ``int16`` layers times the product's scale factor,
    0.0001: reflectance. Fill -9999 becomes -0.9999 and is dropped. Rows:

        B01 coastal, B02 blue, B03 green, B04 red, B05 red_edge_1,
        B06 red_edge_2, B07 red_edge_3, B08 nir_broad (842 nm),
        B8A nir (865 nm), B11 swir1, B12 swir2

    B09 (water vapour) and B10 (cirrus) are left out. No thermal band. QA:
    the raw ``Fmask`` layer (uint8), decoded by
    ``ccdcpp.qa_tables.hls_fmask``, aerosol ignored, as for :func:`hls_l30`.

    Detection: B03, B04, B8A, B11, B12, with the default thresholds. HLS
    adjusts these, and B01 and B02, to the OLI bandpasses; the red edges and
    B08 are not adjusted. They are fitted, reported and saturation-screened,
    with every other row. No statistics cutoff. The cadence is 10 days with
    Sentinel-2A alone (from 2015) and 5 with two satellites (2B from 2017,
    2C from 2024), denser where orbits overlap; the peek window follows it.

    Validation: one cell of MGRS tile T16SGC near Atlanta (Landsat ARD
    h024v013 chunk 00030008), 2020-2023, read from Planetary Computer (LP
    DAAC's HLS v2.0). S30 ran at a median gap of 5 days (peek window 19,
    change threshold 6.83). It finds 41% more break pixels than
    :func:`hls_l30` on the same cell, a median 16 days earlier, in classes
    that change rather than in stable forest; taking the red edges and B08
    out of the saturation screen changes no output. Fmask decodes as
    documented on every value present. Caveat, Fmask over bright surfaces:
    Fmask on Sentinel-2, which has no thermal band, calls bright barren land
    cloud. On the cell's quarries 61% of S30 observations are cloud, and
    1,145 of about 1,860 quarry pixels go to the InsufficientClear procedure
    (L30: 2; Landsat ARD: 170).
    """
    return SensorProfile(
        name="hls_s30",
        band_names=("B01",) + _MSI_BANDS,
        roles=("coastal",) + _MSI_ROLES,
        saturation=("coastal",) + _MSI_ROLES,
        qa_table_builder="hls_fmask",
        **_PHYSICAL,
    )


_HLS_ROLES = ("coastal", "blue", "green", "red", "nir", "swir1", "swir2")


def hls():
    """HLS v2.0 L30 and S30 in one series. Validated, with a caveat: Fmask over
    bright surfaces.

    Input: the bands the two products share, in reflectance (the HLS
    ``int16`` layers times 0.0001), rows named by role because the products
    number them differently:

        coastal (L30 B01, S30 B01), blue (B02, B02), green (B03, B03),
        red (B04, B04), nir (L30 B05, S30 B8A), swir1 (B06, B11),
        swir2 (B07, B12)

    No thermal band, since S30 has none. QA: the raw ``Fmask`` layers
    (uint8) of both, decoded by ``ccdcpp.qa_tables.hls_fmask``, aerosol
    ignored.

    Detection: green, red, nir, swir1, swir2, the default thresholds; every
    row is saturation-screened. No statistics cutoff. On the validation cell
    the series ran at a median gap of 4-5 days, and the peek window grew to
    19-24 observations (19 at 5 days, 24 at 4; change threshold 6.83 to
    5.90); ``MEOW_SIZE`` and the ``RETRAIN_*`` counts stay pyccd's
    observation counts, so 12 observations span about 50-60 days.

    An L30 and an S30 observation on one date: the filters keep the first in
    array order, so put L30 before S30 on a shared date.

    Validation: one cell of MGRS tile T16SGC near Atlanta (Landsat ARD
    h024v013 chunk 00030008), 2020-2023, read from Planetary Computer (LP
    DAAC's HLS v2.0). The series mostly follows S30: break agreement per
    pixel, kappa 0.71 with :func:`hls_s30` and 0.45 with :func:`hls_l30`.
    L30 first on a shared date worked on all 85 dates the products share.
    Caveat, Fmask over bright surfaces, through the S30 observations: 1,741
    of 300,288 pixels go to the InsufficientClear procedure (hls_s30: 2,101;
    hls_l30: 2), as described for :func:`hls_s30`.
    """
    return SensorProfile(
        name="hls",
        band_names=_HLS_ROLES,
        roles=_HLS_ROLES,
        saturation=_HLS_ROLES,
        qa_table_builder="hls_fmask",
        **_PHYSICAL,
    )


def sentinel2_l2a():
    """Sentinel-2 L2A (ESA Sen2Cor), native bands. Validated at median level on
    Planetary Computer's Esri/MSFT/S2RP L2A, 2020–23; ESA's own processing
    not sampled.

    Input: BOA reflectance with the product's offset removed,
    ``(DN + BOA_ADD_OFFSET) / 10000``. From processing baseline 04.00 (25
    January 2022) BOA_ADD_OFFSET is a negative, band-dependent constant in
    the product metadata, 0 before; it varies by scene, so ccdcpp cannot
    apply it and the caller must. A series that mixes corrected and
    uncorrected scenes jumps by the offset and detects false breaks. Some
    archives distribute the bands already corrected while their metadata
    still declares the offset (earth-search's ``sentinel-2-l2a`` COGs); do
    not apply it a second time. DN 0 (no data) becomes 0 or negative and is
    dropped, as is any value at or above 1.0. Resampling the 10 m and 20 m
    bands to one grid is the caller's. Rows:

        B02 blue, B03 green, B04 red, B05 red_edge_1, B06 red_edge_2,
        B07 red_edge_3, B08 nir_broad (842 nm), B8A nir (865 nm),
        B11 swir1, B12 swir2

    B01 and B09 (60 m, atmospheric) are left out; L2A has no B10. No thermal
    band. QA: the raw ``SCL`` layer (uint8), decoded by
    ``ccdcpp.qa_tables.sentinel2_scl``: 4, 5 clear; 6 water; 2, 3 shadow;
    7-10 cloud (7, unclassified, included); 11 snow; 0, 1 and anything else
    fill.

    SCL 7 as cloud costs water pixels. On the validation cell SCL 7 is 2.13%
    of the non-fill observations, and with SCL 2 (cast shadows) on small
    water bodies it sends 1,355 of 3,983 water pixels, and 3,426 pixels in
    all, to the InsufficientClear procedure; decoding 7 as fill would send
    2,545, as clear 587. It stays cloud. A caller who wants the water pixels
    passes ``qa_table=ccdcpp.QaTable(codes)``, with ``codes`` the table's
    codes and entry 7 changed.

    Detection: B03, B04, B8A, B11, B12, the default thresholds. The red
    edges and B08 are fitted, reported and saturation-screened. No
    statistics cutoff. The cadence is 10 days with Sentinel-2A alone (from
    mid-2015) and 5 with two satellites (2B from 2017, 2C from 2024), denser
    where orbits overlap. Two observations on one date (overlapping granules
    or datastrips): the filters keep the first in array order; merge them
    first to choose.

    Validation: one cell of MGRS tile T16SGC near Atlanta (Landsat ARD
    h024v013 chunk 00030008), 2020-2023, read from Planetary Computer
    (collection ``sentinel-2-l2a``, at 20 m, with the 10 m bands averaged to
    20 m). With the per-scene offset applied, the window around the 2022
    baseline change looks like any other (0.34% of pixels break within 60
    days of it; 92.3% without the offset). Against :func:`hls_s30` the two
    agree at median level: model levels about 0.01 above S30's with about
    twice the RMSE (no BRDF or bandpass adjustment), break agreement per
    pixel kappa 0.48, and fewer break pixels (2.35% against 3.62%). SCL
    decodes as documented on every class present (0, 2-11).
    """
    return SensorProfile(
        name="sentinel2_l2a",
        band_names=_MSI_BANDS,
        roles=_MSI_ROLES,
        saturation=_MSI_ROLES,
        qa_table_builder="sentinel2_scl",
        **_PHYSICAL,
    )
