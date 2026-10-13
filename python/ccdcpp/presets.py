"""Sensor profiles for the inputs ccdcpp is validated on: Landsat.

Each function returns a new :class:`ccdcpp.SensorProfile`. All three take the
seven Landsat bands in this order, one row each:

    blue, green, red, nir, swir1, swir2, thermal

which is TM/ETM+ bands 1, 2, 3, 4, 5, 7, 6 and OLI/TIRS bands 2, 3, 4, 5, 6,
7, 10. They differ in the unit of the input and in the QA they expect.

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

__all__ = ["landsat_c1_dn", "landsat_c1_physical", "landsat_c2"]

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
