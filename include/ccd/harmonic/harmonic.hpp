#pragma once

#include <vector>
#include <cassert>
#include <algorithm>

#include "ccd/types.hpp"
#include "ccd/pmask.hpp"
#include "ccd/constants.hpp"
#include "ccd/array_view.hpp"

namespace ccd
{

//==============================================================================
//
// HarmonicOptions
//
// Parameters controlling CCD harmonic processing.
//
// This class contains no computation.
//
//==============================================================================

struct HarmonicOptions
{
    //----------------------------------------------------------------------
    // Processing window parameters
    //----------------------------------------------------------------------
    index_t MEOW_SIZE = 12;
    index_t PEEK_SIZE = 6;

    std::int64_t DAY_DELTA = 365;
    
    //--------------------------------------------------------------------------
    // QA Values
    //--------------------------------------------------------------------------
    std::uint8_t QA_FILL      = 255;
    std::uint8_t QA_CLEAR     = 0;
    std::uint8_t QA_WATER     = 1;
    std::uint8_t QA_SHADOW    = 2;
    std::uint8_t QA_SNOW      = 3;
    std::uint8_t QA_CLOUD     = 4;
    std::uint8_t QA_CIRRUS1   = 8;
    std::uint8_t QA_CIRRUS2   = 9;
    std::uint8_t QA_OCCLUSION = 10;

    //--------------------------------------------------------------------------
    // Quality Thresholds
    //--------------------------------------------------------------------------

    // Minimum number of clear observations required.
    index_t CLEAR_OBSERVATION_THRESHOLD = 3;

    // Minimum clear/water percentage required.
    scalar_t CLEAR_PCT_THRESHOLD = 0.25;

    // Snow classification threshold.
    scalar_t SNOW_PCT_THRESHOLD = 0.75;

    scalar_t OUTLIER_THRESHOLD = 35.888186879610423L;

    scalar_t CHANGE_THRESHOLD   = 15.086272469388987l;

    scalar_t T_CONST = 4.89;

    //--------------------------------------------------------------------------
    // Input unit convention
    //
    // Every value in this block is an ABSOLUTE threshold, which makes each one
    // a statement about the unit of the caller's `spectral` array. They are
    // gathered here because they were previously spread across three files as
    // literals and two commented-out "collection-1 / collection-2" branches,
    // and switching collection meant editing C++.
    //
    // The DEFAULTS ARE UNCHANGED: Collection-1 DN exactly as pyccd consumed
    // it, so a caller that sets nothing gets the historical numbers and the
    // regression references in tests/ still hold.
    //
    // THE PHYSICAL-UNIT PRESET
    //
    // One divisor, 10000, applied to every band of the input and to every
    // absolute threshold below. Nothing else.
    //
    //     THERMAL_SCALE = 1.0       THERMAL_OFFSET = 0.0    (identity)
    //     REFL_MIN      = 0.0       REFL_MAX       = 1.0
    //     THERM_MIN     = -0.9320   THERM_MAX      = 0.7070
    //     MEDIAN_GREEN_FILTER = 0.04
    //     LassoOptions::alpha = 1e-4    LassoOptions::coef_floor = 1e-4
    //
    // and the caller pre-divides its own array by 10000:
    //
    //     optical  DN / 10000                  -> reflectance 0..1
    //     thermal  (K*10 - 27315) / 10000      -> degrees Celsius / 100
    //
    // The coefficients then come out in those same units, which is the whole
    // point: no descaling pass on the way out, at any scale, for any band.
    //
    // MEASURED, on the pyccd reference pixel in
    // tests/ccd/procedure/test_3657_3610_observations.csv (443 observations,
    // 5 segments): against the DN defaults, this preset reproduces the
    // processing mask bit for bit, the same 5 segments with identical
    // start/end/break days, observation counts and curve_qa, change
    // probabilities equal to 0.0e+00 absolute, and every coefficient,
    // intercept, rmse and magnitude equal after multiplying by 10000 to a
    // worst relative difference of 4.4e-13. It is the same fit.
    //
    // WHY DEGREES CELSIUS ITSELF IS NOT THE PRESET
    //
    // Feeding plain Celsius is the obvious choice and it is wrong, for a
    // reason that is invisible from here: LassoOptions::alpha is ONE value
    // shared by all seven band fits, and it is an absolute soft-threshold, so
    // the preset can only hold if every band shares one scale factor. Optical
    // DN is 10000x reflectance; the DN thermal path lands on Celsius x 100, so
    // plain Celsius would be a 100x factor -- two scales, one alpha. Measured
    // on the same pixel, the Celsius variant leaves the six optical bands
    // exact (1e-13) and breaks thermal alone: 2 extra nonzero coefficients,
    // rmse off by 1.7e-2 relative, intercept by 4.0e-2. Celsius/100 looks like
    // a strange unit to ask a caller for, and it is exactly the unit the
    // downstream change-detection archive already stores, so nothing is lost.
    //
    // Two more things are worth knowing before changing any of these.
    //
    // 1. The two collection variants of the saturation bound are ONE
    //    predicate. Collection-1's [0, 10000] and Collection-2's
    //    [7273, 43636] both mean "reflectance in [0, 1]" -- C2 surface
    //    reflectance is DN * 2.75e-5 - 0.2, which maps 7273 -> 0.0000 and
    //    43636 -> 1.0000. So in physical units there is no collection branch
    //    left to choose between; it collapses.
    //
    // 2. THERM_MIN/THERM_MAX are tested AFTER the transform above, and the
    //    transform runs for the Standard procedure ONLY. That asymmetry is
    //    inherited from pyccd, not introduced here: kelvin_to_celsius() is
    //    called inside standard_procedure() while snow_procedure_filter()
    //    applies filter_thermal_celsius() to values nothing converted. The
    //    consequence is that the thermal filter is a silent no-op for the
    //    PermanentSnow and InsufficientClear procedures -- Kelvin x 10 is
    //    ~2200..3300, comfortably inside [-9320, 7070]. It is preserved
    //    deliberately, because ccdcpp is a recreation of pyccd and the
    //    regression references encode it. Under the preset the bounds are
    //    scaled by the same 10000, so the no-op holds there too and the
    //    equivalence measured above covers it.
    //
    // Where the guard lives, and what each part of it is for.
    // tests/ccd/procedure/test_units.cpp asserts the equivalence above and
    // then, one constant at a time, that leaving it behind is DETECTED. The
    // three shapes of detection there are not interchangeable:
    //
    //   alpha, coef_floor    caught by the equivalence test directly. alpha
    //                        also moves the segment count; coef_floor does
    //                        not and leaves sparsity identical, so only the
    //                        value comparison catches it.
    //   REFL_*, THERM_*      NOT caught by it. The preset makes these bounds
    //                        narrower, so leaving them at the DN value merely
    //                        loosens a screen, and a screen that rejects
    //                        nothing is indistinguishable from a correct one.
    //                        The guard runs them the other way -- DN input,
    //                        physical bound -- where the mask empties and the
    //                        pixel returns 0 segments instead of 5.
    //   MEDIAN_GREEN_FILTER  unreachable through detect() on the reference
    //                        pixel, because apply_green_median_filter is
    //                        called from filter::insufficientclear only and
    //                        that pixel takes the Standard procedure. Even 0.0
    //                        changes nothing. Its guard drives
    //                        ccd::InsufficientClear directly.
    //
    // If you add a constant to this block, add its guard in that file and work
    // out which of the three shapes it is. Two of the five are invisible to
    // the obvious test.
    //--------------------------------------------------------------------------

    // Applied to the thermal band in place before the Standard procedure runs:
    //
    //     thermal <- thermal * THERMAL_SCALE + THERMAL_OFFSET
    //
    // The default is pyccd's Kelvin x 10 -> Celsius x 100. The identity
    // (1.0, 0.0) skips the pass entirely, which matters for more than speed:
    // `spectral` is a view onto the CALLER's buffer, so skipping the write
    // makes detect() idempotent on the same array.
    scalar_t THERMAL_SCALE  = KELVIN_SCALE;
    scalar_t THERMAL_OFFSET = -KELVIN_OFFSET;

    // Saturation bound for the six optical bands, inclusive on both ends
    // (an observation is rejected on value < REFL_MIN || value > REFL_MAX).
    scalar_t REFL_MIN = 0.0;
    scalar_t REFL_MAX = 10000.0;

    // Valid range for the thermal band, exclusive on both ends, tested after
    // the THERMAL_SCALE/THERMAL_OFFSET transform. Defaults are Celsius x 100;
    // the preset's are those divided by 10000, i.e. [-0.9320, 0.7070].
    scalar_t THERM_MIN = MIN_CELSIUS;
    scalar_t THERM_MAX = MAX_CELSIUS;

    // Offset added to the median green value in the InsufficientClear
    // procedure's cloud screen. 400 is Collection-1 DN; the reflectance
    // equivalent is 0.04, which is why this is scalar_t and not index_t --
    // as an integer it could not represent the preset's value at all.
    scalar_t MEDIAN_GREEN_FILTER = 400.0;

    //--------------------------------------------------------------------------
    // Statistics
    //--------------------------------------------------------------------------
    // Landsat CCD historical cutoff date.
    std::int64_t STAT_ORD = 736694;

    //--------------------------------------------------------------------------
    // Harmonic Coefficients
    //--------------------------------------------------------------------------
    // Minimum coefficient count.

    // Example:
    // constant + trend + annual harmonic
    index_t COEFFICIENT_MIN = 4;
    // Add second harmonic.
    index_t COEFFICIENT_MID = 6;
    // Add third harmonic.
    index_t COEFFICIENT_MAX = 8;

    // Value used to determine the 
    // minimum number of observations required for
    // defined number of coefficients
    // e.g. COEFFICIENT_MIN * NUM_OBS_FACTOR = 12
    index_t NUM_OBS_FACTOR = 3;

    // Define spectral band indices on input observations array
    index_t BLUE_IDX    = 0;
    index_t GREEN_IDX   = 1;
    index_t RED_IDX     = 2;
    index_t NIR_IDX     = 3;
    index_t SWIR1_IDX   = 4;
    index_t SWIR2_IDX   = 5;
    index_t THERMAL_IDX = 6;
    index_t QA_IDX      = 7;

    // spectral bands used for detection change
    const std::vector<index_t> DETECTION_BANDS = {1, 2, 3, 4, 5};

    // spectral bands used for Tmask filtering
    const std::vector<index_t> TMASK_BANDS = {1, 4};

};


//==============================================================================
//
// HarmonicWorkspace
//
// Immutable workspace describing one pixel.
//
// Represents a single pixel time series with multiple bands / features/

// Stores immutable observations together with reusable scratch
// buffers used during harmonic fitting. The workspace itself
// contains no fitting logic.
//
// Responsibilities:
//
//  • Validate input arrays
//  • Store dates
//  • Store spectral observations
//  • Store QA observations
//
// Everything derived from these inputs (processing masks, harmonic basis,
// quality statistics, etc.) is computed elsewhere.
//
//==============================================================================

class HarmonicWorkspace
{
public:

    //----------------------------------------------------------------------
    // Constructor
    //----------------------------------------------------------------------

    HarmonicWorkspace(
        ArrayView<const std::int64_t, 1> dates,
        ArrayView<const scalar_t, 2> spectral,
        ArrayView<const std::uint8_t, 1> qas,
        HarmonicOptions& options
    )
        :
        dates_(dates),
        spectral_(spectral),
        qas_(qas),
        options_(options)
    {   
        validate();
    }

    //----------------------------------------------------------------------
    // Validation
    //----------------------------------------------------------------------

    void validate() const
    {
        assert(dates_.size() == spectral_.extent(1));
        assert(dates_.size() == qas_.size());

        assert(spectral_.is_contiguous());

        assert(
            std::is_sorted(
                dates_.data(),
                dates_.data() + dates_.size()
            )
        );
        // assert spectral has the correct layout
    }

    //----------------------------------------------------------------------
    // Input data
    //----------------------------------------------------------------------

    ArrayView<const std::int64_t, 1> dates() const
    {
        return dates_;
    }

    ArrayView<const scalar_t, 2> spectral() const
    {
        return spectral_;
    }

    ArrayView<const std::uint8_t, 1> qas() const
    {
        return qas_;
    }

    HarmonicOptions& options()
    {
        return options_;
    }

    const HarmonicOptions& options() const
    {
        return options_;
    }
    
private:

    ArrayView<const std::int64_t, 1> dates_;
    ArrayView<const scalar_t, 2> spectral_;
    ArrayView<const std::uint8_t, 1> qas_;

    HarmonicOptions options_;

};

struct MaskedData
{
    std::vector<std::int64_t> dates;
    std::vector<scalar_t> spectral;

    index_t bands;
    index_t observations;

    ArrayView<std::int64_t, 1> dates_view()
    {
        return ArrayView<std::int64_t,1>::contiguous(
            dates.data(),
            {observations}
        );
    }

    ArrayView<scalar_t, 2> spectral_view()
    {
        return ArrayView<scalar_t,2>::contiguous(
            spectral.data(),
            {bands, observations}
        );
    }
};

MaskedData apply_mask(
    const HarmonicWorkspace& workspace,
    const ProcessingMask& mask
);

} // namespace ccd