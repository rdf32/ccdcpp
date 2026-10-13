#pragma once

#include <vector>
#include <cassert>
#include <algorithm>

#include "ccd/types.hpp"
#include "ccd/band_list.hpp"
#include "ccd/pmask.hpp"
#include "ccd/constants.hpp"
#include "ccd/array_view.hpp"

namespace ccd
{

//==============================================================================
//
// GreenFilterScope
//
// Which observations the InsufficientClear procedure's green median filter
// drops. The threshold is the same in each: the median green of the masked
// observations dated <= STAT_ORD, plus MEDIAN_GREEN_FILTER. A masked
// observation whose green is >= the threshold is dropped
//
//   All          at any date;
//   StatWindow   only when dated <= STAT_ORD, which is the scope pyccd's
//                insufficient_clear_filter computes the mask for (the
//                default);
//   Off          never: the filter does not run.
//
// When no masked observation is dated <= STAT_ORD there is no median and
// nothing is dropped.
//
//==============================================================================

enum class GreenFilterScope : std::uint8_t
{
    All,
    StatWindow,
    Off
};

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
    //
    // The codes the procedures compare `qas` against, one per observation:
    // the caller's uint8 codes, or a QaTable's output (qa_table.hpp), which
    // the table builders take from these fields.
    //
    //   QA_FILL               not a valid observation in the statistics
    //   QA_CLEAR, QA_WATER    kept for the fit; the clear percentage
    //   QA_SNOW               the snow percentage; PermanentSnow's fit
    //   QA_CLOUD              the statistics' cloud count
    //
    // Every other code keeps the observation out of the fit and counts only
    // toward the total. QA_SHADOW is not read by the procedures; the table
    // builders write it. QA_CIRRUS1, QA_CIRRUS2 and QA_OCCLUSION are
    // informational: nothing in ccdcpp reads them.
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
    // Cadence and change probability
    //
    // The Standard procedure's constants that are statements about the
    // revisit interval of the input or the false-alarm rate of the change
    // test. The defaults are pyccd's, tuned for 16-day Landsat. ccd::validate()
    // rejects zero or negative days, counts below 1, a probability outside
    // (0, 1) and a non-positive factor or divisor.
    //
    // MEOW_SIZE and the RETRAIN_* constants are observation counts and are
    // not scaled with the cadence, here or in pyccd: at a denser cadence the
    // same count spans less time.
    //--------------------------------------------------------------------------

    // Days. adjust_peek() scales PEEK_SIZE by PEEK_REFERENCE_DAYS over the
    // median gap between the statistics-window observations, and never below
    // PEEK_SIZE.
    scalar_t PEEK_REFERENCE_DAYS = 16.0;

    // Days. adjusted_variogram() raises the lag until the most common
    // separation exceeds this, then keeps only the pairs further apart than
    // it.
    std::int64_t VARIOGRAM_MIN_LAG_DAYS = 30;

    // Observations. Lookforward refits the models while the window holds
    // fewer than RETRAIN_MIN_OBS observations, or once the window spans at
    // least RETRAIN_SPAN_FACTOR times the span of the last fit. A window of
    // at most RETRAIN_MIN_OBS observations compares against the model RMSE;
    // a longer one against the seasonal RMSE below.
    index_t RETRAIN_MIN_OBS = 24;

    // Unitless ratio of spans; see RETRAIN_MIN_OBS.
    scalar_t RETRAIN_SPAN_FACTOR = 1.33;

    // Observations. The seasonal RMSE takes the fit-window residuals of the
    // SEASONAL_RMSE_COUNT observations closest in day of year to the last
    // peek observation...
    index_t SEASONAL_RMSE_COUNT = 24;

    // ...and is the norm of those residuals divided by SEASONAL_RMSE_DIVISOR.
    // Unitless. The default is pyccd's 4, sqrt(24 - 8): 24 residuals less the
    // 8 coefficients of the full model. It is a separate option and is not
    // derived from SEASONAL_RMSE_COUNT.
    scalar_t SEASONAL_RMSE_DIVISOR = 4.0;

    // Probability, in (0, 1). For a peek window of `peek` observations longer
    // than the default `defpeek`, adjust_change_threshold() takes the chi2
    // quantile at 1 - (1 - CHANGE_PROBABILITY)^(defpeek / peek). The default
    // CHANGE_THRESHOLD is the chi2(5) quantile at this probability, so a
    // caller changing one sets the other to match (ccdcpp.chi2_quantile).
    scalar_t CHANGE_PROBABILITY = 0.99;

    //--------------------------------------------------------------------------
    // Input unit convention
    //
    // Every value in this block is an ABSOLUTE threshold, which makes each one
    // a statement about the unit of the caller's `spectral` array, and
    // switching collection or unit is a change of options, not of C++.
    //
    // The defaults are Collection-1 DN as pyccd consumes it, and the
    // regression references in tests/ are computed with them.
    //
    // The *_INCLUSIVE flags beside the bounds are not unit constants: they say
    // whether a value equal to a bound is kept, which is the same question in
    // every unit, so the preset leaves them as they are.
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
    //    predicate. Collection-1's (0, 10000) and Collection-2's
    //    (7273, 43636) both mean "reflectance in (0, 1)" -- C2 surface
    //    reflectance is DN * 2.75e-5 - 0.2, which maps 7273 -> 0.0000 and
    //    43636 -> 1.0000. So in physical units there is no collection branch
    //    left to choose between; it collapses.
    //
    // 2. THERM_MIN/THERM_MAX are tested AFTER the transform above, and the
    //    transform runs for the Standard procedure ONLY. That asymmetry is
    //    inherited from pyccd, not introduced here: kelvin_to_celsius() is
    //    called inside standard_procedure() while snow_procedure_filter()
    //    applies filter_thermal_celsius() to values nothing converted. Under
    //    the DN defaults the PermanentSnow and InsufficientClear procedures
    //    therefore test Kelvin x 10 against Celsius x 100 bounds, and since
    //    Kelvin x 10 is ~2200..3300, comfortably inside [-9320, 7070], the
    //    filter rejects nothing a sensor records there. It is preserved
    //    deliberately, because ccdcpp is a recreation of pyccd and the
    //    regression references encode it.
    //
    //    Under the preset the caller has already converted thermal to
    //    Celsius / 100 for every procedure, so PermanentSnow and
    //    InsufficientClear test converted values against converted bounds,
    //    as Standard does, and the filter is live there: a clear or water
    //    observation outside (-93.20, 70.70) degrees Celsius is dropped
    //    (PermanentSnow then adds back every snow observation regardless).
    //    The DN defaults keep that observation. So on those two procedures
    //    the preset and the DN defaults agree only while no clear or water
    //    observation's thermal value is out of range; the equivalence
    //    measured above is on a Standard pixel and does not cover them.
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

    // Applied to the thermal band of detect()'s per-pixel copy before the
    // Standard procedure runs:
    //
    //     thermal <- thermal * THERMAL_SCALE + THERMAL_OFFSET
    //
    // The default is pyccd's Kelvin x 10 -> Celsius x 100. The caller's
    // buffer is never written. The identity (1.0, 0.0) skips the pass, and
    // with it the copy when the pixel is already contiguous float64, and so
    // does THERMAL_IDX == NO_BAND.
    scalar_t THERMAL_SCALE  = KELVIN_SCALE;
    scalar_t THERMAL_OFFSET = -KELVIN_OFFSET;

    // Saturation bound for the SATURATION_BANDS. An observation is kept when
    // every one of them lies within (REFL_MIN, REFL_MAX), each end inclusive
    // or exclusive by its flag (default exclusive on both ends, pyccd's
    // 0 < value < 10000).
    //
    // With either end exclusive NaN is rejected. With both ends inclusive the
    // test is !(value < REFL_MIN || value > REFL_MAX), under which NaN passes.
    scalar_t REFL_MIN = 0.0;
    scalar_t REFL_MAX = 10000.0;
    bool REFL_MIN_INCLUSIVE = false;
    bool REFL_MAX_INCLUSIVE = false;

    // Valid range for the THERMAL_IDX band, tested after the
    // THERMAL_SCALE/THERMAL_OFFSET transform, each end inclusive or exclusive
    // by its flag (default exclusive on both ends, pyccd's -9320 < value <
    // 7070, which rejects NaN). Defaults are Celsius x 100; the preset's are
    // those divided by 10000, i.e. [-0.9320, 0.7070]. NaN passes only with
    // both ends inclusive, as for the saturation bound.
    scalar_t THERM_MIN = MIN_CELSIUS;
    scalar_t THERM_MAX = MAX_CELSIUS;
    bool THERM_MIN_INCLUSIVE = false;
    bool THERM_MAX_INCLUSIVE = false;

    // Offset added to the median green value in the InsufficientClear
    // procedure's cloud screen. 400 is Collection-1 DN; the reflectance
    // equivalent is 0.04, which is why this is scalar_t and not index_t --
    // as an integer it could not represent the preset's value at all.
    scalar_t MEDIAN_GREEN_FILTER = 400.0;

    // Which observations the green median filter drops; see GreenFilterScope.
    // Not a unit constant.
    GreenFilterScope GREEN_FILTER_SCOPE = GreenFilterScope::StatWindow;

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

    //--------------------------------------------------------------------------
    // Band layout
    //
    // The rows of the spectral array. NUM_BANDS is the band count every call
    // must pass. Each row is fitted and reported -- ChangeModel::bands has
    // one entry per row, whatever its role -- and the three lists and two
    // indices below are the only places a procedure reads a band by role.
    // ccd::validate() checks them against the input once per call.
    //
    // The defaults are the Landsat layout: blue, green, red, nir, swir1,
    // swir2, thermal.
    //--------------------------------------------------------------------------
    index_t NUM_BANDS = 7;

    // Bands whose residuals make up the change magnitude, accumulated in
    // this order. The count is the degrees of freedom of the chi-squared
    // distribution behind the thresholds: CHANGE_THRESHOLD and
    // OUTLIER_THRESHOLD above are chi2(5) quantiles, so a caller changing the
    // count sets them to match, and adjust_change_threshold() takes its df
    // from here.
    BandList DETECTION_BANDS = {1, 2, 3, 4, 5};

    // Bands the Tmask outlier screen fits.
    BandList TMASK_BANDS = {1, 4};

    // Bands the saturation filter tests against REFL_MIN and REFL_MAX. May be
    // empty; must not hold THERMAL_IDX.
    BandList SATURATION_BANDS = {0, 1, 2, 3, 4, 5};

    // GREEN_IDX is read by InsufficientClear's green median filter and is
    // required. THERMAL_IDX is read by the thermal transform and the thermal
    // filter; NO_BAND says the input has no thermal band and skips both.
    //
    // BLUE_IDX, RED_IDX, NIR_IDX, SWIR1_IDX, SWIR2_IDX and QA_IDX are
    // informational: no procedure reads them.
    index_t BLUE_IDX    = 0;
    index_t GREEN_IDX   = 1;
    index_t RED_IDX     = 2;
    index_t NIR_IDX     = 3;
    index_t SWIR1_IDX   = 4;
    index_t SWIR2_IDX   = 5;
    index_t THERMAL_IDX = 6;
    index_t QA_IDX      = 7;

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

    // The workspace keeps its own copy of `options`: the Standard procedure
    // adjusts PEEK_SIZE and CHANGE_THRESHOLD per pixel.
    HarmonicWorkspace(
        ArrayView<const std::int64_t, 1> dates,
        ArrayView<const scalar_t, 2> spectral,
        ArrayView<const std::uint8_t, 1> qas,
        const HarmonicOptions& options
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

        // Date order is checked once per call at the entry points
        // (ccd::validate_dates), not per pixel here.
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

    //----------------------------------------------------------------------
    // Remove the observation at masked position `index`, in place.
    //
    // The result is what apply_mask() gives for the mask with that
    // observation cleared: the same values, in the same order.
    //----------------------------------------------------------------------
    void erase(index_t index)
    {
        assert(index < observations);

        dates.erase(dates.begin() + index);

        // Compacting forward never overwrites an unread value: the write
        // position trails the read position.
        index_t out = 0;
        for (index_t b = 0; b < bands; ++b)
        {
            for (index_t i = 0; i < observations; ++i)
            {
                if (i != index)
                    spectral[out++] = spectral[b * observations + i];
            }
        }

        --observations;
        spectral.resize(bands * observations);
    }
};

MaskedData apply_mask(
    const HarmonicWorkspace& workspace,
    const ProcessingMask& mask
);

// apply_mask() into `out`, reusing its storage.
void apply_mask(
    const HarmonicWorkspace& workspace,
    const ProcessingMask& mask,
    MaskedData& out
);

} // namespace ccd