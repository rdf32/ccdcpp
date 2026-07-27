#pragma once

#include <vector>
#include <cassert>
#include <algorithm>

#include "ccd/types.hpp"
#include "ccd/pmask.hpp"
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

    index_t MEDIAN_GREEN_FILTER = 400;

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