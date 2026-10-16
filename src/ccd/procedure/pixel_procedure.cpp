#include "ccd/procedure/pixel_procedure.hpp"

#include <cassert>
#include <utility>

#include "ccd/logger.hpp"
#include "ccd/procedure/kernels.hpp"

namespace ccd
{

PixelProcedure select_procedure(
    const Quality::Statistics& statistics,
    const HarmonicOptions& options
)
{
    if (!statistics.enough_clear(options.CLEAR_PCT_THRESHOLD)) {
        if (statistics.enough_snow(options.SNOW_PCT_THRESHOLD)) {
            return PixelProcedure::PermanentSnow;
        } else {
            return PixelProcedure::InsufficientClear;
        }
    }
    return PixelProcedure::Standard;
}

FitResult fit_whole_series(
    PixelProcedure procedure,
    const HarmonicWorkspace& hworkspace,
    ProcessingMask mask,
    LassoWorkspace& lworkspace,
    LassoSolver& solver
)
{
    assert(procedure != PixelProcedure::Standard);

    LOG_DEBUG(procedure == PixelProcedure::PermanentSnow ? "Snow" : "Clear");

    const auto& options = hworkspace.options();

    //----------------------------------------------------------
    // Not enough observations
    //----------------------------------------------------------
    if (mask.count() < options.MEOW_SIZE)
    {
        return {};
    }

    //----------------------------------------------------------
    // Mask dates, spectral given ProcessingMask
    //----------------------------------------------------------
    MaskedData masked_data =
        apply_mask(hworkspace, mask);

    const auto masked_dates =
        masked_data.dates_view();

    const auto masked_spectral =
        masked_data.spectral_view();

    //----------------------------------------------------------
    // Build result
    //----------------------------------------------------------
    FitResult results;
    ChangeModel result;
    result.start_day = hworkspace.dates()(0);
    result.end_day   = hworkspace.dates()(hworkspace.dates().size() - 1);
    result.break_day = result.end_day;

    result.observation_count = mask.count();
    result.change_probability = 0.0;
    result.curve_qa =
        procedure == PixelProcedure::PermanentSnow
            ? CurveQA::PersistentSnow
            : CurveQA::InsufficientClear;

    //----------------------------------------------------------
    // Fit every spectral band over every kept observation. The
    // magnitude of these segments is 0 and nothing reads a
    // returned segment's residuals.
    //----------------------------------------------------------
    fit_bands(
        masked_dates,
        masked_spectral,
        options.COEFFICIENT_MIN,
        KeepResiduals::No,
        lworkspace,
        solver,
        result.bands
    );

    results.models.push_back(std::move(result));
    results.mask = std::move(mask);

    return results;
}

} // namespace ccd
