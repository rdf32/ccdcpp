#include "ccd/machine/pixel_procedure.hpp"

#include <cassert>
#include <utility>

#include "ccd/core/logger.hpp"
#include "ccd/machine/kernels.hpp"

namespace ccd
{

FitResult fit_whole_series(
    PixelProcedure procedure,
    const HarmonicWorkspace& hworkspace,
    ProcessingMask mask,
    index_t min_observations,
    index_t num_coefficients,
    LassoWorkspace& lworkspace,
    LassoSolver& solver
)
{
    assert(procedure != PixelProcedure::Standard);

    LOG_DEBUG((procedure == PixelProcedure::PermanentSnow ? "Snow" : "Clear"));

    //----------------------------------------------------------
    // Not enough observations: no segment, and the mask
    //----------------------------------------------------------
    if (mask.count() < min_observations)
    {
        FitResult none;
        none.mask = std::move(mask);
        return none;
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
    // magnitude of these segments is 0.
    //----------------------------------------------------------
    std::vector<LassoResult> fits;
    fit_bands(
        masked_dates,
        masked_spectral,
        num_coefficients,
        KeepResiduals::No,
        lworkspace,
        solver,
        fits
    );

    result.bands.reserve(fits.size());
    for (const LassoResult& fit : fits)
        result.bands.push_back(band_fit(fit));

    results.models.push_back(std::move(result));
    results.mask = std::move(mask);

    return results;
}

} // namespace ccd
