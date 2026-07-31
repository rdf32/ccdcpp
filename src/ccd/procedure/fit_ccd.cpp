#include "ccd/procedure/fit_ccd.hpp"

#include <algorithm>
#include <boost/math/distributions/chi_squared.hpp>

#include "ccd/maths.hpp"
#include "ccd/constants.hpp"


namespace ccd
{

CCDFitResult CCDFitProcedure::run(
    HarmonicWorkspace& hworkspace,
    CCDLassoWorkspace& lworkspace,
    CCDLassoSolver& solver
) { 
    const auto& options = hworkspace.options();
    //----------------------------------------------------------
    // Build processing mask
    //----------------------------------------------------------
    ProcessingMask mask =
        select_observations(hworkspace);

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
    // Determine harmonic complexity
    //----------------------------------------------------------
    const index_t num_coef =
        coefficient_count(masked_dates, options);

    //----------------------------------------------------------
    // Build harmonic basis
    //----------------------------------------------------------
    CCDLassoProblem input_storage = 
        ccd_lasso_basis(masked_dates, num_coef);

    lworkspace.resize(mask.count());
    //----------------------------------------------------------
    // Build result
    //----------------------------------------------------------
    CCDFitResult results;
    CCDChangeModel result;
    result.start_day = hworkspace.dates()(0);
    result.end_day   = hworkspace.dates()(hworkspace.dates().size() - 1);
    result.break_day = result.end_day;

    result.observation_count = mask.count();
    result.change_probability = 0.0;
    result.curve_qa = curve_qa();

    //----------------------------------------------------------
    // Fit every spectral band
    //----------------------------------------------------------
    for (index_t band = 0; band < masked_spectral.extent(0); ++band)
    {   
        lworkspace.reset();
        auto y = masked_spectral.slice(
            fixed(band), 
            all()
        );

        CCDLassoModel model = solver.fit(
            lworkspace,
            input_storage,
            y
        );

        model.predict(
            input_storage.X(),
            lworkspace.predictions
        );

        auto metrics = score(
            y,
            lworkspace.predictions,
            lworkspace.residuals,
            num_coef,
            true
        );
        metrics.magn = 0.0;

        result.bands[band] = CCDLassoResult{
            std::move(model),
            std::move(metrics)
        };
    }

    results.models.push_back(result);
    results.mask = mask;

    return results;
}

} // namespace ccd