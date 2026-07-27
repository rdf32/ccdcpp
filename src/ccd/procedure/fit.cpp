#include "ccd/procedure/fit.hpp"


namespace ccd
{

FitResult FitProcedure::run(
    HarmonicWorkspace& workspace,
    LassoSolver& solver
) { 
    const auto& options = workspace.options();
    //----------------------------------------------------------
    // Build processing mask
    //----------------------------------------------------------
    ProcessingMask mask =
        select_observations(workspace);

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
        apply_mask(workspace, mask);

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
    auto X_storage = 
        lasso_basis(masked_dates, num_coef);

    auto X = ArrayView<const scalar_t, 2>::contiguous(
        X_storage.data(),
        {masked_dates.size(), 7} // always shape (num_obs, 7) for lasso basis
    );

    //----------------------------------------------------------
    // Build result
    //----------------------------------------------------------
    FitResult results;
    ChangeModel result;
    result.start_day = masked_dates(0);
    result.end_day   = masked_dates(masked_dates.size() - 1);
    result.break_day = result.end_day;

    result.observation_count = mask.count();
    result.change_probability = 0.0;
    result.curve_qa = curve_qa();

    //----------------------------------------------------------
    // Fit every spectral band
    //----------------------------------------------------------
    for (index_t band = 0; band < masked_spectral.extent(0); ++band)
    {
        auto y = masked_spectral.slice(
            fixed(band), 
            all()
        );

        LassoModel model = solver.fit(
            X,
            y
        );

        auto preds = model.predict(
            X
        );

        auto metrics = score(
            y,
            preds,
            num_coef,
            true
        );

        LassoResult band_result = {
            model,
            metrics
        };

        result.bands[band] = band_result;
    }

    results.models.push_back(result);
    results.mask = mask;

    return results;
}


} // namespace ccd