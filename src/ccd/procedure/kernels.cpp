#include "ccd/procedure/kernels.hpp"

#include <cassert>
#include <utility>

#include "ccd/logger.hpp"

namespace ccd
{

namespace
{

// One band's fit, over the basis already in `lworkspace`.
void fit_band(
    ArrayView<const scalar_t, 1> y,
    index_t num_coefficients,
    KeepResiduals keep,
    LassoWorkspace& lworkspace,
    LassoSolver& solver,
    LassoResult& fit
)
{
    LassoModel model = solver.fit(
        lworkspace,
        y
    );

    model.predict(
        lworkspace.X(),
        lworkspace.predictions
    );

    fit.model = std::move(model);
    fit.score.rmse = rmse(
        y,
        lworkspace.predictions,
        lworkspace.residuals,
        num_coefficients,
        true
    );
    fit.score.magn = 0.0;

    if (keep == KeepResiduals::Yes)
    {
        fit.score.residuals.assign(
            lworkspace.residuals.begin(),
            lworkspace.residuals.end()
        );
    }
    else
    {
        fit.score.residuals.clear();
    }
}

} // namespace

void fit_bands(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    index_t num_coefficients,
    const BandList& bands,
    KeepResiduals keep,
    LassoWorkspace& lworkspace,
    LassoSolver& solver,
    std::vector<LassoResult>& fits
)
{
    lworkspace.resize(dates.size());
    lworkspace.build_basis(dates, num_coefficients);

    for (const index_t band : bands)
    {
        fit_band(
            spectral.slice(fixed(band), all()),
            num_coefficients,
            keep,
            lworkspace,
            solver,
            fits[band]
        );
    }
}

void fit_bands(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    index_t num_coefficients,
    KeepResiduals keep,
    LassoWorkspace& lworkspace,
    LassoSolver& solver,
    std::vector<LassoResult>& fits
)
{
    lworkspace.resize(dates.size());
    lworkspace.build_basis(dates, num_coefficients);

    const index_t num_bands = spectral.extent(0);
    fits.resize(num_bands);

    for (index_t band = 0; band < num_bands; ++band)
    {
        fit_band(
            spectral.slice(fixed(band), all()),
            num_coefficients,
            keep,
            lworkspace,
            solver,
            fits[band]
        );
    }
}

ChangeModel catch_segment(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    const Window& window,
    index_t num_coefficients,
    CurveQA curve_qa,
    LassoWorkspace& lworkspace,
    LassoSolver& solver
)
{
    LOG_DEBUG(
        "Catching observations: ("
        << window.start
        << ", "
        << window.stop
        << ")"
    );

    std::vector<LassoResult> fits;
    fit_bands(
        dates.slice(range(window.start, window.stop)),
        spectral.slice(all(), range(window.start, window.stop)),
        num_coefficients,
        KeepResiduals::No,
        lworkspace,
        solver,
        fits
    );

    ChangeModel result;
    result.bands.reserve(fits.size());
    for (const LassoResult& fit : fits)
        result.bands.push_back(band_fit(fit));

    if (window.stop >= dates.size()) {
        result.break_day = dates(dates.size() - 1);
    } else {
        result.break_day = dates(window.stop);
    }
    result.start_day = dates(window.start);
    result.end_day   = dates(window.stop - 1);

    result.observation_count = window.stop - window.start;
    result.change_probability = 0.0;
    result.curve_qa = curve_qa;

    return result;
}

const char* to_string(InitializeOutcome outcome)
{
    switch (outcome)
    {
        case InitializeOutcome::Exhausted:        return "exhausted";
        case InitializeOutcome::ShortSpan:        return "short span, grown";
        case InitializeOutcome::AllOutliers:      return "all Tmask outliers, grown";
        case InitializeOutcome::RejectedOutliers: return "Tmask drop rejected, grown";
        case InitializeOutcome::Unstable:         return "unstable, shifted";
        case InitializeOutcome::Stable:           return "stable";
        case InitializeOutcome::DroppedThenUnstable:
            return "Tmask outliers dropped, unstable, shifted";
        case InitializeOutcome::DroppedThenStable:
            return "Tmask outliers dropped, stable";
    }
    return "?";
}

InitializeOutcome initialize_step(
    MaskedSeries& series,
    Window& window,
    const InitializeCriteria& criteria,
    const Variogram& variogram,
    LassoWorkspace& lworkspace,
    LassoSolver& solver,
    std::vector<LassoResult>& models
)
{
    assert(window.size() >= criteria.meow_size);

    //--------------------------------------------------------------------------
    // No candidate left before the end of the series
    //--------------------------------------------------------------------------
    if (!(window.stop + criteria.meow_size < series.size()))
    {
        LOG_DEBUG("Final model window: (" << window.start << ", " << window.stop << ")");
        return InitializeOutcome::Exhausted;
    }

    // The window's observations; a Tmask drop below replaces them.
    auto dates =
        series.dates().slice(range(window.start, window.stop));
    auto spectral =
        series.spectral().slice(all(), range(window.start, window.stop));

    //--------------------------------------------------------------------------
    // Check minimum time span
    //--------------------------------------------------------------------------
    if (!enough_time(dates, criteria.day_delta))
    {
        window.grow();
        return InitializeOutcome::ShortSpan;
    }

    LOG_DEBUG(
        "Checking window ("
        << window.start
        << ", "
        << window.stop
        << ")"
    );
    //--------------------------------------------------------------------------
    // Run TMask
    //--------------------------------------------------------------------------
    auto outliers =
        tmask(
            dates,
            spectral,
            variogram,
            criteria.tmask_bands,
            criteria.t_const
        );
    LOG_DEBUG("Number of Tmask outliers found: " << outliers.count());
    //--------------------------------------------------------------------------
    // TMask removed everything
    //--------------------------------------------------------------------------
    if (outliers.count() == window.size())
    {
        window.grow();
        return InitializeOutcome::AllOutliers;
    }

    //--------------------------------------------------------------------------
    // Update persistent mask
    //--------------------------------------------------------------------------
    bool dropped = false;
    if (outliers.any())
    {
        //----------------------------------------------------------------------
        // Drop them only if the shortened window is still usable; otherwise
        // grow, with nothing dropped
        //----------------------------------------------------------------------
        dropped = series.drop_outliers_if(
            outliers,
            window,
            [&criteria](ArrayView<const std::int64_t, 1> shortened)
            {
                return enough_time(shortened, criteria.day_delta) &&
                    enough_samples(shortened, criteria.meow_size);
            }
        );

        if (!dropped)
        {
            window.grow();
            return InitializeOutcome::RejectedOutliers;
        }

        dates = series.dates().slice(range(window.start, window.stop));
        spectral = series.spectral().slice(all(), range(window.start, window.stop));
    }

    LOG_DEBUG("Generating models to check for stability");
    LOG_DEBUG(
        "window for stability: ("
        << window.start
        << ", "
        << window.stop
        << ")"
    );
    //--------------------------------------------------------------------------
    // Fit the detection bands
    //
    // Only these are read: stable() here and the caller after it. The
    // magnitude of these models is never read.
    //--------------------------------------------------------------------------
    fit_bands(
        dates,
        spectral,
        criteria.num_coefficients,
        criteria.detection_bands,
        KeepResiduals::Yes,
        lworkspace,
        solver,
        models
    );

    //--------------------------------------------------------------------------
    // Stability check
    //--------------------------------------------------------------------------
    if (stable(
            dates,
            variogram,
            models,
            criteria.change_threshold,
            criteria.detection_bands))
    {
        LOG_DEBUG("Stable start found: (" << window.start << ", " << window.stop << ")");
        return dropped ? InitializeOutcome::DroppedThenStable : InitializeOutcome::Stable;
    }

    window.shift();
    LOG_DEBUG("Unstable model, shift window to: (" << window.start << ", " << window.stop << ")");
    return dropped ? InitializeOutcome::DroppedThenUnstable : InitializeOutcome::Unstable;
}

} // namespace ccd
