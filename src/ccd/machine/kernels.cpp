#include "ccd/machine/kernels.hpp"

#include <algorithm>
#include <cassert>
#include <limits>
#include <utility>

#include "../core/logger.hpp"
#include "ccd/strategies/model.hpp"
#include "ccd/strategies/stages.hpp"

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
    fits.resize(spectral.extent(0));
    fit_bands(dates, spectral, num_coefficients, every_band(spectral.extent(0)), keep,
              lworkspace, solver, fits);
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
    const InitializeStages& stages
)
{
    const StartSearch& search = stages.search;
    assert(window.size() >= search.min_window());

    //--------------------------------------------------------------------------
    // No candidate left before the end of the series
    //--------------------------------------------------------------------------
    const index_t seen = criteria.last_day == std::numeric_limits<std::int64_t>::max()
        ? series.size()
        : count_through(series.dates(), criteria.last_day);
    if (search.exhausted(window, seen))
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
    if (search.short_span(dates))
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
    // The window screen (Tmask)
    //--------------------------------------------------------------------------
    auto outliers = stages.screen.screen(dates, spectral, variogram);
    LOG_DEBUG("Number of Tmask outliers found: " << outliers.count());
    //--------------------------------------------------------------------------
    // The screen removed everything
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
            [&search](ArrayView<const std::int64_t, 1> shortened)
            {
                return search.usable(shortened);
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
    // Only these are read: the stability rule here and Lookback after it.
    //--------------------------------------------------------------------------
    stages.fit.fit(
        dates,
        spectral,
        criteria.num_coefficients,
        criteria.max_coefficients,
        criteria.detection_bands,
        KeepResiduals::Yes
    );

    //--------------------------------------------------------------------------
    // Stability check
    //--------------------------------------------------------------------------
    if (stages.rule.stable(
            stages.fit,
            dates,
            variogram,
            criteria.detection_bands,
            stages.rule_scratch))
    {
        LOG_DEBUG("Stable start found: (" << window.start << ", " << window.stop << ")");
        return dropped ? InitializeOutcome::DroppedThenStable : InitializeOutcome::Stable;
    }

    window.shift();
    LOG_DEBUG("Unstable model, shift window to: (" << window.start << ", " << window.stop << ")");
    return dropped ? InitializeOutcome::DroppedThenUnstable : InitializeOutcome::Unstable;
}

namespace
{

// One band's peek residuals, in place: the prediction, then |y - prediction|.
void peek_band_residuals(
    const HarmonicBasis& peek_basis,
    ArrayView<const scalar_t, 1> y,
    const LassoModel& model,
    std::vector<scalar_t>& residuals
)
{
    residuals.resize(y.size());

    model.predict(
        peek_basis.X(),
        residuals
    );

    for (index_t i = 0; i < y.size(); ++i)
    {
        residuals[i] =
            std::abs(y(i) - residuals[i]);
    }
}

} // namespace

void peek_residuals(
    const HarmonicBasis& peek_basis,
    ArrayView<const scalar_t, 2> peek_spectral,
    const std::vector<LassoResult>& models,
    const BandList& bands,
    std::vector<std::vector<scalar_t>>& residuals
)
{
    assert(peek_basis.samples() == peek_spectral.extent(1));

    for (const index_t band : bands)
    {
        peek_band_residuals(
            peek_basis,
            peek_spectral.slice(fixed(band), all()),
            models[band].model,
            residuals[band]
        );
    }
}

} // namespace ccd
