#include "ccd/strategies/edge_segments/lasso.hpp"

#include <cassert>
#include <utility>

#include "../../core/logger.hpp"
#include "ccd/input/masked_series.hpp"
#include "ccd/strategies/model/lasso.hpp"

namespace ccd
{

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

FitResult fit_whole_series(
    PixelProcedure procedure,
    const PixelSeries& series,
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
        apply_mask(series, mask);

    const auto masked_dates =
        masked_data.dates_view();

    const auto masked_spectral =
        masked_data.spectral_view();

    //----------------------------------------------------------
    // Build result
    //----------------------------------------------------------
    FitResult results;
    ChangeModel result;
    result.start_day = series.dates()(0);
    result.end_day   = series.dates()(series.dates().size() - 1);
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

namespace strategies
{

namespace
{

// One thread's solver and the workspace its fits build their basis in.
class EdgeScratch final : public Scratch
{
public:
    explicit EdgeScratch(const LassoOptions& options) : solver(options) {}

    LassoSolver solver;
    LassoWorkspace workspace{0};
};

} // namespace

const std::vector<config::Field<LassoEdges::Params>>& LassoEdges::schema()
{
    using config::Unit;
    static const std::vector<config::Field<Params>> s = {
        {"coefficients.min", &Params::coefficients_min, Unit::None, config::at_least(1),
         "coefficients of every edge segment's fit", config::Source::Shared, "coefficients.min"},
        {"min_observations", &Params::min_observations, Unit::Observations, config::any(),
         "kept observations a whole-series curve needs", config::Source::Shared,
         "min_observations"},
        {"lasso.max_iter", &Params::lasso_max_iter, Unit::None, config::at_least(0),
         "coordinate descent sweeps at most", config::Source::Shared, "lasso.max_iter"},
        {"lasso.alpha", &Params::lasso_alpha, Unit::ReflectanceDelta, config::at_least(0),
         "the L1 penalty", config::Source::Shared, "lasso.alpha"},
        {"lasso.tolerance", &Params::lasso_tolerance, Unit::None, config::at_least(0),
         "a sweep whose largest update is at most this times max(largest coefficient, "
         "coef_floor) checks the duality gap", config::Source::Shared, "lasso.tolerance"},
        {"lasso.coef_floor", &Params::lasso_coef_floor, Unit::ReflectanceDelta,
         config::at_least(0), "the least coefficient size the tolerance scales",
         config::Source::Shared, "lasso.coef_floor"},
    };
    return s;
}

std::unique_ptr<Scratch> LassoEdges::make_scratch() const
{
    return std::make_unique<EdgeScratch>(
        LassoOptions{p_.lasso_max_iter, p_.lasso_alpha, p_.lasso_tolerance, p_.lasso_coef_floor});
}

ChangeModel LassoEdges::catch_window(ArrayView<const std::int64_t, 1> dates,
                                     ArrayView<const scalar_t, 2> spectral, const Window& window,
                                     CurveQA curve_qa, Scratch& scratch) const
{
    EdgeScratch& s = static_cast<EdgeScratch&>(scratch);
    return catch_segment(dates, spectral, window, p_.coefficients_min, curve_qa, s.workspace,
                         s.solver);
}

FitResult LassoEdges::whole_series(PixelProcedure procedure, const PixelSeries& pixel,
                                   ProcessingMask mask, Scratch& scratch) const
{
    EdgeScratch& s = static_cast<EdgeScratch&>(scratch);
    return fit_whole_series(procedure, pixel, std::move(mask), p_.min_observations,
                            p_.coefficients_min, s.workspace, s.solver);
}

} // namespace strategies
} // namespace ccd
