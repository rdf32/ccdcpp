#include "ccd/strategies/change_test/chi2.hpp"

#include <cassert>

#include "../../core/logger.hpp"
#include "ccd/core/maths.hpp"
#include "ccd/strategies/pyccd/peek_statistics.hpp"
#include "ccd/strategies/model.hpp"

namespace ccd
{
namespace strategies
{

namespace
{

// The name of chi2's per-observation statistic.
const char* const CHANGE_MAGNITUDE = "change_magnitude";

} // namespace

namespace
{

// One thread's chi2 state: the pixel's threshold (prepare()), and buffers.
// `seasonal` is the fit window's observations by phase in the year, built at
// the first test that reads it after a refit (for refit number
// `seasonal_fit`).
class Chi2Scratch final : public Scratch
{
public:
    scalar_t change_threshold = 0;
    ChangeThresholdMemo thresholds;
    SeasonalOrder seasonal;
    index_t seasonal_fit = 0;
    std::vector<index_t> closest;      // the seasonal RMSE's observations
};

} // namespace

const std::vector<config::Field<Chi2::Params>>& Chi2::schema()
{
    using config::Unit;
    static const std::vector<config::Field<Params>> s = {
        {"peek_size", &Params::peek_size, Unit::Observations, config::at_least(2),
         "observations a test reads at the reference cadence"},
        {"peek_reference_days", &Params::peek_reference_days, Unit::Days,
         config::greater_than(0), "the revisit interval peek_size is for"},
        {"change_threshold", &Params::change_threshold, Unit::None, config::any(),
         "the threshold for a peek of peek_size observations", config::Source::Chi2, "change_probability"},
        {"change_probability", &Params::change_probability, Unit::None,
         config::open_interval(0, 1), "the probability a longer peek's threshold keeps",
         config::Source::Shared, "change_probability"},
        {"model_rmse_up_to", &Params::model_rmse_up_to, Unit::Observations, config::at_least(1),
         "windows of at most this many observations compare against the model's RMSE"},
        {"seasonal_count", &Params::seasonal_count, Unit::Observations, config::at_least(1),
         "fit-window observations the seasonal RMSE reads"},
        {"seasonal_divisor", &Params::seasonal_divisor, Unit::None, config::greater_than(0),
         "the seasonal RMSE is their residuals' norm over this"},
    };
    return s;
}

Capabilities Chi2::needs() const
{
    return Capability::PeekResiduals | Capability::ModelRmse | Capability::InWindowResiduals;
}

Capabilities Chi2::provides() const
{
    return Capability::ComparisonRmse;
}

std::vector<std::string> Chi2::statistics() const
{
    return {CHANGE_MAGNITUDE};
}

std::unique_ptr<Scratch> Chi2::make_scratch() const
{
    return std::make_unique<Chi2Scratch>();
}

PeekPlan Chi2::prepare(ArrayView<const std::int64_t, 1> statistics_dates,
                       index_t detection_bands, Scratch& scratch) const
{
    Chi2Scratch& s = static_cast<Chi2Scratch&>(scratch);

    const index_t peek =
        adjust_peek(
            statistics_dates,
            p_.peek_size,
            p_.peek_reference_days
        );
    assert(peek >= p_.peek_size);

    // Compares the adjusted peek with peek_size and starts from
    // change_threshold.
    s.change_threshold =
        s.thresholds(
            peek,
            p_.peek_size,
            p_.change_threshold,
            detection_bands,
            p_.change_probability
        );

    return PeekPlan{peek, true, 0, p_.peek_size};
}

TestOutcome Chi2::test(const Peek& peek, Scratch& scratch, PeekResult& result) const
{
    Chi2Scratch& s = static_cast<Chi2Scratch&>(scratch);
    const BandList& detection_bands = peek.bands;
    const Window& window = peek.window;
    const Window& peek_window = peek.peek;
    const auto& masked_dates = peek.dates;

    // Indexed by position in the detection bands.
    std::vector<scalar_t>& comp_rmses = result.comparison_rmse;
    comp_rmses.resize(detection_bands.size());
    if (window.stop - window.start <= p_.model_rmse_up_to) {
        LOG_DEBUG("normal residuals and rmse");
        for (index_t i = 0; i < detection_bands.size(); ++i)
        {
            comp_rmses[i] = peek.model.rmse(detection_bands[i]);
        }
    } else {

        LOG_DEBUG("closest day residuals and rmse");
        // The seasonal_count fit-window observations nearest the peek's
        // last in the year, by the fit window's SeasonalOrder.
        if (s.seasonal_fit != peek.fit)
        {
            s.seasonal.build(masked_dates, peek.fit_window);
            s.seasonal_fit = peek.fit;
        }
        std::vector<index_t>& closest_indexes = s.closest;
        s.seasonal.closest(
            masked_dates(peek_window.stop - 1), p_.seasonal_count, closest_indexes);

        LOG_DEBUG("period: " << masked_dates);
        LOG_DEBUG("peek window: (" << peek_window.start << ", " << peek_window.stop << ")");
        LOG_DEBUG("fit window: (" << peek.fit_window.start << ", " << peek.fit_window.stop << ")");
        LOG_DEBUG("closest inds: " << closest_indexes);

        for (index_t i = 0; i < detection_bands.size(); ++i)
        {
            comp_rmses[i] = seasonal_rmse(peek.model.residuals(detection_bands[i]), closest_indexes, p_.seasonal_divisor);
        }
    }

    std::vector<scalar_t>& magnitudes = result.provide(CHANGE_MAGNITUDE);
    change_magnitude(
        peek.residuals,
        detection_bands,
        peek.variogram,
        comp_rmses,
        magnitudes
    );

    return detect_change(magnitudes, s.change_threshold) ? TestOutcome::Change
                                                         : TestOutcome::NoChange;
}

index_t Chi2::break_at(const Peek& peek, const Scratch&) const
{
    return peek.peek.start;
}

scalar_t Chi2::magnitude(const Peek& peek, const Scratch&, index_t band) const
{
    // median() orders its argument in place; the peek's residuals stay as
    // they are.
    std::vector<scalar_t> residuals = peek.residuals[band];
    return median(residuals);
}

} // namespace strategies
} // namespace ccd
