#include "ccd/strategies/stability_rule/pyccd.hpp"

#include <algorithm>
#include <cmath>

#include "ccd/strategies/model.hpp"

namespace ccd
{
namespace strategies
{

namespace
{

// One thread's pixel threshold (prepare()).
class StabilityScratch final : public Scratch
{
public:
    scalar_t change_threshold = 0;
    ChangeThresholdMemo thresholds;
};

} // namespace

const std::vector<config::Field<PyccdStability::Params>>& PyccdStability::schema()
{
    using config::Unit;
    static const std::vector<config::Field<Params>> s = {
        {"change_threshold", &Params::change_threshold, Unit::None, config::any(),
         "the threshold for the change test's reference peek (or `peek`)", config::Source::Chi2,
         "change_probability"},
        {"change_probability", &Params::change_probability, Unit::None,
         config::open_interval(0, 1), "the probability a longer peek's threshold keeps",
         config::Source::Shared, "change_probability"},
        {"peek", &Params::peek, Unit::Observations, config::any(),
         "0: the change test's peek; N: a peek of N, change_threshold for it"},
    };
    return s;
}

Capabilities PyccdStability::needs() const
{
    return Capability::ModelRmse | Capability::InWindowResiduals;
}

std::unique_ptr<Scratch> PyccdStability::make_scratch() const
{
    return std::make_unique<StabilityScratch>();
}

void PyccdStability::prepare(const PeekPlan& peek, index_t detection_bands,
                             Scratch& scratch) const
{
    StabilityScratch& s = static_cast<StabilityScratch&>(scratch);
    const index_t at = p_.peek ? p_.peek : peek.peek;
    const index_t reference = p_.peek ? p_.peek : (peek.reference ? peek.reference : peek.peek);
    s.change_threshold = s.thresholds(at, reference, p_.change_threshold, detection_bands,
                                      p_.change_probability);
}

bool PyccdStability::stable(const Model& fit, ArrayView<const std::int64_t, 1> dates_window,
                            const Variogram& variogram, const BandList& detection_bands,
                            const Scratch& scratch) const
{
    const scalar_t change_threshold = static_cast<const StabilityScratch&>(scratch).change_threshold;

    scalar_t euclidean_norm = 0.0;
    for (const auto idx: detection_bands) {

        const std::vector<scalar_t>& residuals = fit.residuals(idx);

         const scalar_t rmse_norm =
            std::max(variogram[idx], fit.rmse(idx));

        const scalar_t slope =
            fit.band_fit(idx).coefficients[0] *
            (dates_window(dates_window.size() - 1) - dates_window(0));

        const scalar_t first_resid =
            residuals[0];

        const scalar_t last_resid =
            residuals[
                residuals.size() - 1
            ];

        const scalar_t check_val =
            (std::abs(slope)
            + std::abs(first_resid)
            + std::abs(last_resid))
            / rmse_norm;

        euclidean_norm += (check_val * check_val); // sum of squares
    }

    return (euclidean_norm < change_threshold);
}

} // namespace strategies
} // namespace ccd
