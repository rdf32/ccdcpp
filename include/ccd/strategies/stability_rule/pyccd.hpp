#pragma once

#include <memory>
#include <vector>

#include "ccd/config/params.hpp"
#include "ccd/strategies/stages.hpp"

namespace ccd
{
namespace strategies
{

//------------------------------------------------------------------------------
// stability_rule "pyccd": pyccd's check. Over the detection bands, the sum of
// squares of
//
//     (|slope * window span| + |first residual| + |last residual|)
//         / max(variogram, RMSE)
//
// is below the threshold: change_threshold for a peek of peek_size
// observations, lowered for a longer peek as the change test's is
// (adjust_change_threshold(), change_probability). The peek is the change
// test's, or with follow_peek false peek_size's (the threshold unchanged).
//------------------------------------------------------------------------------
struct PyccdStability final : StabilityRule
{
    struct Params
    {
        index_t peek_size = 6;
        scalar_t change_threshold = 15.086272469388987l;
        scalar_t change_probability = 0.99;
        bool follow_peek = true;
    };

    static const std::vector<config::Field<Params>>& schema();

    explicit PyccdStability(const Params& p) : p_(p) {}

    Capabilities needs() const override;
    std::unique_ptr<Scratch> make_scratch() const override;
    void prepare(index_t peek, index_t detection_bands, Scratch& scratch) const override;
    bool stable(const Model& fit, ArrayView<const std::int64_t, 1> dates,
                const Variogram& variogram, const BandList& bands,
                const Scratch& scratch) const override;

private:
    Params p_;
};

} // namespace strategies
} // namespace ccd
