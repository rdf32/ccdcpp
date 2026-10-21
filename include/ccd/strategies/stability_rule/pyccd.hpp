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
// is below the threshold: change_threshold for the change test's reference
// peek, lowered for its longer peek as the change test's is
// (adjust_change_threshold(), change_probability); with `peek` N (not 0),
// change_threshold for a peek of N, unchanged.
//------------------------------------------------------------------------------
struct PyccdStability final : StabilityRule
{
    struct Params
    {
        scalar_t change_threshold = 15.086272469388987l;
        scalar_t change_probability = 0.99;
        index_t peek = 0;   // 0: the change test's ("change_test")
    };

    static const std::vector<config::Field<Params>>& schema();

    explicit PyccdStability(const Params& p) : p_(p) {}

    Capabilities needs() const override;
    std::unique_ptr<Scratch> make_scratch() const override;
    void prepare(const PeekPlan& peek, index_t detection_bands, Scratch& scratch) const override;
    bool stable(const Model& fit, ArrayView<const std::int64_t, 1> dates,
                const BandNoise& variogram, const BandList& bands,
                const Scratch& scratch) const override;

private:
    Params p_;
};

} // namespace strategies
} // namespace ccd
