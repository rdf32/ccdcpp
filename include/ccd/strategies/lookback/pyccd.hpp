#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "ccd/config/params.hpp"
#include "ccd/strategies/stages.hpp"

namespace ccd
{
namespace strategies
{

// Lookback's peek, read backward from the observation before the window:
// positions start, start - 1, ..., stop + 1 (stop is exclusive and may be -1).
struct ReversePeek
{
    std::ptrdiff_t start;
    std::ptrdiff_t stop;
};

// Two cases, for a window starting at `window_start` > prev_break; neither
// reads below prev_break, into the previous segment:
//
//   * window_start - prev_break > peek: start - 1 down to start - peek, the
//     pixel's peek of observations, as Extend's;
//   * else: start - 1 down to prev_break (stop is -1 when prev_break is 0).
ReversePeek lookback_peek(index_t window_start, index_t prev_break, index_t peek);

//------------------------------------------------------------------------------
// lookback "pyccd": pyccd's Lookback. The peek is lookback_peek()'s, of the
// change test's peek (with `peek` N, not 0, of N); each
// observation's statistic is change_magnitude() of the stability fit's peek
// residuals against its RMSE (nothing is refitted
// while the window grows backward). Change when every statistic is above the
// threshold (change_threshold for the test's reference peek, lowered for a
// longer peek as chi2's is; with `peek` N, unchanged); else Drop when the first is above outlier_threshold;
// else Include.
//------------------------------------------------------------------------------
struct PyccdLookback final : Lookback
{
    struct Params
    {
        scalar_t change_threshold = 15.086272469388987l;
        scalar_t change_probability = 0.99;
        scalar_t outlier_threshold = 35.888186879610423L;
        index_t peek = 0;
    };

    static const std::vector<config::Field<Params>>& schema();

    explicit PyccdLookback(const Params& p) : p_(p) {}

    Capabilities needs() const override;
    std::unique_ptr<Scratch> make_scratch() const override;
    void prepare(const PeekPlan& peek, index_t detection_bands, Scratch& scratch) const override;
    LookbackJudgement judge(Model& fit, ArrayView<const std::int64_t, 1> dates,
                            ArrayView<const scalar_t, 2> spectral, const Window& window,
                            index_t prev_break, const Variogram& variogram,
                            const BandList& bands, Scratch& scratch) const override;

private:
    Params p_;
};

} // namespace strategies
} // namespace ccd
