#pragma once

#include <cstdint>
#include <vector>

#include "ccd/types.hpp"

namespace ccd
{
namespace bfp
{

//==============================================================================
//
// The Band-First Probability change test (Tollerud et al. 2023, appendix A.3
// and D), as kernels.
//
// A model fitted by ordinary least squares on n observations with q
// coefficients predicts the next N observations of a peek window. For a
// subset S of k of them and a band b, with residuals r and the model's
// variance MSE_b = SSR_b / (n - q),
//
//     F_b(S) = r_S' H[S,S]^-1 r_S / (k MSE_b),   H = I + X_p (X'X)^-1 X_p'
//
// is F(k, n - q) distributed when nothing changed (Chow's forecast test: H
// is the prediction errors' covariance in units of the variance), and
//
//     p_b(S) = P(F(k, n - q) > F_b(S)).
//
// The bands combine by Fisher's method: P(S) = P(chi2(2B) > -2 sum_b ln p_b).
// Each observation is "good" with probability g, and the subsets the test
// mixes over are those whose consecutive members are at least min_days
// apart; for the first m peek observations, with w(A) = g^|A| (1-g)^(m-|A|)
// and the empty subset's P = 1,
//
//     P0(m) = sum_{valid A} w(A) P(A) / sum_{valid A} w(A),
//
// the probability that nothing changed. decide() turns P0 and the subsets'
// counts into the test's decision at each m.
//
//==============================================================================

// P(F(d1, d2) > f), the F distribution's upper tail; f < 0 is read as 0
// (tail 1). d1, d2 > 0.
scalar_t f_upper_tail(scalar_t f, scalar_t d1, scalar_t d2);

// Fisher's combination of `bands` probabilities whose logarithms sum to
// log_p_sum (<= 0): P(chi2(2 bands) > -2 log_p_sum), exactly
// e^L sum_{j < bands} (-L)^j / j! for L = log_p_sum; 0 when L is -infinity
// (some band's p is 0).
scalar_t fisher_tail(scalar_t log_p_sum, index_t bands);

//------------------------------------------------------------------------------
// Mixture
//
// P0(m) as the peek grows by one observation at a time. reset() takes a
// peek of N observations; each add() takes the next one, m = 1, 2, ... N,
// and computes P(A) for the valid subsets that contain it (the others are
// the previous add()'s), so a whole peek costs each valid subset once.
//
// A subset's Cholesky factor of H[A,A] and its whitened residuals
// z_b = L^-1 u_A,b are extended by one row at a time (adding an observation
// to a subset costs a forward solve, not a factorisation), and
// F_b = |z_b|^2 / k.
//------------------------------------------------------------------------------
class Mixture
{
public:

    struct Inputs
    {
        // H = I + X_p (X'X)^-1 X_p', N x N, row-major, symmetric positive
        // definite.
        std::vector<scalar_t> h;

        // The residuals scaled by 1 / sqrt(MSE_b): bands x N, row-major.
        std::vector<scalar_t> u;
        index_t bands = 0;

        // The peek's dates, N, non-decreasing.
        std::vector<std::int64_t> dates;

        // n - q.
        scalar_t degrees_of_freedom = 0;

        // g, and the spacing a subset's consecutive members keep.
        scalar_t good = 0.95;
        std::int64_t min_days = 10;
    };

    // The state after add() has taken observation m (of 1 .. N).
    struct Stats
    {
        index_t m = 0;

        // The largest valid subset's size, and the count of valid subsets
        // of any size (the empty one not counted).
        index_t order = 0;
        std::int64_t combos = 0;

        // sum_{valid A} w(A), the empty subset's (1-g)^m included.
        scalar_t total_weight = 0;

        // P0(m), and the mean P over the valid subsets of size `order`.
        scalar_t p0 = 1;
        scalar_t top_mean = 1;
    };

    void reset(Inputs inputs);

    // Takes the next observation; returns the state over the first m.
    Stats add();

    index_t size() const { return static_cast<index_t>(in_.dates.size()); }

    // P({j}) of an observation add() has taken, the test's single-observation
    // probability.
    scalar_t single(index_t j) const { return single_[j]; }

private:

    void extend(index_t depth, index_t last);

    Inputs in_;
    index_t m_ = 0;   // observations taken

    // By subset size k (index k, 1 .. N): the valid subsets' count and the
    // sum of their P.
    std::vector<std::int64_t> count_;
    std::vector<scalar_t> sum_;
    std::vector<scalar_t> single_;

    // The enumeration's working rows, one per depth: the subset's members,
    // its Cholesky factor (row d holds d + 1 entries), the whitened
    // residuals and their squared norms per band.
    std::vector<index_t> members_;
    std::vector<scalar_t> chol_;
    std::vector<scalar_t> z_;
    std::vector<scalar_t> norm2_;
    std::vector<scalar_t> row_;
};

//------------------------------------------------------------------------------
// decide
//
// The test's decision at m, from add()'s state, as JuliaCCD's
// breakTestBFP() takes it (its bounds on P0 are an economy and decide the
// same):
//
//   1. a peek spanning more than max_days_for_break days: NoBreak, the test
//      ends;
//   2. fewer than min_observations or min_days_for_break: Continue;
//   3. when the largest valid subset is smaller than m and the empty
//      subset's share (1-g)^m / total_weight alone is above the threshold:
//      Continue, before 4;
//   4. top_mean above no_change_cutoff: NoBreak, the test ends;
//   5. P0 below the threshold: Break at m;
//   6. otherwise Continue.
//
// The threshold is `threshold`, times combos / 2^order with scale_threshold.
// At the end of the peek a Continue is undecided.
//------------------------------------------------------------------------------
struct Rule
{
    scalar_t good = 0.95;
    scalar_t threshold = 1e-10;
    bool scale_threshold = true;
    scalar_t no_change_cutoff = 0.001;
    index_t min_observations = 1;
    std::int64_t min_days_for_break = 0;
    std::int64_t max_days_for_break = 10000;
};

enum class Decision : std::uint8_t
{
    Continue,
    NoBreak,
    Break
};

// `span_days` is the peek's last taken date minus its first.
Decision decide(const Mixture::Stats& stats, std::int64_t span_days, const Rule& rule);

} // namespace bfp
} // namespace ccd
