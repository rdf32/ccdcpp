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
// the probability that nothing changed. decide() turns it into the test's
// decision at each m.
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
// The terms of P0(m) as the peek grows by one observation at a time:
// reset() takes a peek of N observations (N <= MAX_PEEK), each add() the
// next one, m = 1, 2, ... N, and keeps the count of valid subsets of each
// size (a recurrence over the subsets' last members, no subset scored).
// sum(k) scores the valid subsets of size k when asked, each subset once per
// peek whatever the m (a cache keyed by the subset), so decide() reads the
// sizes it needs, largest first, and a peek decided by its largest subsets
// scores little else.
//
// A subset's Cholesky factor of H[A,A] and its whitened residuals
// z_b = L^-1 u_A,b are built one member at a time along the enumeration
// (adding an observation costs a forward solve, not a factorisation), and
// F_b = |z_b|^2 / k.
//
// Valid subsets are those whose consecutive members are at least min_days
// apart (inclusive), for min_days >= 2: the spacing JuliaCCD's test and its
// presets (10 days) use. (Its code takes another branch for min_days <= 1.)
//------------------------------------------------------------------------------
class Mixture
{
public:

    // The longest peek reset() takes: the cache holds 2^MAX_PEEK subsets.
    static constexpr index_t MAX_PEEK = 20;

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

        // g, and the spacing a subset's consecutive members keep (>= 2).
        scalar_t good = 0.95;
        std::int64_t min_days = 10;
    };

    // The counts after add() has taken observation m (of 1 .. N).
    struct Stats
    {
        index_t m = 0;

        // The largest valid subset's size, and the count of valid subsets
        // of any size (the empty one not counted).
        index_t order = 0;
        std::int64_t combos = 0;

        // sum_{valid A} w(A), the empty subset's (1-g)^m included.
        scalar_t total_weight = 0;
    };

    void reset(Inputs inputs);

    // Takes the next observation; returns the counts over the first m.
    Stats add();

    index_t size() const { return static_cast<index_t>(in_.dates.size()); }
    scalar_t good() const { return in_.good; }

    // The valid subsets of size k (1 .. m) within the first m observations:
    // their count, and the sum of their P.
    std::int64_t count(index_t k) const { return count_[k]; }
    scalar_t sum(index_t k);

    // P({j}) for an observation add() has taken, the single-observation
    // probability.
    scalar_t single(index_t j);

    // P0(m) and the mean P over the largest valid subsets, every size
    // scored: what decide() bounds.
    scalar_t p0();
    scalar_t top_mean();

private:

    scalar_t score(index_t k);
    void enumerate(index_t depth, index_t k, scalar_t& total);

    Inputs in_;
    index_t m_ = 0;   // observations taken

    // ending_[s * N + i]: the valid subsets of size s whose last member is i;
    // count_[s] their sum over i < m. reach_[i]: the longest valid chain
    // from i within the first m.
    std::vector<std::int64_t> ending_;
    std::vector<std::int64_t> count_;
    std::vector<index_t> reach_;

    // P(A) by the subset's bit mask, valid where stamp_ is generation_.
    std::vector<scalar_t> cache_;
    std::vector<std::uint32_t> stamp_;
    std::uint32_t generation_ = 0;

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
// The test's decision at m, JuliaCCD's breakTestBFP() step for step:
//
//   1. a peek spanning more than max_days_for_break days: NoBreak, the test
//      ends;
//   2. fewer than min_observations or min_days_for_break: Continue;
//   3. the sizes from m down: after each, P0 lies between
//          lower = sum / total_weight and
//          upper = (sum + the weight of the sizes not yet scored) / total_weight,
//      sum the weighted P of the empty subset and the sizes scored so far.
//      At the largest valid size, `order`, the mean P of its subsets above
//      no_change_cutoff is NoBreak, the test ends; then upper below the
//      threshold is Break at m, lower above it Continue. The sizes above
//      `order` hold no subset, so the empty subset's share alone can be
//      enough to Continue before the cutoff is read.
//
// The threshold is `threshold`, times combos / 2^order with scale_threshold.
// Once every size is scored lower == upper == P0, so the decision is P0's;
// the bounds settle it early. At the end of the peek a Continue is
// undecided.
//------------------------------------------------------------------------------
struct Rule
{
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
Decision decide(Mixture& mixture, const Mixture::Stats& stats, std::int64_t span_days,
                const Rule& rule);

} // namespace bfp
} // namespace ccd
