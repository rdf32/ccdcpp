//==============================================================================
//
// The BFP test's kernels (include/ccd/procedure/bfp.hpp).
//
//   * f_upper_tail(): the F tail against its closed form at d1 = 2 and
//     against Boost's F distribution, at the edges 0, negative and infinite.
//   * fisher_tail(): against Boost's chi-squared distribution.
//   * Mixture: every add() against brute force -- every subset of the first
//     m observations tested for the spacing, each P(A) from a dense inverse
//     of H[A,A], the counts and sums by size, P0, the top-order mean -- on
//     random peeks with gaps under and over min_days and repeated dates.
//   * decide(): its bounds against the exact rule over every size, at every
//     m of random peeks, including dense peeks whose largest subset is
//     smaller than m.
//
//==============================================================================

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/fisher_f.hpp>

#include "ccd/procedure/bfp.hpp"

namespace
{

using ccd::index_t;
using ccd::scalar_t;
namespace bfp = ccd::bfp;

// The machine-independent part of a random draw: mt19937_64's output is
// specified, the standard distributions' mapping is not.
scalar_t uniform(std::mt19937_64& r)
{
    return static_cast<scalar_t>(r() >> 11) * 0x1.0p-53;
}

bool near(scalar_t a, scalar_t b, scalar_t rel)
{
    return std::abs(a - b) <= rel * std::max({std::abs(a), std::abs(b), 1e-300});
}

} // namespace

TEST(FUpperTail, ClosedFormBoostAndEdges)
{
    // d1 = 2: P(F > f) = (d2 / (d2 + 2 f))^(d2 / 2).
    for (const scalar_t d2 : {1.0, 7.0, 40.0, 900.0})
        for (const scalar_t f : {1e-6, 0.3, 1.0, 4.5, 30.0, 400.0})
            EXPECT_TRUE(near(bfp::f_upper_tail(f, 2, d2), std::pow(d2 / (d2 + 2 * f), d2 / 2), 1e-13))
                << d2 << " " << f;

    std::mt19937_64 r(91);
    for (int i = 0; i < 2000; ++i)
    {
        const scalar_t d1 = 1 + static_cast<scalar_t>(r() % 18);
        const scalar_t d2 = 4 + static_cast<scalar_t>(r() % 2000);
        const scalar_t f = std::exp(uniform(r) * 8 - 4);
        const scalar_t expected = boost::math::cdf(
            boost::math::complement(boost::math::fisher_f_distribution<scalar_t>(d1, d2), f));
        EXPECT_TRUE(near(bfp::f_upper_tail(f, d1, d2), expected, 1e-12)) << d1 << " " << d2 << " " << f;
    }

    EXPECT_EQ(bfp::f_upper_tail(0.0, 3, 50), 1.0);
    EXPECT_EQ(bfp::f_upper_tail(-2.0, 3, 50), 1.0);
    EXPECT_EQ(bfp::f_upper_tail(INFINITY, 3, 50), 0.0);
    // A tail far below double's epsilon keeps its relative precision.
    EXPECT_TRUE(near(bfp::f_upper_tail(1e6, 2, 30), std::pow(30.0 / (30.0 + 2e6), 15.0), 1e-12));
}

TEST(FisherTail, ChiSquaredWithTwiceTheBands)
{
    std::mt19937_64 r(92);
    for (int i = 0; i < 2000; ++i)
    {
        const index_t bands = 1 + r() % 10;
        const scalar_t log_p = -std::exp(uniform(r) * 9 - 4);   // -0.02 .. -150
        const scalar_t expected = boost::math::cdf(boost::math::complement(
            boost::math::chi_squared_distribution<scalar_t>(2.0 * bands), -2 * log_p));
        EXPECT_TRUE(near(bfp::fisher_tail(log_p, bands), expected, 1e-12)) << bands << " " << log_p;
    }
    EXPECT_EQ(bfp::fisher_tail(0.0, 5), 1.0);
    EXPECT_EQ(bfp::fisher_tail(-INFINITY, 5), 0.0);
}

namespace
{

// A random peek: H = I + G'G (G q x N), u bands x N, dates with gaps from 0
// to 25 days.
bfp::Mixture::Inputs random_peek(std::mt19937_64& r, index_t N, index_t bands)
{
    bfp::Mixture::Inputs in;
    const index_t q = 4 + r() % 5;
    Eigen::MatrixXd G(q, N);
    for (index_t i = 0; i < q; ++i)
        for (index_t j = 0; j < N; ++j)
            G(i, j) = (uniform(r) - 0.5) * 0.8;
    const Eigen::MatrixXd H = Eigen::MatrixXd::Identity(N, N) + G.transpose() * G;
    in.h.resize(N * N);
    for (index_t i = 0; i < N; ++i)
        for (index_t j = 0; j < N; ++j)
            in.h[i * N + j] = H(i, j);

    in.bands = bands;
    in.u.resize(bands * N);
    const scalar_t shift = (r() % 3 == 0) ? 4.0 : 0.0;   // some peeks look like a change
    for (auto& v : in.u)
        v = (uniform(r) - 0.5) * 3.0 + shift;

    std::int64_t day = 730000;
    for (index_t j = 0; j < N; ++j)
    {
        in.dates.push_back(day);
        day += static_cast<std::int64_t>(r() % 26);
    }
    in.degrees_of_freedom = 10 + static_cast<scalar_t>(r() % 400);
    in.good = (r() % 2) ? 0.95 : 0.97;
    in.min_days = (r() % 4 == 0) ? 2 : 10;
    return in;
}

struct Brute
{
    std::vector<std::int64_t> count;
    std::vector<scalar_t> sum;
};

// Every subset of the first m observations: valid when its consecutive
// members are min_days apart; P(A) from a dense solve.
Brute brute(const bfp::Mixture::Inputs& in, index_t m)
{
    const index_t N = in.dates.size();
    Brute out{std::vector<std::int64_t>(m + 1, 0), std::vector<scalar_t>(m + 1, 0.0)};
    for (std::uint32_t mask = 1; mask < (1u << m); ++mask)
    {
        std::vector<index_t> a;
        for (index_t j = 0; j < m; ++j)
            if (mask & (1u << j))
                a.push_back(j);
        bool valid = true;
        for (std::size_t i = 1; i < a.size(); ++i)
            valid = valid && in.dates[a[i]] - in.dates[a[i - 1]] >= in.min_days;
        if (!valid)
            continue;

        const index_t k = a.size();
        Eigen::MatrixXd Hs(k, k);
        for (index_t i = 0; i < k; ++i)
            for (index_t j = 0; j < k; ++j)
                Hs(i, j) = in.h[a[i] * N + a[j]];
        const Eigen::MatrixXd inv = Hs.inverse();
        scalar_t log_p = 0;
        for (index_t b = 0; b < in.bands; ++b)
        {
            Eigen::VectorXd u(k);
            for (index_t i = 0; i < k; ++i)
                u(i) = in.u[b * N + a[i]];
            const scalar_t f = u.dot(inv * u) / static_cast<scalar_t>(k);
            log_p += std::log(bfp::f_upper_tail(f, static_cast<scalar_t>(k), in.degrees_of_freedom));
        }
        out.count[k] += 1;
        out.sum[k] += bfp::fisher_tail(log_p, in.bands);
    }
    return out;
}

} // namespace

TEST(Mixture, EveryAddEqualsBruteForce)
{
    std::mt19937_64 r(93);
    index_t peeks = 0, subsets = 0;
    for (int trial = 0; trial < 60; ++trial)
    {
        const index_t N = 1 + r() % 12;
        const bfp::Mixture::Inputs in = random_peek(r, N, 1 + r() % 6);
        bfp::Mixture mixture;
        mixture.reset(in);
        for (index_t m = 1; m <= N; ++m)
        {
            const bfp::Mixture::Stats s = mixture.add();
            const Brute b = brute(in, m);
            const scalar_t g = in.good;
            scalar_t weight = std::pow(1 - g, m), weighted = weight;
            std::int64_t combos = 0;
            index_t order = 0;
            const std::string at = "trial " + std::to_string(trial) + " m " + std::to_string(m);
            for (index_t k = 1; k <= m; ++k)
            {
                EXPECT_EQ(mixture.count(k), b.count[k]) << at << " k " << k;
                EXPECT_TRUE(near(mixture.sum(k), b.sum[k], 1e-11)) << at << " k " << k;
                if (b.count[k] == 0)
                    continue;
                const scalar_t w = std::pow(g, k) * std::pow(1 - g, m - k);
                weight += w * b.count[k];
                weighted += w * b.sum[k];
                combos += b.count[k];
                order = k;
            }
            EXPECT_EQ(s.m, m) << at;
            EXPECT_EQ(s.order, order) << at;
            EXPECT_EQ(s.combos, combos) << at;
            EXPECT_TRUE(near(s.total_weight, weight, 1e-13)) << at;
            EXPECT_TRUE(near(mixture.p0(), weighted / weight, 1e-11)) << at;
            EXPECT_TRUE(near(mixture.top_mean(), b.sum[order] / b.count[order], 1e-11)) << at;
            subsets += combos;
        }

        // P({j}), the single-observation probability: H[j,j] is its variance.
        for (index_t j = 0; j < N; ++j)
        {
            scalar_t log_p = 0;
            for (index_t b = 0; b < in.bands; ++b)
            {
                const scalar_t f = in.u[b * N + j] * in.u[b * N + j] / in.h[j * N + j];
                log_p += std::log(bfp::f_upper_tail(f, 1, in.degrees_of_freedom));
            }
            EXPECT_TRUE(near(mixture.single(j), bfp::fisher_tail(log_p, in.bands), 1e-12));
        }
        ++peeks;
    }
    EXPECT_EQ(peeks, 60);
    EXPECT_GT(subsets, 5000);
}

namespace
{

// The decision at m from every size: the span and count guards; the empty
// subset's share alone when the largest subset is smaller than m; the
// no-change cutoff on the top-order mean; P0 against the threshold.
bfp::Decision exact_rule(bfp::Mixture& mixture, const bfp::Mixture::Stats& s,
                         std::int64_t span, const bfp::Rule& rule)
{
    if (span > rule.max_days_for_break)
        return bfp::Decision::NoBreak;
    if (span < rule.min_days_for_break || s.m < rule.min_observations)
        return bfp::Decision::Continue;
    const scalar_t threshold =
        rule.scale_threshold ? rule.threshold * s.combos / std::ldexp(1.0, s.order) : rule.threshold;
    if (s.order < s.m && std::pow(1 - mixture.good(), s.m) / s.total_weight > threshold)
        return bfp::Decision::Continue;
    if (mixture.top_mean() > rule.no_change_cutoff)
        return bfp::Decision::NoBreak;
    if (mixture.p0() < threshold)
        return bfp::Decision::Break;
    return bfp::Decision::Continue;
}

} // namespace

TEST(Decide, TheBoundsDecideAsTheExactRule)
{
    std::mt19937_64 r(94);
    std::size_t decided = 0, breaks = 0, cutoffs = 0, dense = 0, undecided = 0;
    for (int trial = 0; trial < 400; ++trial)
    {
        const index_t N = 1 + r() % 16;
        bfp::Mixture::Inputs in = random_peek(r, N, 5);
        if (trial % 3 == 0)   // a clear change: every residual shifted
            for (auto& v : in.u)
                v += 6.0;
        bfp::Rule rule;
        rule.threshold = (trial % 5 == 0) ? 1e-6 : 1e-10;
        rule.scale_threshold = trial % 7 != 0;

        bfp::Mixture bounded, exact;
        bounded.reset(in);
        exact.reset(in);
        bfp::Decision last = bfp::Decision::Continue;
        for (index_t m = 1; m <= N; ++m)
        {
            const bfp::Mixture::Stats s = bounded.add();
            const bfp::Mixture::Stats e = exact.add();
            const std::int64_t span = in.dates[m - 1] - in.dates[0];
            const bfp::Decision d = bfp::decide(bounded, s, span, rule);
            ASSERT_EQ(d, exact_rule(exact, e, span, rule))
                << "trial " << trial << " m " << m << " p0 " << exact.p0();
            if (s.order < s.m)
                ++dense;
            last = d;
            if (d != bfp::Decision::Continue)
                break;
        }
        ++decided;
        breaks += last == bfp::Decision::Break;
        cutoffs += last == bfp::Decision::NoBreak;
        undecided += last == bfp::Decision::Continue;
    }
    EXPECT_EQ(decided, 400u);
    EXPECT_GT(breaks, 20u);
    EXPECT_GT(cutoffs, 20u);
    EXPECT_GT(undecided, 5u);
    EXPECT_GT(dense, 50u);
}

TEST(Decide, TheGuardsComeFirst)
{
    std::mt19937_64 r(95);
    bfp::Mixture::Inputs in = random_peek(r, 8, 5);
    for (auto& v : in.u)
        v += 8.0;   // a change no size could miss
    bfp::Mixture mixture;
    mixture.reset(in);
    bfp::Mixture::Stats s;
    for (index_t m = 1; m <= 8; ++m)
        s = mixture.add();

    bfp::Rule rule;
    EXPECT_EQ(bfp::decide(mixture, s, 10001, rule), bfp::Decision::NoBreak);
    bfp::Rule more = rule;
    more.min_observations = 9;
    EXPECT_EQ(bfp::decide(mixture, s, 100, more), bfp::Decision::Continue);
    more = rule;
    more.min_days_for_break = 101;
    EXPECT_EQ(bfp::decide(mixture, s, 100, more), bfp::Decision::Continue);
}
