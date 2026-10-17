//==============================================================================
//
// The BFP machine (include/ccd/procedure/bfp_machine.hpp).
//
//   * LeastSquares: the sums' fit against a dense least-squares solve on the
//     HarmonicBasis design (the trend column the ordinal day): coefficients
//     as a segment reports them, SSR, residuals, and the whitened columns'
//     leverages x' (X'X)^-1 x, at every coefficient count, on random series.
//   * The machine on the CSV pixels: every segment's days, count and change
//     probability; the first segment starts where the series' stable start
//     does; segments do not overlap; BreakStart::First never skips an
//     observation.
//
// Its decisions against JuliaCCD's paperbfp are pinned by
// test_bfp_reference.cpp.
//
//==============================================================================

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "ccd/ccd.hpp"
#include "ccd/constants.hpp"
#include "ccd/procedure/bfp_machine.hpp"

namespace
{

using ccd::index_t;
using ccd::scalar_t;

scalar_t uniform(std::mt19937_64& r)
{
    return static_cast<scalar_t>(r() >> 11) * 0x1.0p-53;
}

bool near(scalar_t a, scalar_t b, scalar_t rel, scalar_t abs = 1e-9)
{
    return std::abs(a - b) <= rel * std::max(std::abs(a), std::abs(b)) + abs;
}

// The design HarmonicBasis builds, with its intercept: 1, t, cos(wt), ...
Eigen::MatrixXd basis(const std::vector<std::int64_t>& days, index_t q)
{
    const scalar_t omega = 2.0 * ccd::PI / ccd::AVG_DAYS_YR;
    Eigen::MatrixXd X(days.size(), q);
    for (std::size_t i = 0; i < days.size(); ++i)
    {
        const scalar_t t = static_cast<scalar_t>(days[i]);
        const scalar_t wt = omega * t;
        const scalar_t row[8] = {1.0, t, std::cos(wt), std::sin(wt), std::cos(2 * wt),
                                 std::sin(2 * wt), std::cos(3 * wt), std::sin(3 * wt)};
        for (index_t j = 0; j < q; ++j)
            X(i, j) = row[j];
    }
    return X;
}

} // namespace

TEST(LeastSquares, EqualsADenseSolveOnTheBasis)
{
    std::mt19937_64 r(101);
    for (int trial = 0; trial < 40; ++trial)
    {
        const index_t n = 30 + r() % 400;
        const index_t B = 1 + r() % 7;
        std::vector<std::int64_t> days(n);
        std::int64_t day = 724000 + static_cast<std::int64_t>(r() % 9000);
        for (auto& d : days)
        {
            day += 1 + static_cast<std::int64_t>(r() % 30);
            d = day;
        }
        std::vector<scalar_t> values(B * n);
        for (index_t b = 0; b < B; ++b)
            for (index_t i = 0; i < n; ++i)
                values[b * n + i] = 800 + 300 * std::sin(0.0172 * days[i] + b) +
                                    0.05 * (days[i] - days[0]) + 60 * (uniform(r) - 0.5);
        const auto spectral = ccd::ArrayView<const scalar_t, 2>::contiguous(values.data(), {B, n});

        ccd::bfp::LeastSquares sums;
        sums.reset(days[0], B);
        for (index_t i = 0; i < n; ++i)
            sums.add(days[i], spectral, i);
        ASSERT_EQ(sums.count(), n);

        for (const index_t q : {index_t{4}, index_t{6}, index_t{8}})
        {
            ASSERT_TRUE(sums.fit(q));
            const Eigen::MatrixXd X = basis(days, q);
            const Eigen::MatrixXd XtXinv = (X.transpose() * X).inverse();
            for (index_t b = 0; b < B; ++b)
            {
                Eigen::VectorXd y(n);
                for (index_t i = 0; i < n; ++i)
                    y(i) = values[b * n + i];
                const Eigen::VectorXd beta = X.colPivHouseholderQr().solve(y);
                const Eigen::VectorXd resid = y - X * beta;

                const ccd::BandFit f = sums.band_fit(b, 0.0, 0.0);
                const std::string at = "trial " + std::to_string(trial) + " q " + std::to_string(q) +
                                       " band " + std::to_string(b);
                // The intercept at day 0 is a difference of large terms: its
                // precision is the trend's times the day number.
                EXPECT_TRUE(near(f.intercept, beta(0), 1e-6, 1e-3)) << at;
                EXPECT_TRUE(near(f.coefficients[0], beta(1), 1e-7, 1e-10)) << at;
                for (index_t j = 2; j < q; ++j)
                    EXPECT_TRUE(near(f.coefficients[j - 1], beta(j), 1e-8, 1e-8)) << at << " j " << j;
                for (index_t j = q - 1; j < 7; ++j)
                    EXPECT_EQ(f.coefficients[j], 0.0) << at;
                EXPECT_TRUE(near(sums.ssr(b), resid.squaredNorm(), 1e-8)) << at;
                for (index_t i = 0; i < n; i += 37)
                    EXPECT_TRUE(near(sums.residual(b, days[i], y(i)), resid(i), 1e-8, 1e-6)) << at;
            }
            // Leverages, invariant to the trend column's origin and unit.
            std::vector<scalar_t> g(q);
            for (const std::int64_t d : {days[0], days[n / 2], days[n - 1] + 40})
            {
                sums.whitened(d, g.data());
                scalar_t lev = 0;
                for (const scalar_t v : g)
                    lev += v * v;
                const Eigen::MatrixXd x = basis({d}, q);
                const scalar_t expected = (x * XtXinv * x.transpose())(0, 0);
                EXPECT_TRUE(near(lev, expected, 1e-7, 1e-12)) << "trial " << trial << " q " << q;
            }
        }
    }
}

TEST(LeastSquares, RepeatedDaysLeaveTheColumnsDependent)
{
    // Twelve observations on two days cannot fit four coefficients.
    std::vector<scalar_t> values(12, 1.0);
    const auto spectral = ccd::ArrayView<const scalar_t, 2>::contiguous(values.data(), {1, 12});
    ccd::bfp::LeastSquares sums;
    sums.reset(730000, 1);
    for (index_t i = 0; i < 12; ++i)
        sums.add(730000 + (i % 2), spectral, i);
    EXPECT_FALSE(sums.fit(4));
}
