#include "ccd/procedure/bfp.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <utility>

#include <boost/math/special_functions/beta.hpp>

namespace ccd
{
namespace bfp
{

scalar_t f_upper_tail(scalar_t f, scalar_t d1, scalar_t d2)
{
    assert(d1 > 0 && d2 > 0);
    if (!(f > 0))
        return 1.0;
    if (std::isinf(f))
        return 0.0;

    // P(F > f) = I_y(d2/2, d1/2) at y = d2 / (d2 + d1 f): y computed as
    // that ratio, not as 1 - x, so a large f keeps y's relative precision.
    const scalar_t y = d2 / (d2 + d1 * f);
    return boost::math::ibeta(d2 / 2, d1 / 2, y);
}

scalar_t fisher_tail(scalar_t log_p_sum, index_t bands)
{
    assert(bands > 0);
    if (std::isinf(log_p_sum))
        return 0.0;

    // Q(bands, x) at x = -L, the chi2(2 bands) tail at -2L: e^-x
    // sum_{j < bands} x^j / j!, its terms each the last times x / j.
    const scalar_t x = -log_p_sum;
    scalar_t term = 1.0;
    scalar_t sum = 1.0;
    for (index_t j = 1; j < bands; ++j)
    {
        term *= x / static_cast<scalar_t>(j);
        sum += term;
    }
    return std::exp(log_p_sum) * sum;
}

void Mixture::reset(Inputs inputs)
{
    in_ = std::move(inputs);
    const index_t N = size();
    assert(in_.h.size() == N * N);
    assert(in_.u.size() == in_.bands * N);

    m_ = 0;
    count_.assign(N + 1, 0);
    sum_.assign(N + 1, 0.0);
    single_.assign(N, 0.0);
    members_.assign(N, 0);
    chol_.assign(N * N, 0.0);
    z_.assign(in_.bands * N, 0.0);
    norm2_.assign((N + 1) * in_.bands, 0.0);
    row_.assign(N, 0.0);
}

namespace
{

// Extends the Cholesky factor of H[S,S], S the first `depth` entries of
// `members` (rows of `chol`, row i holding i + 1 entries, N wide), by
// observation t: writes the new row's off-diagonal entries into `row` and
// returns its diagonal entry.
scalar_t border(const std::vector<scalar_t>& h, index_t N, const std::vector<index_t>& members,
                const std::vector<scalar_t>& chol, index_t depth, index_t t, std::vector<scalar_t>& row)
{
    scalar_t sq = 0.0;
    for (index_t i = 0; i < depth; ++i)
    {
        scalar_t v = h[members[i] * N + t];
        for (index_t p = 0; p < i; ++p)
            v -= chol[i * N + p] * row[p];
        v /= chol[i * N + i];
        row[i] = v;
        sq += v * v;
    }
    return std::sqrt(h[t * N + t] - sq);
}

} // namespace

// The subset B (members_[0 .. depth), last member `last`, its factor and
// whitened residuals in place): scores B and m, then descends into each
// B and j that can still precede m.
void Mixture::extend(index_t depth, index_t last)
{
    const index_t N = size();
    const index_t B = in_.bands;
    const index_t m = m_;
    const auto& dates = in_.dates;

    //--------------------------------------------------------------------------
    // A = B and m: one more row, the bands' F, Fisher's combination.
    //--------------------------------------------------------------------------
    const scalar_t d = border(in_.h, N, members_, chol_, depth, m, row_);
    const index_t k = depth + 1;
    scalar_t log_p = 0.0;
    for (index_t b = 0; b < B; ++b)
    {
        scalar_t v = in_.u[b * N + m];
        for (index_t i = 0; i < depth; ++i)
            v -= row_[i] * z_[b * N + i];
        v /= d;
        const scalar_t f = (norm2_[depth * B + b] + v * v) / static_cast<scalar_t>(k);
        log_p += std::log(f_upper_tail(f, static_cast<scalar_t>(k), in_.degrees_of_freedom));
    }
    const scalar_t p = fisher_tail(log_p, B);
    count_[k] += 1;
    sum_[k] += p;
    if (depth == 0)
        single_[m] = p;

    //--------------------------------------------------------------------------
    // B and j, for each j after `last` far enough from it and from m.
    //--------------------------------------------------------------------------
    const index_t first = depth == 0 ? 0 : last + 1;
    for (index_t j = first; j < m; ++j)
    {
        if (dates[m] - dates[j] < in_.min_days)
            break;   // dates do not decrease: no later j is far enough from m
        if (depth > 0 && dates[j] - dates[last] < in_.min_days)
            continue;

        const scalar_t dj = border(in_.h, N, members_, chol_, depth, j, row_);
        for (index_t i = 0; i < depth; ++i)
            chol_[depth * N + i] = row_[i];
        chol_[depth * N + depth] = dj;
        members_[depth] = j;
        for (index_t b = 0; b < B; ++b)
        {
            scalar_t v = in_.u[b * N + j];
            for (index_t i = 0; i < depth; ++i)
                v -= chol_[depth * N + i] * z_[b * N + i];
            v /= dj;
            z_[b * N + depth] = v;
            norm2_[(depth + 1) * B + b] = norm2_[depth * B + b] + v * v;
        }
        extend(depth + 1, j);
    }
}

Mixture::Stats Mixture::add()
{
    assert(m_ < size());
    extend(0, 0);
    ++m_;

    const scalar_t g = in_.good;
    const index_t m = m_;
    Stats s;
    s.m = m;

    const scalar_t empty = std::pow(1.0 - g, static_cast<scalar_t>(m));
    scalar_t weight = empty;
    scalar_t weighted = empty;   // the empty subset's P is 1
    for (index_t k = 1; k <= m; ++k)
    {
        if (count_[k] == 0)
            continue;
        const scalar_t w = std::pow(g, static_cast<scalar_t>(k)) *
                           std::pow(1.0 - g, static_cast<scalar_t>(m - k));
        weight += w * static_cast<scalar_t>(count_[k]);
        weighted += w * sum_[k];
        s.combos += count_[k];
        s.order = k;
    }
    s.total_weight = weight;
    s.p0 = weighted / weight;
    s.top_mean = sum_[s.order] / static_cast<scalar_t>(count_[s.order]);
    return s;
}

Decision decide(const Mixture::Stats& stats, std::int64_t span_days, const Rule& rule)
{
    if (span_days > rule.max_days_for_break)
        return Decision::NoBreak;
    if (span_days < rule.min_days_for_break || stats.m < rule.min_observations)
        return Decision::Continue;

    const scalar_t threshold =
        rule.scale_threshold
            ? rule.threshold * static_cast<scalar_t>(stats.combos) /
                  std::ldexp(1.0, static_cast<int>(stats.order))
            : rule.threshold;

    // The sizes above `order` hold no subset: the test reads only the empty
    // subset's share there, and moves on if that alone is above the
    // threshold, before the no-change cutoff.
    const scalar_t empty_share =
        std::pow(1.0 - rule.good, static_cast<scalar_t>(stats.m)) / stats.total_weight;
    if (stats.order < stats.m && empty_share > threshold)
        return Decision::Continue;

    if (stats.top_mean > rule.no_change_cutoff)
        return Decision::NoBreak;
    if (stats.p0 < threshold)
        return Decision::Break;
    return Decision::Continue;
}

} // namespace bfp
} // namespace ccd
