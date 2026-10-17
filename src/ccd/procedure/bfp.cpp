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
    assert(N <= MAX_PEEK);
    assert(in_.min_days >= 2);
    assert(in_.h.size() == N * N);
    assert(in_.u.size() == in_.bands * N);

    m_ = 0;
    ending_.assign((N + 1) * N, 0);
    count_.assign(N + 1, 0);
    reach_.assign(N, 0);

    const std::size_t subsets = std::size_t{1} << N;
    if (cache_.size() < subsets)
    {
        cache_.resize(subsets);
        stamp_.assign(subsets, 0);
        generation_ = 0;
    }
    if (++generation_ == 0)
    {
        std::fill(stamp_.begin(), stamp_.end(), 0);
        generation_ = 1;
    }

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

Mixture::Stats Mixture::add()
{
    assert(m_ < size());
    const index_t N = size();
    const index_t i = m_;
    const auto& dates = in_.dates;

    // The subsets ending at i: {i}, and each size s - 1 one ending at a j at
    // least min_days before it, with i appended.
    ending_[1 * N + i] = 1;
    for (index_t s = 2; s <= i + 1; ++s)
    {
        std::int64_t c = 0;
        for (index_t j = 0; j < i; ++j)
            if (dates[i] - dates[j] >= in_.min_days)
                c += ending_[(s - 1) * N + j];
        ending_[s * N + i] = c;
    }
    for (index_t s = 1; s <= i + 1; ++s)
        count_[s] += ending_[s * N + i];
    ++m_;

    // The longest valid chain from each observation within the first m.
    for (index_t j = m_; j-- > 0;)
    {
        index_t longest = 0;
        for (index_t t = j + 1; t < m_; ++t)
            if (dates[t] - dates[j] >= in_.min_days)
                longest = std::max(longest, reach_[t]);
        reach_[j] = 1 + longest;
    }

    const scalar_t g = in_.good;
    Stats s;
    s.m = m_;
    s.total_weight = std::pow(1.0 - g, static_cast<scalar_t>(m_));
    for (index_t k = 1; k <= m_; ++k)
    {
        if (count_[k] == 0)
            continue;
        s.total_weight += std::pow(g, static_cast<scalar_t>(k)) *
                          std::pow(1.0 - g, static_cast<scalar_t>(m_ - k)) *
                          static_cast<scalar_t>(count_[k]);
        s.combos += count_[k];
        s.order = k;
    }
    return s;
}

// P(A) of the subset members_[0 .. k), whose whitened residuals' squared
// norms are at norm2_[k].
scalar_t Mixture::score(index_t k)
{
    const index_t B = in_.bands;
    scalar_t log_p = 0.0;
    for (index_t b = 0; b < B; ++b)
    {
        const scalar_t f = norm2_[k * B + b] / static_cast<scalar_t>(k);
        log_p += std::log(f_upper_tail(f, static_cast<scalar_t>(k), in_.degrees_of_freedom));
    }
    return fisher_tail(log_p, B);
}

// The valid subsets of size k within the first m that extend members_[0 ..
// depth): adds their P to `total`.
void Mixture::enumerate(index_t depth, index_t k, scalar_t& total)
{
    const index_t N = size();
    const index_t B = in_.bands;
    const auto& dates = in_.dates;

    std::uint32_t mask = 0;
    for (index_t i = 0; i < depth; ++i)
        mask |= std::uint32_t{1} << members_[i];

    const index_t first = depth == 0 ? 0 : members_[depth - 1] + 1;
    for (index_t j = first; j < m_; ++j)
    {
        if (depth > 0 && dates[j] - dates[members_[depth - 1]] < in_.min_days)
            continue;
        if (reach_[j] < k - depth)
            continue;   // no chain from j long enough to finish the subset

        const std::uint32_t with_j = mask | (std::uint32_t{1} << j);
        if (depth + 1 == k && stamp_[with_j] == generation_)
        {
            total += cache_[with_j];
            continue;
        }

        const scalar_t d = border(in_.h, N, members_, chol_, depth, j, row_);
        for (index_t i = 0; i < depth; ++i)
            chol_[depth * N + i] = row_[i];
        chol_[depth * N + depth] = d;
        members_[depth] = j;
        for (index_t b = 0; b < B; ++b)
        {
            scalar_t v = in_.u[b * N + j];
            for (index_t i = 0; i < depth; ++i)
                v -= chol_[depth * N + i] * z_[b * N + i];
            v /= d;
            z_[b * N + depth] = v;
            norm2_[(depth + 1) * B + b] = norm2_[depth * B + b] + v * v;
        }

        if (depth + 1 == k)
        {
            const scalar_t p = score(k);
            cache_[with_j] = p;
            stamp_[with_j] = generation_;
            total += p;
        }
        else
        {
            enumerate(depth + 1, k, total);
        }
    }
}

scalar_t Mixture::sum(index_t k)
{
    assert(k >= 1 && k <= m_);
    if (count_[k] == 0)
        return 0.0;
    scalar_t total = 0.0;
    enumerate(0, k, total);
    return total;
}

scalar_t Mixture::single(index_t j)
{
    assert(j < m_);
    const std::uint32_t mask = std::uint32_t{1} << j;
    if (stamp_[mask] != generation_)
    {
        const index_t N = size();
        const index_t B = in_.bands;
        const scalar_t var = in_.h[j * N + j];
        for (index_t b = 0; b < B; ++b)
            norm2_[1 * B + b] = in_.u[b * N + j] * in_.u[b * N + j] / var;
        cache_[mask] = score(1);
        stamp_[mask] = generation_;
    }
    return cache_[mask];
}

scalar_t Mixture::p0()
{
    const scalar_t g = in_.good;
    const scalar_t empty = std::pow(1.0 - g, static_cast<scalar_t>(m_));
    scalar_t weight = empty, weighted = empty;
    for (index_t k = 1; k <= m_; ++k)
    {
        if (count_[k] == 0)
            continue;
        const scalar_t w = std::pow(g, static_cast<scalar_t>(k)) *
                           std::pow(1.0 - g, static_cast<scalar_t>(m_ - k));
        weight += w * static_cast<scalar_t>(count_[k]);
        weighted += w * sum(k);
    }
    return weighted / weight;
}

scalar_t Mixture::top_mean()
{
    index_t order = 0;
    for (index_t k = 1; k <= m_; ++k)
        if (count_[k] > 0)
            order = k;
    return sum(order) / static_cast<scalar_t>(count_[order]);
}

Decision decide(Mixture& mixture, const Mixture::Stats& stats, std::int64_t span_days,
                const Rule& rule)
{
    if (span_days > rule.max_days_for_break)
        return Decision::NoBreak;
    if (span_days < rule.min_days_for_break || stats.m < rule.min_observations)
        return Decision::Continue;

    const index_t m = stats.m;
    const scalar_t g = mixture.good();
    const scalar_t threshold =
        rule.scale_threshold
            ? rule.threshold * static_cast<scalar_t>(stats.combos) /
                  std::ldexp(1.0, static_cast<int>(stats.order))
            : rule.threshold;

    const scalar_t empty = std::pow(1.0 - g, static_cast<scalar_t>(m));
    scalar_t sum = empty;
    scalar_t scored_weight = empty;
    for (index_t k = m; k >= 1; --k)
    {
        const std::int64_t n = mixture.count(k);
        const scalar_t t = n > 0 ? mixture.sum(k) : 0.0;
        const scalar_t w = std::pow(g, static_cast<scalar_t>(k)) *
                           std::pow(1.0 - g, static_cast<scalar_t>(m - k));
        sum += w * t;
        scored_weight += w * static_cast<scalar_t>(n);

        const scalar_t upper = (sum + (stats.total_weight - scored_weight)) / stats.total_weight;
        const scalar_t lower = sum / stats.total_weight;

        if (k == stats.order && t / static_cast<scalar_t>(n) > rule.no_change_cutoff)
            return Decision::NoBreak;
        if (upper < threshold)
            return Decision::Break;
        if (lower > threshold)
            return Decision::Continue;
    }
    return Decision::Continue;
}

} // namespace bfp
} // namespace ccd
