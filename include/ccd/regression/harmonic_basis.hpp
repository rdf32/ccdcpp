#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "ccd/core/types.hpp"
#include "ccd/core/constants.hpp"
#include "ccd/core/array_view.hpp"

namespace ccd
{

//==============================================================================
//
// HarmonicBasis
//
// The design matrix of CCDC's harmonic model over a set of dates, shared by
// every regression on that model: the Lasso fits read it through
// LassoWorkspace, the peek predictions read it directly. Its harmonic
// columns are harmonic_terms() of each date, the one place they are
// computed, which any other fit on the model (a least-squares one, a sum of
// rows) uses for its rows too.
//
// A model with q coefficients (q = 4, 6 or 8) is
//
//     y(t) = c0 + c1 t
//          + c2 cos(wt)  + c3 sin(wt)
//          + c4 cos(2wt) + c5 sin(2wt)
//          + c6 cos(3wt) + c7 sin(3wt),     w = 2 pi / AVG_DAYS_YR,
//
// t the ordinal day, with c_q ... c7 = 0. The intercept c0 is not a column:
// the regressions carry it apart (LassoModel::intercept()). Column j holds
// the term of coefficient j + 1:
//
//     column   0   1         2         3          4          5          6
//     term     t   cos(wt)   sin(wt)   cos(2wt)   sin(2wt)   cos(3wt)   sin(3wt)
//
// build() with num_coefficients q fills whole harmonics and sets the rest to
// 0: columns 0-2 always, 3-4 when q >= 6, 5-6 when q >= 8, so the first
// q - 1 columns for q = 4, 6, 8. Prepending a column of ones gives the order [1, t, cos1, sin1,
// cos2, sin2, cos3, sin3], whose leading q x q Gram block is the
// q-coefficient model's.
//
// There are always CCD_MAX_COEFS (7) columns, and a model predicts over all
// of them, so a model is evaluated correctly on a basis built with at least
// its own q. The basis knows nothing of which q a procedure uses: those
// counts are the callers' (the Standard procedure's peek bases are
// the model's coefficient_max wide, its initialization fits coefficient_min).
//
// X(j, i) is column j at observation i; each column is contiguous. The
// storage keeps its capacity across build().
//
//==============================================================================

// The harmonic terms of a num_coefficients-coefficient model at ordinal day
// `day`, the basis' columns 1 to 6: cos(wt), sin(wt), cos(2wt), sin(2wt),
// cos(3wt), sin(3wt) with w = 2 pi / AVG_DAYS_YR, the harmonics above the
// model's 0, as the basis holds them bit for bit. The time column is the
// caller's (the basis' is the day itself).
struct HarmonicTerms
{
    scalar_t cos1, sin1, cos2 = 0, sin2 = 0, cos3 = 0, sin3 = 0;
};

inline HarmonicTerms harmonic_terms(std::int64_t day, index_t num_coefficients = 8)
{
    const scalar_t omega = 2.0 * PI / AVG_DAYS_YR;
    const scalar_t wt = omega * static_cast<scalar_t>(day);
    HarmonicTerms h{std::cos(wt), std::sin(wt)};
    if (num_coefficients >= 6)
    {
        const scalar_t w2t = static_cast<scalar_t>(2.0) * wt;
        h.cos2 = std::cos(w2t);
        h.sin2 = std::sin(w2t);
    }
    if (num_coefficients >= 8)
    {
        const scalar_t w3t = static_cast<scalar_t>(3.0) * wt;
        h.cos3 = std::cos(w3t);
        h.sin3 = std::sin(w3t);
    }
    return h;
}

class HarmonicBasis
{
public:

    // The basis of a num_coefficients-coefficient model over `dates`.
    void build(
        ArrayView<const std::int64_t, 1> dates,
        index_t num_coefficients
    );

    index_t samples() const noexcept
    {
        return samples_;
    }

    // (CCD_MAX_COEFS, samples()).
    ArrayView<const scalar_t, 2> X() const noexcept
    {
        return ArrayView<const scalar_t, 2>::contiguous(
            columns_.data(),
            {CCD_MAX_COEFS, samples_}
        );
    }

private:

    index_t samples_ = 0;
    std::vector<scalar_t> columns_;
};

} // namespace ccd
