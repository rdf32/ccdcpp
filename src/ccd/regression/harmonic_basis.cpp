#include "ccd/regression/harmonic_basis.hpp"

#include <cmath>

namespace ccd
{

void HarmonicBasis::build(
    ArrayView<const std::int64_t, 1> dates,
    index_t num_coefficients
) {
    const index_t n = dates.size();
    samples_ = n;
    columns_.resize(CCD_MAX_COEFS * n);

    const scalar_t omega = 2.0 * PI / AVG_DAYS_YR;

    for (index_t i = 0; i < n; ++i) {

        const scalar_t t =
            static_cast<scalar_t>(
                dates(i)
            );
        const scalar_t wt = omega * t;

        columns_[0 * n + i] = t;
        columns_[1 * n + i] = std::cos(wt);
        columns_[2 * n + i] = std::sin(wt);

        columns_[3 * n + i] = scalar_t{0};
        columns_[4 * n + i] = scalar_t{0};
        columns_[5 * n + i] = scalar_t{0};
        columns_[6 * n + i] = scalar_t{0};

        if (num_coefficients >= 6) {
            const scalar_t w2t =
                static_cast<scalar_t>(2.0) * wt;

            columns_[3 * n + i] = std::cos(w2t);
            columns_[4 * n + i] = std::sin(w2t);
        }

        if (num_coefficients >= 8) {
            const scalar_t w3t =
                static_cast<scalar_t>(3.0) * wt;

            columns_[5 * n + i] = std::cos(w3t);
            columns_[6 * n + i] = std::sin(w3t);
        }
    }
}

} // namespace ccd
