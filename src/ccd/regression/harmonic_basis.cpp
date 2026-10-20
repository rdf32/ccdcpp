#include "ccd/regression/harmonic_basis.hpp"


namespace ccd
{

void HarmonicBasis::build(
    ArrayView<const std::int64_t, 1> dates,
    index_t num_coefficients
) {
    const index_t n = dates.size();
    samples_ = n;
    columns_.resize(CCD_MAX_COEFS * n);

    for (index_t i = 0; i < n; ++i)
    {
        const HarmonicTerms h = harmonic_terms(dates(i), num_coefficients);

        columns_[0 * n + i] = static_cast<scalar_t>(dates(i));
        columns_[1 * n + i] = h.cos1;
        columns_[2 * n + i] = h.sin1;
        columns_[3 * n + i] = h.cos2;
        columns_[4 * n + i] = h.sin2;
        columns_[5 * n + i] = h.cos3;
        columns_[6 * n + i] = h.sin3;
    }
}

} // namespace ccd
