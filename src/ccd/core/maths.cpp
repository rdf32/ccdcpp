#include "ccd/core/maths.hpp"

#include <boost/math/distributions/chi_squared.hpp>

namespace ccd
{

scalar_t chi2_quantile(scalar_t p, index_t df)
{
    boost::math::chi_squared dist(static_cast<scalar_t>(df));

    return boost::math::quantile(dist, p);
}

} // namespace ccd
