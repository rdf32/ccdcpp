#include "ccd/strategies/model/lasso.hpp"

#include "ccd/procedure/kernels.hpp"
#include "ccd/regression/harmonic_basis.hpp"
#include "ccd/strategies/model.hpp"

namespace ccd
{
namespace strategies
{

namespace
{

// One thread's Lasso fits, with the workspace their fits build the window's
// basis in and the basis of the peek they predict.
class LassoFits final : public Model
{
public:

    explicit LassoFits(const LassoOptions& options) : solver_(options) {}

    void fit(
        ArrayView<const std::int64_t, 1> dates,
        ArrayView<const scalar_t, 2> spectral,
        index_t num_coef,
        index_t max_coef,
        const BandList& bands,
        KeepResiduals keep
    ) override
    {
        fits_.resize(spectral.extent(0));
        max_coef_ = max_coef;
        fit_bands(dates, spectral, num_coef, bands, keep, workspace_, solver_, fits_);
    }

    // max_coef wide: the fits predict with zeros above their own count.
    void peek_residuals(
        ArrayView<const std::int64_t, 1> dates,
        ArrayView<const scalar_t, 2> spectral,
        const BandList& bands,
        std::vector<std::vector<scalar_t>>& residuals
    ) override
    {
        peek_basis_.build(dates, max_coef_);
        ccd::peek_residuals(peek_basis_, spectral, fits_, bands, residuals);
    }

    scalar_t rmse(index_t band) const override { return fits_[band].score.rmse; }

    const std::vector<scalar_t>& residuals(index_t band) const override
    {
        return fits_[band].score.residuals;
    }

    BandFit band_fit(index_t band) const override { return ccd::band_fit(fits_[band]); }

private:

    LassoSolver solver_;
    LassoWorkspace workspace_{0};
    HarmonicBasis peek_basis_;
    std::vector<LassoResult> fits_;
    index_t max_coef_ = 0;
};

} // namespace

const std::vector<config::Field<Lasso::Params>>& Lasso::schema()
{
    using config::Unit;
    static const std::vector<config::Field<Params>> s = {
        {"max_iter", &Params::max_iter, Unit::None, config::at_least(0),
         "coordinate descent sweeps at most"},
        {"alpha", &Params::alpha, Unit::Reflectance, config::at_least(0),
         "the L1 penalty, in the input's units (1 for Collection-1 DN, 1e-4 for reflectance)"},
        {"tolerance", &Params::tolerance, Unit::None, config::at_least(0),
         "a sweep whose largest update is at most this times max(largest coefficient, "
         "coef_floor) checks the duality gap"},
        {"coef_floor", &Params::coef_floor, Unit::Reflectance, config::at_least(0),
         "the least coefficient size the tolerance scales"},
    };
    return s;
}

LassoOptions Lasso::solver_options() const
{
    LassoOptions o;
    o.max_iter = p_.max_iter;
    o.alpha = p_.alpha;
    o.tolerance = p_.tolerance;
    o.coef_floor = p_.coef_floor;
    return o;
}

Capabilities Lasso::provides() const
{
    return Capability::PeekResiduals | Capability::ModelRmse | Capability::InWindowResiduals;
}

std::unique_ptr<Model> Lasso::make_model() const
{
    return std::make_unique<LassoFits>(solver_options());
}

} // namespace strategies
} // namespace ccd
