#include "ccd/strategies/model/lasso.hpp"

#include <cassert>
#include <cmath>
#include <utility>

#include "ccd/regression/harmonic_basis.hpp"
#include "ccd/strategies/model.hpp"

namespace ccd
{

namespace
{

// One band's fit, over the basis already in `lworkspace`.
void fit_band(
    ArrayView<const scalar_t, 1> y,
    index_t num_coefficients,
    KeepResiduals keep,
    LassoWorkspace& lworkspace,
    LassoSolver& solver,
    LassoResult& fit
)
{
    LassoModel model = solver.fit(
        lworkspace,
        y
    );

    model.predict(
        lworkspace.X(),
        lworkspace.predictions
    );

    fit.model = std::move(model);
    fit.score.rmse = rmse(
        y,
        lworkspace.predictions,
        lworkspace.residuals,
        num_coefficients,
        true
    );
    fit.score.magn = 0.0;

    if (keep == KeepResiduals::Yes)
    {
        fit.score.residuals.assign(
            lworkspace.residuals.begin(),
            lworkspace.residuals.end()
        );
    }
    else
    {
        fit.score.residuals.clear();
    }
}

// One band's peek residuals, in place: the prediction, then |y - prediction|.
void peek_band_residuals(
    const HarmonicBasis& peek_basis,
    ArrayView<const scalar_t, 1> y,
    const LassoModel& model,
    std::vector<scalar_t>& residuals
)
{
    residuals.resize(y.size());

    model.predict(
        peek_basis.X(),
        residuals
    );

    for (index_t i = 0; i < y.size(); ++i)
    {
        residuals[i] =
            std::abs(y(i) - residuals[i]);
    }
}

} // namespace

BandFit band_fit(const LassoResult& fit)
{
    BandFit out;
    out.intercept = fit.model.intercept();
    out.coefficients = fit.model.coefficients();
    out.rmse = fit.score.rmse;
    out.magnitude = fit.score.magn;
    out.iterations = fit.model.iterations();
    return out;
}

void fit_bands(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    index_t num_coefficients,
    const BandList& bands,
    KeepResiduals keep,
    LassoWorkspace& lworkspace,
    LassoSolver& solver,
    std::vector<LassoResult>& fits
)
{
    lworkspace.resize(dates.size());
    lworkspace.build_basis(dates, num_coefficients);

    for (const index_t band : bands)
    {
        fit_band(
            spectral.slice(fixed(band), all()),
            num_coefficients,
            keep,
            lworkspace,
            solver,
            fits[band]
        );
    }
}

void fit_bands(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    index_t num_coefficients,
    KeepResiduals keep,
    LassoWorkspace& lworkspace,
    LassoSolver& solver,
    std::vector<LassoResult>& fits
)
{
    fits.resize(spectral.extent(0));
    fit_bands(dates, spectral, num_coefficients, every_band(spectral.extent(0)), keep,
              lworkspace, solver, fits);
}

void peek_residuals(
    const HarmonicBasis& peek_basis,
    ArrayView<const scalar_t, 2> peek_spectral,
    const std::vector<LassoResult>& models,
    const BandList& bands,
    std::vector<std::vector<scalar_t>>& residuals
)
{
    assert(peek_basis.samples() == peek_spectral.extent(1));

    for (const index_t band : bands)
    {
        peek_band_residuals(
            peek_basis,
            peek_spectral.slice(fixed(band), all()),
            models[band].model,
            residuals[band]
        );
    }
}

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

    bool fit(
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
        return true;
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
        {"lasso.max_iter", &Params::lasso_max_iter, Unit::None, config::at_least(0),
         "coordinate descent sweeps at most", config::Source::Shared, "lasso.max_iter"},
        {"lasso.alpha", &Params::lasso_alpha, Unit::ReflectanceDelta, config::at_least(0),
         "the L1 penalty", config::Source::Shared, "lasso.alpha"},
        {"lasso.tolerance", &Params::lasso_tolerance, Unit::None, config::at_least(0),
         "a sweep whose largest update is at most this times max(largest coefficient, "
         "coef_floor) checks the duality gap", config::Source::Shared, "lasso.tolerance"},
        {"lasso.coef_floor", &Params::lasso_coef_floor, Unit::ReflectanceDelta,
         config::at_least(0), "the least coefficient size the tolerance scales",
         config::Source::Shared, "lasso.coef_floor"},
        {"coefficients.min", &Params::coefficients_min, Unit::None, config::at_least(1),
         "coefficients of a short window's fit, and of a start window's", config::Source::Shared,
         "coefficients.min"},
        {"coefficients.mid", &Params::coefficients_mid, Unit::None, config::at_least(1),
         "coefficients from mid x per observations", config::Source::Shared,
         "coefficients.mid"},
        {"coefficients.max", &Params::coefficients_max, Unit::None, config::at_least(1),
         "coefficients from max x per observations; the basis width",
         config::Source::Shared, "coefficients.max"},
        {"coefficients.per", &Params::coefficients_per, Unit::None, config::at_least(1),
         "observations per coefficient", config::Source::Shared, "coefficients.per"},
    };
    return s;
}

Lasso::Lasso(const Params& p) : p_(p)
{
    coefficients().check("model 'lasso'");
}

CoefficientRule Lasso::coefficients() const
{
    return CoefficientRule{p_.coefficients_min, p_.coefficients_mid, p_.coefficients_max,
                           p_.coefficients_per};
}

LassoOptions Lasso::solver_options() const
{
    return LassoOptions{p_.lasso_max_iter, p_.lasso_alpha, p_.lasso_tolerance, p_.lasso_coef_floor};
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
