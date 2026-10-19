#include "ccd/strategies/edge_fit/lasso.hpp"

#include <utility>

#include "ccd/machine/kernels.hpp"

namespace ccd
{
namespace strategies
{

namespace
{

// One thread's solver and the workspace its fits build their basis in.
class EdgeScratch final : public Scratch
{
public:
    explicit EdgeScratch(const LassoOptions& options) : solver(options) {}

    LassoSolver solver;
    LassoWorkspace workspace{0};
};

} // namespace

const std::vector<config::Field<LassoEdges::Params>>& LassoEdges::schema()
{
    using config::Unit;
    static const std::vector<config::Field<Params>> s = {
        {"coefficients", &Params::coefficients, Unit::None, config::at_least(1),
         "coefficients of every edge fit"},
        {"min_observations", &Params::min_observations, Unit::Observations, config::any(),
         "kept observations a whole-series curve needs"},
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

std::unique_ptr<Scratch> LassoEdges::make_scratch() const
{
    LassoOptions o;
    o.max_iter = p_.max_iter;
    o.alpha = p_.alpha;
    o.tolerance = p_.tolerance;
    o.coef_floor = p_.coef_floor;
    return std::make_unique<EdgeScratch>(o);
}

ChangeModel LassoEdges::catch_window(ArrayView<const std::int64_t, 1> dates,
                                     ArrayView<const scalar_t, 2> spectral, const Window& window,
                                     CurveQA curve_qa, Scratch& scratch) const
{
    EdgeScratch& s = static_cast<EdgeScratch&>(scratch);
    return catch_segment(dates, spectral, window, p_.coefficients, curve_qa, s.workspace,
                         s.solver);
}

FitResult LassoEdges::whole_series(PixelProcedure procedure, const HarmonicWorkspace& pixel,
                                   ProcessingMask mask, Scratch& scratch) const
{
    EdgeScratch& s = static_cast<EdgeScratch&>(scratch);
    return fit_whole_series(procedure, pixel, std::move(mask), p_.min_observations,
                            p_.coefficients, s.workspace, s.solver);
}

} // namespace strategies
} // namespace ccd
