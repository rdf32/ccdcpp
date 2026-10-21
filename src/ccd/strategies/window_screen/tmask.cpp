#include "ccd/strategies/window_screen/tmask.hpp"

#include <cmath>
#include <stdexcept>

#include "ccd/regression/robust_solver.hpp"

namespace ccd
{

ProcessingMask tmask(
    ArrayView<const std::int64_t, 1> dates_window,
    ArrayView<const scalar_t, 2> spect_window,
    const Variogram& variogram,
    const BandList& bands,
    scalar_t t_const
) {
    const index_t n =
        spect_window.extent(1);
    //----------------------------------------------------------
    // Output mask
    // true = outlier
    //----------------------------------------------------------
    ProcessingMask outliers(n);

    Eigen::MatrixXd X =
        tmask_basis(dates_window);
    
    Eigen::VectorXd y(n);
    
    RobustOptions options;
    options.max_iter = 5;
    RobustSolver solver(options);

    // Every band is fitted against the same X.
    RobustDesign design(X);

    //----------------------------------------------------------
    // Test each spectral band
    //----------------------------------------------------------
    for(const auto band : bands)
    {

        for (index_t i = 0; i < n; ++i)
        {
            y(i) = spect_window(band, i);
        }
        //------------------------------------------------------
        // Fit
        //------------------------------------------------------
        RobustModel model = solver.fit(
            design,
            y
        );

        //------------------------------------------------------
        // Residual threshold
        //------------------------------------------------------
        const scalar_t threshold =
            variogram[band] * t_const;

        auto preds = model.predict(X);

        for(index_t i = 0; i < n; ++i)
        {
            const scalar_t residual = std::abs(preds[i] - y(i));

            if(
                residual
                >
                threshold
            )
            {
                outliers.set(i,true);
            }
        }
        
    }

    return outliers;
}

namespace strategies
{

const std::vector<config::Field<Tmask::Params>>& Tmask::schema()
{
    using config::Unit;
    static const std::vector<config::Field<Params>> s = {
        {"bands", &Params::bands, Unit::Bands, config::any(),
         "the bands Tmask fits (pyccd: green and swir1)", config::Source::Own, nullptr,
         config::roles({"green", "swir1"})},
        {"t_const", &Params::t_const, Unit::None, config::any(),
         "an observation is an outlier above this many variograms"},
    };
    return s;
}

Tmask::Tmask(const Params& p) : p_(p)
{
    if (p_.bands.empty())
        throw std::invalid_argument("window_screen 'tmask' parameter 'bands' is empty");
}

ProcessingMask Tmask::screen(ArrayView<const std::int64_t, 1> dates,
                             ArrayView<const scalar_t, 2> spectral,
                             const Variogram& variogram) const
{
    return tmask(dates, spectral, variogram, p_.bands, p_.t_const);
}

} // namespace strategies
} // namespace ccd
