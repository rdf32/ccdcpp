#include "ccd/ccd.hpp"

#include <iostream>

#include "ccd/regression/lasso_solver.hpp"

#include "ccd/harmonic/quality.hpp"

#include "ccd/procedure/standard.hpp"
#include "ccd/procedure/permanent_snow.hpp"
#include "ccd/procedure/insufficient_clear.hpp"


namespace ccd 
{

// single pixel timeseries detection
FitResult detect(
    ArrayView<const std::int64_t, 1>& dates, // shape -> (timesteps)
    ArrayView<scalar_t, 2>& spectral,  // shape -> (bands, timesteps)
    ArrayView<const std::uint8_t, 1>& qas,   // shape -> (timesteps)
    HarmonicOptions hoptions,
    LassoOptions loptions
) {

    Quality quality;
    Quality::Statistics quality_all = quality.compute(
        qas,
        dates,
        hoptions
    );
    Quality::Statistics quality_stat = quality.compute_until(
        qas,
        dates,
        hoptions
    );

    FitProcedureType fit_type;
    if (!quality_stat.enough_clear(hoptions.CLEAR_PCT_THRESHOLD)) {
        if (quality_stat.enough_snow(hoptions.SNOW_PCT_THRESHOLD)) {
            fit_type = FitProcedureType::PermanentSnow;
        } else {
            fit_type = FitProcedureType::InsufficientClear;
        }
    } else {
        fit_type = FitProcedureType::Standard;
        // convert thermal from kelvin to celsius for standard_procedure
        const index_t T = dates.size();
        const index_t thermal_idx = hoptions.THERMAL_IDX;
        for (index_t t = 0; t < T; ++t) {
            spectral(thermal_idx, t) = 
                spectral(thermal_idx, t) * 10.0 - 27315.0;
        }
    }

    HarmonicWorkspace hworkspace(
        dates,
        spectral,
        qas,
        hoptions
    );

    LassoSolver solver(
        loptions
    );

    FitResult final_result;
    switch(fit_type)
    {
        case FitProcedureType::Standard:
        {
            std::cout << "standard procedure: " << std::endl;
            StandardProcedure fit_procedure;
            final_result = fit_procedure.run(
                hworkspace,
                solver
            );
            break;
        }

        case FitProcedureType::InsufficientClear:
        {   
            std::cout << "insufficient clear: " << std::endl;
            InsufficientClear fit_procedure;
            final_result = fit_procedure.run(
                hworkspace,
                solver
            );
            break;
        }

        case FitProcedureType::PermanentSnow:
        {
            std::cout << "permanent snow: " << std::endl;
            PermanentSnow fit_procedure;
            final_result = fit_procedure.run(
                hworkspace,
                solver
            );
            break;
        }
  
    }

    print_change_models(
        final_result.models
    );

    std::cout << "Processing Mask: " << std::endl;
    for (std::size_t obs = 0; obs < final_result.mask.size(); ++obs)
    {
        std::cout << static_cast<int>(final_result.mask[obs]) << ", ";
    }
    std::cout << "\n";
    std::cout << "number of change models: " << final_result.models.size() << std::endl;

    return final_result;
}


} // namespace ccd