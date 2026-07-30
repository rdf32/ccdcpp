
#pragma once

#include <iostream>
#include <chrono>

#include "ccd/types.hpp"
#include "ccd/array_view.hpp"
#include "ccd/ccd.hpp"

#include "ccd/procedure/fit.hpp"

namespace ccd
{

enum class FitProcedureType
{
    Standard,
    InsufficientClear,
    PermanentSnow
};

using Clock = std::chrono::high_resolution_clock;

inline auto elapsed_ms(
    Clock::time_point start,
    Clock::time_point end
)
{
    return std::chrono::duration<double, std::milli>(
        end - start
    ).count();
}

inline void print_change_model(const ChangeModel& model)
{
    std::cout << "\n========================================\n";
    std::cout << "ChangeModel\n";
    std::cout << "========================================\n";

    std::cout << "Start Day          : " << model.start_day << '\n';
    std::cout << "End Day            : " << model.end_day << '\n';
    std::cout << "Break Day          : " << model.break_day << '\n';
    std::cout << "Observation Count  : " << model.observation_count << '\n';
    std::cout << "Change Probability : " << model.change_probability << '\n';
    std::cout << "Curve QA           : " << static_cast<int>(model.curve_qa) << '\n';

    std::cout << "\nSpectral Models\n";
    std::cout << "----------------------------------------\n";

    for (std::size_t band = 0; band < model.bands.size(); ++band)
    {
        const auto& r = model.bands[band];
        const auto& m = r.model;
        const auto& s = r.score;
        const auto& coef = m.coefficients();
        // const auto& resi = s.residuals;

        std::cout << "Band " << band << '\n';
        std::cout << "  Iterations : " << m.iterations() << '\n';
        std::cout << "  Intercept  : " << m.intercept() << '\n';
        std::cout << "  RMSE       : " << s.rmse << '\n';
        std::cout << "  Magnitude  : " << s.magn << '\n';

        std::cout << "  Coefficients (" << coef.size() << "): ";

        for (std::size_t i = 0; i < coef.size(); ++i)
        {
            std::cout << coef[i];

            if (i + 1 != coef.size())
                std::cout << ", ";
        }
        std::cout << "\n";

        // std::cout << "  Residuals (" << resi.size() << "): ";

        // for (std::size_t i = 0; i < resi.size(); ++i)
        // {
        //     std::cout << resi[i];

        //     if (i + 1 != resi.size())
        //         std::cout << ", ";
        // }
        // std::cout << "\n";

        std::cout << "\n\n";
    }
    std::cout << "========================================\n";
}

inline void print_change_models(
    const std::vector<ccd::ChangeModel>& models
)
{
    for(std::size_t i = 0;
        i < models.size();
        ++i)
    {
        std::cout << "\n######## Change Model "
                  << i + 1
                  << " ########\n";

        print_change_model(
            models[i]
        );
    }
}

FitResult detect(
    ArrayView<const std::int64_t, 1>& dates, // shape -> (timesteps)
    ArrayView<scalar_t, 2>& spectral,  // shape -> (bands, timesteps)
    ArrayView<const std::uint8_t, 1>& qas,   // shape -> (timesteps)
    HarmonicOptions hoptions,
    LassoOptions loptions
);

} // namespace ccd