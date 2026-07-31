#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <chrono>

#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "ccd/types.hpp"
#include "ccd/array_view.hpp"

#include "ccd/harmonic/harmonic.hpp"

#include "ccd/procedure/fit.hpp"
#include "ccd/procedure/permanent_snow.hpp"


using Clock = std::chrono::high_resolution_clock;

inline void print_change_model(const ccd::ChangeModel& model)
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
                  << i
                  << " ########\n";

        print_change_model(
            models[i]
        );
    }
}

struct TestData
{
    std::vector<std::int64_t> dates;
    std::vector<std::uint8_t> qas;
    std::vector<ccd::scalar_t> spectra_storage;
};

TestData read_data(
    const std::string& filename
)
{
    std::ifstream file(filename);

    if(!file)
    {
        throw std::runtime_error(
            "Unable to open file"
        );
    }

    std::vector<std::int64_t> dates;
    std::vector<std::uint8_t> qas;

    std::array<
        std::vector<ccd::scalar_t>,
        7
    > bands;

    std::string line;

    while(std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string value;

        // date
        std::getline(ss,value,',');
        dates.push_back(
            std::stoll(value)
        );
        // blue-green-red-nir-swir1-swir2-thermal
        for(ccd::index_t b = 0; b < 7; ++b)
        {
            std::getline(ss,value,',');

            bands[b].push_back(
                std::stod(value)
            );
        }
        // qa
        std::getline(ss,value,',');
        qas.push_back(
            static_cast<std::uint8_t>(
                std::stoi(value)
            )
        );
    }

    TestData result;
    result.dates = dates;
    result.qas   = qas;

    const ccd::index_t T = dates.size();
    // same layout as numpy:
    // spectra.shape = (7,T)
    result.spectra_storage.resize(7*T);

    for(ccd::index_t b=0; b < 7; ++b)
    {
        for(ccd::index_t t = 0; t < T; ++t)
        {
            result.spectra_storage[
                b*T+t
            ] = bands[b][t];
        }
    }


    return result;
}

auto elapsed_ms(
    Clock::time_point start,
    Clock::time_point end
)
{
    return std::chrono::duration<double, std::milli>(
        end - start
    ).count();
}

struct ExpectedBand
{
    double rmse;
    double intercept;
    double magnitude;

    std::vector<double> coefficients;
};


struct ExpectedModel
{
    int start_day;
    int end_day;
    int break_day;

    int observation_count;

    double change_probability;

    int curve_qa;

    std::array<ExpectedBand,7> bands;
};

ExpectedModel read_snow_reference(
    const std::string& filename
)
{
    std::ifstream file(filename);

    if(!file)
        throw std::runtime_error(
            "Unable to open snow_reference.csv"
        );

    std::string line;

    // skip header
    std::getline(file,line);

    ExpectedModel result{};

    if(!std::getline(file,line))
    {
        throw std::runtime_error(
            "Empty reference file"
        );
    }

    std::stringstream ss(line);
    std::string value;

    auto next = [&]()
    {
        std::getline(ss,value,',');
        return value;
    };

    result.start_day =
        std::stoi(next());

    result.end_day =
        std::stoi(next());

    result.break_day =
        std::stoi(next());

    result.observation_count =
        std::stoi(next());

    result.change_probability =
        std::stod(next());

    result.curve_qa =
        std::stoi(next());

    constexpr std::array<const char*,7> bands =
    {
        "blue",
        "green",
        "red",
        "nir",
        "swir1",
        "swir2",
        "thermal"
    };

    for(size_t b = 0; b < 7; ++b)
    {
        auto& band = result.bands[b];

        band.rmse =
            std::stod(next());

        band.intercept =
            std::stod(next());

        band.magnitude =
            std::stod(next());

        band.coefficients.clear();

        for(int i = 0; i < 7; i++)
        {
            band.coefficients.push_back(
                std::stod(next())
            );
        }
    }

    return result;
}

std::vector<int> read_mask_reference(
    const std::string& filename
)
{
    std::ifstream file(filename);

    if(!file)
        throw std::runtime_error(
            "Unable to open snow_mask_reference.csv"
        );

    std::vector<int> mask;
    std::string line;

    // header
    std::getline(file,line);

    while(std::getline(file,line))
    {
        if(!line.empty())
        {
            mask.push_back(
                std::stoi(line)
            );
        }
    }

    return mask;
}


TEST(CCD, DetectSnow)
{
    auto data = read_data("test_3657_3610_observations.csv");

    auto dates = ccd::ArrayView<const std::int64_t, 1>::contiguous(
        data.dates.data(),
        {data.dates.size()}
    );

    auto spectral = ccd::ArrayView<const ccd::scalar_t, 2>::contiguous(
        data.spectra_storage.data(),
        {7, data.dates.size()}
    );

    auto qas = ccd::ArrayView<const std::uint8_t, 1>::contiguous(
        data.qas.data(),
        {data.dates.size()}
    );

    ccd::HarmonicOptions hoptions;
    ccd::HarmonicWorkspace hworkspace(
        dates,
        spectral,
        qas,
        hoptions
    );

    ccd::LassoOptions loptions;
    ccd::LassoSolver solver(loptions);
    ccd::LassoWorkspace lworkspace(data.dates.size());

    ccd::PermanentSnow fit_procedure;
    ccd::FitResult result = fit_procedure.run(
        hworkspace,
        lworkspace,
        solver
    );

    print_change_models(result.models);

    auto expected =
        read_snow_reference(
            "snow_reference.csv"
        );

    auto expected_mask =
        read_mask_reference(
            "snow_mask_reference.csv"
        );

    ASSERT_EQ(
        result.models.size(),
        1
    );

    const auto& actual =
        result.models[0];

    EXPECT_EQ(
        actual.start_day,
        expected.start_day
    );

    EXPECT_EQ(
        actual.end_day,
        expected.end_day
    );

    EXPECT_EQ(
        actual.break_day,
        expected.break_day
    );

    EXPECT_EQ(
        actual.observation_count,
        expected.observation_count
    );


    EXPECT_NEAR(
        actual.change_probability,
        expected.change_probability,
        1e-8
    );

    EXPECT_EQ(
        static_cast<int>(actual.curve_qa),
        expected.curve_qa
    );

    for(size_t b=0;b<7;b++)
    {
        const auto& actual_band =
            actual.bands[b];

        const auto& expected_band =
            expected.bands[b];


        EXPECT_NEAR(
            actual_band.score.rmse,
            expected_band.rmse,
            1e-6
        );


        EXPECT_NEAR(
            actual_band.model.intercept(),
            expected_band.intercept,
            1e-6
        );


        EXPECT_NEAR(
            actual_band.score.magn,
            expected_band.magnitude,
            1e-6
        );


        const auto& coef =
            actual_band.model.coefficients();


        ASSERT_EQ(
            coef.size(),
            expected_band.coefficients.size()
        );


        for(size_t i=0;i<coef.size();i++)
        {
            EXPECT_NEAR(
                coef[i],
                expected_band.coefficients[i],
                1e-6
            );
        }
    }

    ASSERT_EQ(
        result.mask.size(),
        expected_mask.size()
    );


    for(size_t i=0;i<expected_mask.size();i++)
    {
        EXPECT_EQ(
            static_cast<int>(
                result.mask.test(i)
            ),
            expected_mask[i]
        );
    }

}
