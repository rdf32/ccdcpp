#include <vector>
#include <array>

#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "ccd/core/types.hpp"
#include "ccd/core/array_view.hpp"

#include "ccd/input/pixel_series.hpp"

#include "ccd/results/segment.hpp"
#include "ccd/strategies/edge_segments/lasso.hpp"
#include "ccd/input/mask/filter.hpp"
#include "ccd/input/procedure.hpp"
#include "../golden/config.hpp"


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
    std::ifstream file(std::string(CCD_TEST_DATA_DIR "/") + filename);

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
    std::ifstream file(std::string(CCD_TEST_DATA_DIR "/") + filename);

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
    std::ifstream file(std::string(CCD_TEST_DATA_DIR "/") + filename);

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

    const ccd::Input input;
    ccd::PixelSeries series(
        dates,
        spectral,
        qas
    );

    ccd::LassoOptions solver_options;
    ccd::LassoSolver solver(solver_options);
    ccd::LassoWorkspace lworkspace(data.dates.size());

    ccd::FitResult result = ccd::fit_whole_series(
        ccd::PixelProcedure::PermanentSnow,
        series,
        golden::chain_mask(series, input, "snow", golden::ccdc()),
        static_cast<ccd::index_t>(
            golden::parameter<std::int64_t>(golden::resolved().stages, "edge_segments", "min_observations")),
        static_cast<ccd::index_t>(
            golden::parameter<std::int64_t>(golden::resolved().stages, "edge_segments", "coefficients")),
        lworkspace,
        solver
    );


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
            actual_band.rmse,
            expected_band.rmse,
            1e-6
        );


        EXPECT_NEAR(
            actual_band.intercept,
            expected_band.intercept,
            1e-6
        );


        EXPECT_NEAR(
            actual_band.magnitude,
            expected_band.magnitude,
            1e-6
        );


        const auto& coef =
            actual_band.coefficients;


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
