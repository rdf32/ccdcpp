#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <chrono>

#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "ccd/ccd.hpp"
#include "ccd/types.hpp"
#include "ccd/array_view.hpp"

#include "ccd/harmonic/harmonic.hpp"

#include "ccd/harmonic/filter.hpp"
#include "ccd/procedure/fit.hpp"
#include "ccd/procedure/standard.hpp"


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
        const ccd::BandFit& fit = model.bands[band];
        const auto& coef = fit.coefficients;

        std::cout << "Band " << band << '\n';
        std::cout << "  Iterations : " << fit.iterations << '\n';
        std::cout << "  Intercept  : " << fit.intercept << '\n';
        std::cout << "  RMSE       : " << fit.rmse << '\n';
        std::cout << "  Magnitude  : " << fit.magnitude << '\n';

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

        ::print_change_model(
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

std::vector<ExpectedModel> read_reference(
    const std::string& filename
)
{
    std::ifstream file(std::string(CCD_TEST_DATA_DIR "/") + filename);

    if (!file)
    {
        throw std::runtime_error(
            "Unable to open standard_reference.csv"
        );
    }

    std::string line;

    // Skip header
    std::getline(file, line);

    std::vector<ExpectedModel> models;

    while (std::getline(file, line))
    {
        if (line.empty())
            continue;

        std::stringstream ss(line);
        std::string value;

        auto next = [&]()
        {
            std::getline(ss, value, ',');
            return value;
        };

        ExpectedModel result{};

        result.start_day          = std::stoi(next());
        result.end_day            = std::stoi(next());
        result.break_day          = std::stoi(next());
        result.observation_count  = std::stoi(next());
        result.change_probability = std::stod(next());
        result.curve_qa           = std::stoi(next());

        for (size_t b = 0; b < 7; ++b)
        {
            auto& band = result.bands[b];

            band.rmse      = std::stod(next());
            band.intercept = std::stod(next());
            band.magnitude = std::stod(next());

            band.coefficients.clear();

            for (int i = 0; i < 7; ++i)
            {
                band.coefficients.push_back(
                    std::stod(next())
                );
            }
        }

        models.push_back(std::move(result));
    }

    return models;
}


std::vector<int> read_mask_reference(
    const std::string& filename
)
{
    std::ifstream file(std::string(CCD_TEST_DATA_DIR "/") + filename);

    if(!file)
        throw std::runtime_error(
            "Unable to open standard_mask_reference.csv"
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


//------------------------------------------------------------------------------
// The pyccd reference pixel against pyccd 2021.7.19's output (1e-6): every
// segment's days, count, change probability and curve_qa, and the mask.
// Segments 0 and 1 grow through a window of exactly RETRAIN_MIN_OBS (24)
// observations, which ccdcpp refits and pyccd does not (it refits while
// the window holds fewer), so their fits are ccdcpp's own and differ from
// pyccd's; the other three are fitted as pyccd fits them.
//------------------------------------------------------------------------------
TEST(CCD, DetectStandard)
{
    auto data = read_data("test_3657_3610_observations.csv");

    auto dates = ccd::ArrayView<const std::int64_t, 1>::contiguous(
        data.dates.data(),
        {data.dates.size()}
    );

    auto spectral = ccd::ArrayView<ccd::scalar_t, 2>::contiguous(
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

    const ccd::index_t T = dates.size();
    const ccd::index_t thermal_idx = hoptions.THERMAL_IDX;
    for (ccd::index_t t = 0; t < T; ++t) {
        spectral(thermal_idx, t) = 
            spectral(thermal_idx, t) * 10.0 - 27315.0;
    }

    ccd::LassoOptions loptions;
    ccd::LassoSolver solver(loptions);
    ccd::LassoWorkspace lworkspace(dates.size());

    ccd::standard::Procedure fit_procedure;
    ccd::FitResult result = fit_procedure.run(
        hworkspace,
        ccd::filter::standard(hworkspace),
        lworkspace,
        solver
    );

    ::print_change_models(result.models);

    auto expected_models =
        read_reference(
            "standard_reference.csv"
        );

    auto expected_mask =
        read_mask_reference(
            "standard_mask_reference.csv"
        );

    ASSERT_EQ(result.models.size(), expected_models.size());
    ASSERT_EQ(expected_models.size(), 5);

    for (size_t m = 0; m < result.models.size(); ++m) {
        const auto& actual   = result.models[m];
        const auto& expected = expected_models[m];

        EXPECT_EQ(actual.start_day, expected.start_day);
        EXPECT_EQ(actual.end_day, expected.end_day);
        EXPECT_EQ(actual.break_day, expected.break_day);
        EXPECT_EQ(actual.observation_count, expected.observation_count);

        EXPECT_NEAR(
            actual.change_probability,
            expected.change_probability,
            1e-8
        );
        EXPECT_EQ(
            static_cast<int>(actual.curve_qa),
            expected.curve_qa
        );

        if (m <= 1)
        {
            double furthest = 0.0;
            for (size_t b = 0; b < 7; b++)
                furthest = std::max(furthest,
                    std::abs(actual.bands[b].rmse - expected.bands[b].rmse));
            EXPECT_GT(furthest, 1e-3) << "segment " << m << " should be refitted at 24";
            continue;
        }

        for(size_t b = 0; b < 7; b++)
        {
            const auto& actual_band = actual.bands[b];
            const auto& expected_band = expected.bands[b];

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

            for(size_t i = 0; i < coef.size(); i++)
            {
                EXPECT_NEAR(
                    coef[i],
                    expected_band.coefficients[i],
                    1e-6
                );
            }
        }
    }
    

    ASSERT_EQ(result.mask.size(), expected_mask.size());

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


//------------------------------------------------------------------------------
// Atlanta tile 024013, chunk 00030008, pixel (481, 393): a Standard pixel
// whose stat window has an even number of gaps with middle values 8 and 9.
// pyccd's np.median gives 8.5 and a peek window of round(96 / 8.501) = 11;
// a median that truncates to 8 gives round(96 / 8.001) = 12, the lower change
// threshold that goes with it, and different outliers in Extend
// (727551 kept, 732104 dropped), so the segment ended one observation early.
//
// The input is Collection 1 DN, fill rows dropped, the reflectance bound
// applied in the QA (code 4). The reference is pyccd 2021.7.19 on it, with
// the RLM convergence test as |x0 - x| > tol, a stable date sort and a stable
// argsort in find_closest_doy. The segment reaches the series end: ccdcpp
// ends it before the observation its last full peek starts at, its break
// day, so its end day and observation count are one observation short of
// pyccd's (738751, 960; pyccd grows the window once more before its
// series-end check). Its fits, and the mask, are pyccd's.
//------------------------------------------------------------------------------
TEST(CCD, DetectStandardHalfDayMedianGap)
{
    auto data = read_data("test_atlanta_024013_00030008_481_393.csv");
    const std::size_t T = data.dates.size();
    ASSERT_EQ(T, 1760u);

    auto dates = ccd::ArrayView<const std::int64_t, 1>::contiguous(data.dates.data(), {T});
    auto spectral = ccd::ArrayView<const ccd::scalar_t, 2>::contiguous(data.spectra_storage.data(), {7, T});
    auto qas = ccd::ArrayView<const std::uint8_t, 1>::contiguous(data.qas.data(), {T});

    const ccd::HarmonicOptions hoptions;
    const ccd::LassoOptions loptions;

    const auto result = ccd::run_pixel(
        ccd::RunSpec{hoptions, loptions},
        dates,
        spectral,
        qas).result;

    ASSERT_EQ(result.models.size(), 1u);
    const auto& m = result.models[0];
    EXPECT_EQ(m.start_day, 724440);
    EXPECT_EQ(m.end_day, 738727);
    EXPECT_EQ(m.break_day, 738751);
    EXPECT_EQ(m.observation_count, 959);
    EXPECT_EQ(m.change_probability, 0.0);
    EXPECT_EQ(static_cast<int>(m.curve_qa), 8);

    // pyccd's intercept and RMSE per band.
    const double expected[7][2] = {
        {13856.652256036561, 104.84273922901038},
        {17861.914274833238, 108.69803572362223},
        {21401.59535008651, 122.95137094618717},
        {1122.2419983962932, 297.2795540223367},
        {36779.271735154354, 258.6001913744581},
        {27717.356902630087, 180.5723178104279},
        {1652.8467279281745, 358.25635528737234},
    };
    ASSERT_EQ(m.bands.size(), 7u);
    for (std::size_t b = 0; b < 7; ++b)
    {
        EXPECT_NEAR(m.bands[b].intercept, expected[b][0], 1e-6) << "band " << b;
        EXPECT_NEAR(m.bands[b].rmse, expected[b][1], 1e-6) << "band " << b;
    }

    std::size_t kept = 0;
    bool kept_727551 = false, kept_732104 = false;
    for (std::size_t t = 0; t < T; ++t)
    {
        if (!result.mask[t])
            continue;
        ++kept;
        kept_727551 |= data.dates[t] == 727551;
        kept_732104 |= data.dates[t] == 732104;
    }
    EXPECT_EQ(kept, 973u);
    EXPECT_FALSE(kept_727551);
    EXPECT_TRUE(kept_732104);
}
