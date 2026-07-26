#include <fstream>
#include <sstream>

#include <gtest/gtest.h>

#include "ccd/types.hpp"
#include "ccd/array_view.hpp"
#include "ccd/regression/lasso_solver.hpp"


using namespace ccd;


struct LassoReference
{
    index_t band;
    scalar_t intercept;
    std::array<scalar_t, 4> coefficients;
    index_t iterations;
    index_t nonzero;
    scalar_t rmse;
    std::vector<scalar_t> residuals;
};

std::vector<LassoReference> load_reference(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open: " + filename);
    }


    std::string line;
    std::getline(file, line); // skip header

    std::vector<LassoReference> refs;

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string field;

        LassoReference ref;

        std::getline(ss, field, ',');
        ref.band = std::stoi(field);

        std::getline(ss, field, ',');
        ref.intercept = std::stod(field);

        for (int i = 0; i < 4; ++i)
        {
            std::getline(ss, field, ',');
            ref.coefficients[i] = std::stod(field);
        }

        std::getline(ss, field, ',');
        ref.iterations = std::stoi(field);

        std::getline(ss, field, ',');
        ref.nonzero = std::stoi(field);

        std::getline(ss, field, ',');
        ref.rmse = std::stod(field);

        while (std::getline(ss, field, ','))
            ref.residuals.push_back(std::stod(field));

        refs.push_back(ref);
    }

    return refs;
}


TEST(LassoSolver, MultiBandRegression)
{   
    constexpr index_t n_samples = 12;
    constexpr index_t n_bands   = 7;

    auto reference = load_reference("lasso_reference.csv");
    ASSERT_EQ(reference.size(), n_bands);


    //------------------------------------------------------------
    // Observation dates
    //------------------------------------------------------------
    const std::vector<std::int64_t> dates{
        724387, 724419, 724451, 724483,
        724547, 724739, 724867, 724915,
        724931, 724947, 725075, 725091
    };

    //------------------------------------------------------------
    // Design matrix
    //------------------------------------------------------------
    auto X_storage = lasso_basis(
        ArrayView<const std::int64_t,1>::contiguous(
            dates.data(),
            {n_samples}
        ),
        4
    );

    auto X = ArrayView<const scalar_t,2>::contiguous(
        X_storage.data(),
        {n_samples, 7}
    );

    //------------------------------------------------------------
    // Band observations
    //------------------------------------------------------------
    const std::array<std::array<scalar_t,12>,7> y = {{
        {{ 432,  447,  289,  310,  493,  757,  508,  429,  577,  633, 1323,  637 }},
        {{ 514,  602,  484,  549,  731,  954,  793,  652,  803,  968, 1514,  840 }},
        {{ 608,  595,  406,  427,  850, 1131,  853,  745, 1034, 1305, 1662, 1052 }},
        {{ 937, 1891, 2526, 3103, 2355, 1716, 2389, 2168, 2217, 2322, 2240, 1602 }},
        {{1073, 1585, 1547, 1777, 2817, 2838, 2779, 2622, 2974, 3195, 1786, 2740 }},
        {{ 683, 1094,  813,  855, 1752, 2237, 1708, 1595, 1770, 2134, 1157, 1994 }},
        {{1935, 2325, 2665, 2705, 1105,  765, 2415, 1345,  865,  665,  -55,  315 }}
    }};

    //------------------------------------------------------------
    // Solver options
    //------------------------------------------------------------
    LassoOptions options;
    options.alpha = 1.0;
    options.max_iter = 1000;
    options.tolerance = 1e-4;

    LassoSolver solver(options);

    //------------------------------------------------------------
    // Fit each band
    //------------------------------------------------------------
    for(index_t band = 0; band < n_bands; ++band)
    {
        auto y_band = ArrayView<const scalar_t,1>::contiguous(
            y[band].data(),
            {n_samples}
        );

        auto model = solver.fit(X, y_band);

        auto predictions = model.predict(X);

        auto metrics = score(
            y_band,
            predictions,
            4,
            true
        );

        std::cout << "Band " << band << "\n";

        std::cout << "  Intercept : "
                  << model.intercept()
                  << "\n";

        std::cout << "  Coefficients: ";

        std::size_t nonzero = 0;

        for(auto w : model.coefficients())
        {
            std::cout << w << " ";

            if(std::abs(w) > 1e-12)
                ++nonzero;
        }

        std::cout << "\n";

        std::cout << "  Non-zero coefficients: "
                  << nonzero
                  << "/"
                  << model.coefficients().size()
                  << "\n";

        std::cout << "  Iterations: "
                  << model.iterations()
                  << "\n";

        std::cout << "  Residuals : ";

        for(auto r : metrics.residuals)
            std::cout << r << " ";

        std::cout << "\n";

        std::cout << "  RMSE : "
                  << metrics.rmse
                  << "\n";

        std::cout << "--------------------------------------------------\n";

        const auto& ref = reference[band];

        EXPECT_NEAR(model.intercept(), ref.intercept, 1e-8);
        ASSERT_EQ(model.coefficients().size(), ref.coefficients.size() + ref.nonzero);

        for (std::size_t i = 0; i < ref.coefficients.size(); ++i)
        {
            EXPECT_NEAR(
                model.coefficients()[i],
                ref.coefficients[i],
                1e-8
            );
        }

        EXPECT_EQ(model.iterations(), ref.iterations);
        EXPECT_NEAR(metrics.rmse, ref.rmse, 1e-8);
        ASSERT_EQ(metrics.residuals.size(), ref.residuals.size());

        for (std::size_t i = 0; i < ref.residuals.size(); ++i)
        {
            EXPECT_NEAR(
                metrics.residuals[i],
                ref.residuals[i],
                1e-8
            );
        }

    }

}


