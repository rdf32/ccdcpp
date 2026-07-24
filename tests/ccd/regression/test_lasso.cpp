
#include <gtest/gtest.h>

#include "ccd/regression/lasso_solver.hpp"


using namespace ccd;


TEST(LassoSolver, MultiBandRegression)
{

    LassoOptions options;

    options.alpha = 1.0;
    options.max_iter = 1000;
    options.tolerance = 1e-4;


    LassoSolver solver(options);

    // // load X
    // std::vector<scalar_t> Xdata = {
    //     ...
    // };

    // auto X =
    //     ArrayView<const scalar_t,2>::contiguous(
    //         Xdata.data(),
    //         {100, 7}
    //     );

    // std::vector<std::vector<scalar_t>> bands;

    // for(int band=0; band<7; band++)
    // {

    //     auto y =
    //         ArrayView<const scalar_t,1>::contiguous(
    //             bands[band].data(),
    //             {100}
    //         );

    //     // same solver
    //     auto model =
    //         solver.fit(
    //             X,
    //             y
    //         );

    //     auto preds =
    //         model.predict(X);

    //     auto score =
    //         ccd::score(
    //             y,
    //             preds,
    //             8
    //         );

    //     std::cout
    //         << "Band "
    //         << band
    //         << "\n";

    //     std::cout
    //         << "Iterations "
    //         << model.iterations()
    //         << "\n";

    //     std::cout
    //         << "Intercept "
    //         << model.intercept()
    //         << "\n";

    //     for(auto c:
    //         model.coefficients())
    //     {
    //         std::cout
    //             << c
    //             << " ";
    //     }

    //     std::cout<<"\n";

    //     EXPECT_EQ(
    //         model.coefficients().size(),
    //         7
    //     );
    // }

}


