#pragma once

#include <vector>

#include "ccd/types.hpp"
#include "ccd/array_view.hpp"

namespace ccd
{

class LinearLeastSquares
{
public:

    LinearLeastSquares() = default;

    void fit(
        ArrayView<const scalar_t,2> X,
        ArrayView<const scalar_t,1> y
    );

    void fit(
        ArrayView<const scalar_t,2> X,
        ArrayView<const scalar_t,1> y,
        ArrayView<const scalar_t,1> weights
    );

    ArrayView<const scalar_t,1> coefficients() const noexcept;

    scalar_t predict(
        ArrayView<const scalar_t,1> row
    ) const noexcept;

private:

    void solve(
        ArrayView<const scalar_t,2> X,
        ArrayView<const scalar_t,1> y,
        const scalar_t* weights
    );

    std::vector<scalar_t> coef_;
};

}