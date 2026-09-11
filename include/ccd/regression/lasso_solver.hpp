#pragma once

#include <array>
#include <vector>
#include <algorithm>

#include "ccd/types.hpp"
#include "ccd/constants.hpp"
#include "ccd/array_view.hpp"

namespace ccd
{


struct LassoOptions {
    
    //--------------------------------------------------------------------------
    // Optimization Parameters
    //--------------------------------------------------------------------------
    // Maximum number of coordinate descent iterations.
    index_t max_iter = 1000;

    // L1 regularization strength.
    // Larger values produce more sparse coefficients.
    //
    // THIS IS ALSO A UNIT CONSTANT, which is not obvious from its name. The
    // objective is `||r||^2/(2n) + alpha*||w||_1` and y is centered but never
    // scaled, so the soft-threshold `soft_threshold(rho, alpha * n)` is an
    // ABSOLUTE cutoff while `rho` scales with y. 1.0 is calibrated for
    // Collection-1 DN; feeding reflectance (a 10,000x smaller y) without
    // changing alpha multiplies the effective regularization by 10,000. A
    // single band's fit in isolation then zeroes all 7 coefficients on the
    // first sweep, leaving intercept = mean(y) and rmse = std(y). Reflectance
    // wants alpha = 1e-4.
    //
    // What that looks like from outside is worth knowing, because it is not
    // the clean failure the paragraph above suggests. Measured through the
    // whole Standard procedure on the pyccd reference pixel: 89 % of
    // coefficients zero rather than 100 %, and the segment count drops from 5
    // to 4, because collapsed fits change the change-detection residuals. So
    // the symptom is "mostly zeros and a different segmentation", it does not
    // raise, and since everything zeroes at once it CONVERGES FASTER -- the
    // wall clock says the run improved.
    //
    // The substitution y -> y/s, alpha -> alpha/s gives w -> w/s exactly, so
    // fitting reflectance at 1e-4 is identical to fitting DN at 1.0 and
    // dividing the coefficients by 10,000.
    scalar_t alpha = static_cast<scalar_t>(1.0);

    // Convergence tolerance.
    // Used for coefficient updates and dual gap stopping criteria.
    scalar_t tolerance = static_cast<scalar_t>(1e-4);

    // Floor on the coefficient magnitude that `tolerance` is measured
    // against: the coefficient-update stopping test is
    //
    //     max_update <= tolerance * max(max_coef, coef_floor)
    //
    // so this is an absolute scale, like `alpha`. At 1.0 (the historical
    // value) a fit whose coefficients are all well below 1 -- which is every
    // reflectance fit -- has its tolerance interpreted as absolute rather
    // than relative, and stops early. Reflectance wants 1e-4, matching alpha.
    //
    // This is NOT a rounding-level concern, which is what it looks like.
    // Measured on the pyccd reference pixel: with the reflectance preset
    // otherwise correct and only coef_floor left at 1.0, the coefficients move
    // by up to 4.0e-2 relative on blue, 2.8e-2 on swir1 and 1.6e+0 on swir2,
    // while the nonzero counts stay identical -- so it degrades the fit
    // without producing any of the signals that would make you look. Setting
    // it to 1e-4 brings the whole fit back to 1e-13 of the DN answer.
    scalar_t coef_floor = static_cast<scalar_t>(1.0);

    bool fit_intercept = true;

    bool warm_start = false;

    //--------------------------------------------------------------------------
    // Evaluation Parameters
    //--------------------------------------------------------------------------
    // Apply degrees-of-freedom correction when calculating RMSE.
    // Matches the unbiased RMSE calculation:
    //     sqrt(RSS / (n - p))
    // instead of:
    //     sqrt(RSS / n)
    bool unbiased_rmse = true;
};

//==============================================================================
// CCD Lasso Workspace
//
// Reused for every fit.
// No coefficient allocation.
//==============================================================================
struct LassoWorkspace
{   
    index_t n_samples = 0;

    std::array<scalar_t, CCD_MAX_COEFS> weights{}; // fit
    std::vector<scalar_t> y_cent; // fit
    std::vector<scalar_t> y_resi; // fit

    std::vector<scalar_t> X_store;   // prediction
    std::vector<scalar_t> X_center;  // prediction

    std::array<scalar_t, CCD_MAX_COEFS> X_mean{}; // prediction
    std::array<scalar_t, CCD_MAX_COEFS> column_norm2{}; // prediction

    std::vector<scalar_t> predictions; // prediction & score
    std::vector<scalar_t> residuals; // score

    explicit LassoWorkspace(index_t max_samples) // build outside loop
    {
        y_cent.resize(max_samples);
        y_resi.resize(max_samples);

        X_store.resize(max_samples);
        X_center.resize(max_samples);

        predictions.resize(max_samples);
        residuals.resize(max_samples);
    }

    void resize(index_t samples) // call inside loop
    {
        y_cent.resize(samples);
        y_resi.resize(samples);

        predictions.resize(samples);
        residuals.resize(samples);
    }

    void reset() // call inside loop
    {
        weights.fill(scalar_t(0));
    }

    void build_basis(
        ArrayView<const std::int64_t, 1> dates,
        index_t num_coefficients
    );

    ArrayView<const scalar_t, 2> X() const noexcept
    {
        return ArrayView<const scalar_t, 2>::contiguous(
            X_store.data(),
            {CCD_MAX_COEFS, n_samples}
        );
    }
};

class LassoModel {
public:

    LassoModel() = default;

    LassoModel(
        index_t iter,
        scalar_t bias,
        const std::array<scalar_t, CCD_MAX_COEFS>& weights
    )
        : iter_(iter),
        bias_(bias),
        weights_(weights)
    {}
    //--------------------------------------------------------------------------
    // Prediction
    //--------------------------------------------------------------------------
    // Computes:
    //      y_hat = X * weights + bias
    // Stores output into &preds
    void predict(
        ArrayView<const scalar_t, 2> X,
        std::vector<scalar_t>& preds
    ) const;

    index_t iterations() const noexcept
    {
        return iter_;
    }

    scalar_t intercept() const noexcept
    {
        return bias_;
    }

    const std::array<scalar_t, CCD_MAX_COEFS>& coefficients() const noexcept
    {
        return weights_;
    }

private:
    index_t  iter_  = 0;
    scalar_t bias_  = 0.0;

    std::array<scalar_t, CCD_MAX_COEFS> weights_{};
};

//==============================================================================
// LassoSolver
//
// Coordinate descent Lasso regression solver.
//
// Solves:
//
//      minimize:
//          (1 / 2n) ||y - Xw||^2 + lambda ||w||_1
//
// using coordinate descent.
//==============================================================================

class LassoSolver {
public:
    //--------------------------------------------------------------------------
    // Construction
    //--------------------------------------------------------------------------
    // LassoWorkspace& workspace,
    explicit LassoSolver(
        const LassoOptions& options
    ) noexcept;

    //--------------------------------------------------------------------------
    // Model fitting
    //--------------------------------------------------------------------------
    // Fits coefficients:
    //      y = X * weights + bias
    // X:
    //      shape = (samples, features)
    // y:
    //      shape = (samples)
    LassoModel fit(
        LassoWorkspace& workspace,
        ArrayView<const scalar_t, 1> y
    );

private:
    static scalar_t soft_threshold(
        scalar_t rho,
        scalar_t lambda
    ) noexcept;

private:
    LassoOptions options_;
};

struct LassoScore {
    scalar_t rmse;
    scalar_t magn;
    std::vector<scalar_t> residuals;
};

struct LassoResult {
    LassoModel model;
    LassoScore score;
};

LassoScore score(
    ArrayView<const scalar_t, 1> y,
    const std::vector<scalar_t>& predictions,
    std::vector<scalar_t>& residuals,
    index_t num_coefficients,
    bool unbiased_rmse = true
);


} // namespace ccd