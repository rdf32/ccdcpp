//==============================================================================
//
// The shared kernels against the per-site loops they stand for, bit for bit,
// on the CSV pixels' masked series.
//
//   * HarmonicBasis::build() and LassoWorkspace::build_basis(): the basis
//     columns, and the centred columns, means and norms the solver reads.
//   * fit_bands(): build the window's basis, then per band fit, predict,
//     rmse(), residuals kept or not; on Initialise's band subset and on every
//     band, and with the whole-series fit's reset before each band.
//   * standard::peek_residuals(): an 8-coefficient basis over the peek,
//     predict, |y - prediction|; on a forward peek (Extend) and a reversed
//     one (Lookback).
//   * standard::change_magnitude(): the per-detection-band sum over copies
//     of the residuals and the variogram (the positional form).
//   * catch_segment(): the catch's fit and day rules.
//
// The procedures' outputs are held by the golden and transitions fixtures;
// these tests say which kernel moved if one ever does.
//
//==============================================================================

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ccd/constants.hpp"
#include "ccd/harmonic/filter.hpp"
#include "ccd/harmonic/harmonic.hpp"
#include "ccd/procedure/kernels.hpp"
#include "ccd/procedure/masked_series.hpp"
#include "ccd/procedure/standard.hpp"
#include "ccd/regression/harmonic_basis.hpp"
#include "ccd/regression/lasso_solver.hpp"

#include "../golden/golden.hpp"

namespace
{

using ccd::index_t;
using ccd::scalar_t;

//------------------------------------------------------------------------------
// The CSV pixels
//------------------------------------------------------------------------------

struct Pixel
{
    std::vector<std::int64_t> dates;
    std::vector<double> spectral;   // (7, T)
    std::vector<std::uint8_t> qas;
};

Pixel read_csv(const std::string& path)
{
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("Unable to open " + path);

    std::array<std::vector<double>, 7> bands;
    Pixel out;

    std::string line;
    std::getline(file, line);   // header (the golden test skips it too)
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string value;

        std::getline(ss, value, ',');
        out.dates.push_back(static_cast<std::int64_t>(std::stod(value)));

        for (std::size_t b = 0; b < 7; ++b)
        {
            std::getline(ss, value, ',');
            bands[b].push_back(std::stod(value));
        }

        std::getline(ss, value, ',');
        out.qas.push_back(static_cast<std::uint8_t>(std::stod(value)));
    }

    const std::size_t T = out.dates.size();
    out.spectral.resize(7 * T);
    for (std::size_t b = 0; b < 7; ++b)
        std::copy(bands[b].begin(), bands[b].end(), out.spectral.begin() + b * T);

    return out;
}

// A pixel's Standard-filter masked series, as the procedures see it.
struct MaskedPixel
{
    Pixel pixel;
    ccd::HarmonicOptions options;
    ccd::MaskedSeries series;

    MaskedPixel(const std::string& path, ccd::HarmonicOptions options_)
        : pixel(read_csv(path)), options(options_)
    {
        const index_t T = pixel.dates.size();
        const ccd::HarmonicWorkspace workspace(
            ccd::ArrayView<const std::int64_t, 1>::contiguous(pixel.dates.data(), {T}),
            ccd::ArrayView<const double, 2>::contiguous(pixel.spectral.data(), {7, T}),
            ccd::ArrayView<const std::uint8_t, 1>::contiguous(pixel.qas.data(), {T}),
            options
        );
        series.reset(workspace, ccd::filter::standard(workspace));
    }
};

// The pyccd reference pixel at the default options, and the Collection 2
// pixel at the golden corpus's C2 options.
struct CsvPixel
{
    std::string path;
    ccd::HarmonicOptions options;
};

const std::vector<CsvPixel>& csv_pixels()
{
    static const std::vector<CsvPixel> pixels{
        {CCD_PYCCD_PIXEL_CSV, ccd::HarmonicOptions{}},
        {CCD_C2_PIXEL_CSV, golden::c2_harmonic()},
    };
    return pixels;
}

// Windows of every length class the procedures fit: the 12-observation
// start window, Extend's 4/6/8-coefficient lengths, and a long one.
std::vector<ccd::Window> windows(index_t n)
{
    std::vector<ccd::Window> out;
    for (const index_t start : {index_t{0}, index_t{7}, index_t{40}})
        for (const index_t size : {index_t{12}, index_t{19}, index_t{30}, index_t{90}})
            if (start + size <= n)
                out.emplace_back(start, start + size);
    out.emplace_back(0, n);
    return out;
}

bool same_bits(const std::vector<scalar_t>& a, const std::vector<scalar_t>& b)
{
    return a.size() == b.size() &&
        (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(scalar_t)) == 0);
}

bool same_bits(scalar_t a, scalar_t b)
{
    return std::memcmp(&a, &b, sizeof(scalar_t)) == 0;
}

void expect_same_fit(const ccd::LassoResult& a, const ccd::LassoResult& b)
{
    EXPECT_EQ(a.model.iterations(), b.model.iterations());
    EXPECT_TRUE(same_bits(a.model.intercept(), b.model.intercept()));
    EXPECT_EQ(
        std::memcmp(a.model.coefficients().data(), b.model.coefficients().data(),
                    sizeof(scalar_t) * ccd::CCD_MAX_COEFS),
        0);
    EXPECT_TRUE(same_bits(a.score.rmse, b.score.rmse));
    EXPECT_TRUE(same_bits(a.score.magn, b.score.magn));
    EXPECT_TRUE(same_bits(a.score.residuals, b.score.residuals));
}

//------------------------------------------------------------------------------
// The per-site loops, as they were written at each site
//------------------------------------------------------------------------------

std::vector<scalar_t> basis_loop(ccd::ArrayView<const std::int64_t, 1> dates, index_t num_coefficients)
{
    const index_t n = dates.size();
    std::vector<scalar_t> X(ccd::CCD_MAX_COEFS * n);
    const scalar_t omega = 2.0 * ccd::PI / ccd::AVG_DAYS_YR;

    for (index_t i = 0; i < n; ++i)
    {
        const scalar_t t = static_cast<scalar_t>(dates(i));
        const scalar_t wt = omega * t;

        X[0 * n + i] = t;
        X[1 * n + i] = std::cos(wt);
        X[2 * n + i] = std::sin(wt);
        X[3 * n + i] = scalar_t{0};
        X[4 * n + i] = scalar_t{0};
        X[5 * n + i] = scalar_t{0};
        X[6 * n + i] = scalar_t{0};

        if (num_coefficients >= 6)
        {
            const scalar_t w2t = static_cast<scalar_t>(2.0) * wt;
            X[3 * n + i] = std::cos(w2t);
            X[4 * n + i] = std::sin(w2t);
        }
        if (num_coefficients >= 8)
        {
            const scalar_t w3t = static_cast<scalar_t>(3.0) * wt;
            X[5 * n + i] = std::cos(w3t);
            X[6 * n + i] = std::sin(w3t);
        }
    }
    return X;
}

// Initialise's and Extend's loop (residuals kept), the catch's (not kept),
// and the whole-series fit's (not kept, the workspace reset per band).
enum class Site { Initialise, Extend, Catch, WholeSeries };

void fit_loop(
    Site site,
    ccd::ArrayView<const std::int64_t, 1> dates,
    ccd::ArrayView<const scalar_t, 2> spectral,
    index_t num_coef,
    const ccd::BandList& detection_bands,
    ccd::LassoWorkspace& lworkspace,
    ccd::LassoSolver& solver,
    std::vector<ccd::LassoResult>& models
)
{
    using namespace ccd;

    lworkspace.resize(dates.size());
    lworkspace.build_basis(dates, num_coef);

    std::vector<index_t> bands;
    if (site == Site::Initialise)
        bands.assign(detection_bands.begin(), detection_bands.end());
    else
        for (index_t b = 0; b < spectral.extent(0); ++b)
            bands.push_back(b);

    for (const index_t band : bands)
    {
        if (site == Site::WholeSeries)
            lworkspace.reset();

        auto y = spectral.slice(fixed(band), all());
        LassoModel model = solver.fit(lworkspace, y);
        model.predict(lworkspace.X(), lworkspace.predictions);

        if (site == Site::Initialise || site == Site::Extend)
        {
            models[band].model = std::move(model);
            models[band].score.rmse =
                rmse(y, lworkspace.predictions, lworkspace.residuals, num_coef, true);
            models[band].score.magn = 0.0;
            models[band].score.residuals = lworkspace.residuals;
        }
        else
        {
            LassoScore metrics{
                rmse(y, lworkspace.predictions, lworkspace.residuals, num_coef, true),
                0.0,
                {}
            };
            models[band] = LassoResult{std::move(model), std::move(metrics)};
        }
    }
}

// Lookback's and Extend's peek loop: the basis in the Lasso workspace,
// predict, |y - prediction|.
void peek_loop(
    ccd::ArrayView<const std::int64_t, 1> peek_dates,
    ccd::ArrayView<const scalar_t, 2> peek_spectral,
    const std::vector<ccd::LassoResult>& models,
    const std::vector<index_t>& bands,
    ccd::LassoWorkspace& lworkspace,
    std::vector<std::vector<scalar_t>>& residuals
)
{
    using namespace ccd;

    lworkspace.resize(peek_dates.size());
    lworkspace.build_basis(peek_dates, 8);

    for (const index_t band : bands)
    {
        auto y = peek_spectral.slice(fixed(band), all());
        models[band].model.predict(lworkspace.X(), lworkspace.predictions);

        residuals[band].resize(y.size());
        for (index_t i = 0; i < y.size(); ++i)
            residuals[band][i] = std::abs(y(i) - lworkspace.predictions[i]);
    }
}

// The positional magnitude: copies of the detection bands' residuals and
// variogram, in DETECTION_BANDS order.
std::vector<scalar_t> magnitude_loop(
    const std::vector<std::vector<scalar_t>>& residuals_by_band,
    const ccd::BandList& bands,
    const ccd::Variogram& variogram,
    const std::vector<scalar_t>& rmse
)
{
    std::vector<std::vector<scalar_t>> residuals;
    std::vector<scalar_t> vario;
    for (const index_t band : bands)
    {
        residuals.push_back(residuals_by_band[band]);
        vario.push_back(variogram[band]);
    }

    const index_t count = residuals.front().size();
    std::vector<scalar_t> magnitude(count, 0.0);
    for (index_t band = 0; band < residuals.size(); ++band)
    {
        const scalar_t norm = std::max(vario[band], rmse[band]);
        if (norm <= 0.0)
            continue;
        for (index_t i = 0; i < count; ++i)
        {
            const scalar_t value = residuals[band][i] / norm;
            magnitude[i] += value * value;
        }
    }
    return magnitude;
}

ccd::LassoOptions lasso_options()
{
    return ccd::LassoOptions{};
}

} // namespace

//------------------------------------------------------------------------------
// HarmonicBasis
//------------------------------------------------------------------------------

TEST(HarmonicBasis, MatchesTheInlineBuilder)
{
    for (const auto& [path, options] : csv_pixels())
    {
        const MaskedPixel p(path, options);
        ASSERT_GE(p.series.size(), index_t{200}) << path;
        const auto dates = p.series.dates();

        ccd::HarmonicBasis basis;
        ccd::LassoWorkspace lworkspace(0);

        for (const auto& w : windows(p.series.size()))
        {
            const auto window_dates = dates.slice(ccd::range(w.start, w.stop));
            const index_t n = w.size();

            for (const index_t q : {index_t{4}, index_t{6}, index_t{8}})
            {
                SCOPED_TRACE(path + " window (" + std::to_string(w.start) + ", " +
                             std::to_string(w.stop) + ") q " + std::to_string(q));

                const std::vector<scalar_t> expected = basis_loop(window_dates, q);

                basis.build(window_dates, q);
                ASSERT_EQ(basis.samples(), n);
                const auto X = basis.X();
                ASSERT_EQ(X.extent(0), ccd::CCD_MAX_COEFS);
                std::vector<scalar_t> got(ccd::CCD_MAX_COEFS * n);
                for (index_t j = 0; j < ccd::CCD_MAX_COEFS; ++j)
                    for (index_t i = 0; i < n; ++i)
                        got[j * n + i] = X(j, i);
                EXPECT_TRUE(same_bits(got, expected));

                // The workspace's centred copy, means and norms of that basis.
                lworkspace.resize(n);
                lworkspace.build_basis(window_dates, q);
                for (index_t j = 0; j < ccd::CCD_MAX_COEFS; ++j)
                {
                    scalar_t mean = 0.0;
                    for (index_t i = 0; i < n; ++i)
                        mean += expected[j * n + i];
                    mean /= n;
                    EXPECT_TRUE(same_bits(lworkspace.X_mean[j], mean));

                    scalar_t norm2 = 0.0;
                    for (index_t i = 0; i < n; ++i)
                    {
                        const scalar_t v = expected[j * n + i] - mean;
                        EXPECT_TRUE(same_bits(lworkspace.X_center[j * n + i], v));
                        norm2 += v * v;
                    }
                    // A sum of squares: whether `s += v * v` fuses into one
                    // multiply-add is the compiler's choice at each site, so
                    // the norm is held to the rounding of that choice.
                    EXPECT_NEAR(lworkspace.column_norm2[j], norm2, 4e-16 * norm2);
                }
            }
        }
    }
}

// The column order the header documents, at one date.
TEST(HarmonicBasis, ColumnOrder)
{
    const std::vector<std::int64_t> day{730000};
    const auto dates = ccd::ArrayView<const std::int64_t, 1>::contiguous(day.data(), {1});
    const scalar_t wt = 2.0 * ccd::PI / ccd::AVG_DAYS_YR * 730000.0;

    ccd::HarmonicBasis basis;
    basis.build(dates, 8);
    const auto X = basis.X();
    EXPECT_EQ(X(0, 0), 730000.0);
    EXPECT_EQ(X(1, 0), std::cos(wt));
    EXPECT_EQ(X(2, 0), std::sin(wt));
    EXPECT_EQ(X(3, 0), std::cos(2.0 * wt));
    EXPECT_EQ(X(4, 0), std::sin(2.0 * wt));
    EXPECT_EQ(X(5, 0), std::cos(3.0 * wt));
    EXPECT_EQ(X(6, 0), std::sin(3.0 * wt));

    basis.build(dates, 4);
    for (index_t j = 3; j < ccd::CCD_MAX_COEFS; ++j)
        EXPECT_EQ(basis.X()(j, 0), 0.0);
}

//------------------------------------------------------------------------------
// fit_bands
//------------------------------------------------------------------------------

TEST(FitBands, MatchesEverySiteLoop)
{
    for (const auto& [path, options] : csv_pixels())
    {
        const MaskedPixel p(path, options);
        ASSERT_GE(p.series.size(), index_t{200}) << path;
        const auto dates = p.series.dates();
        const auto spectral = p.series.spectral();
        const index_t B = spectral.extent(0);

        ccd::LassoSolver solver(lasso_options());
        ccd::LassoWorkspace expected_ws(0), got_ws(0);

        for (const auto& w : windows(p.series.size()))
        {
            const auto wd = dates.slice(ccd::range(w.start, w.stop));
            const auto ws = spectral.slice(ccd::all(), ccd::range(w.start, w.stop));

            for (const index_t q : {index_t{4}, index_t{6}, index_t{8}})
            {
                if (w.size() <= q)
                    continue;   // rmse() needs more observations than coefficients

                for (const Site site : {Site::Initialise, Site::Extend, Site::Catch, Site::WholeSeries})
                {
                    SCOPED_TRACE(path + " window (" + std::to_string(w.start) + ", " +
                                 std::to_string(w.stop) + ") q " + std::to_string(q) +
                                 " site " + std::to_string(static_cast<int>(site)));

                    std::vector<ccd::LassoResult> expected(B), got(B);
                    fit_loop(site, wd, ws, q, p.options.DETECTION_BANDS, expected_ws, solver, expected);

                    switch (site)
                    {
                        case Site::Initialise:
                            ccd::fit_bands(wd, ws, q, p.options.DETECTION_BANDS,
                                           ccd::KeepResiduals::Yes, got_ws, solver, got);
                            break;
                        case Site::Extend:
                            ccd::fit_bands(wd, ws, q, ccd::KeepResiduals::Yes, got_ws, solver, got);
                            break;
                        case Site::Catch:
                        case Site::WholeSeries:
                            got.clear();
                            ccd::fit_bands(wd, ws, q, ccd::KeepResiduals::No, got_ws, solver, got);
                            break;
                    }

                    ASSERT_EQ(got.size(), B);
                    for (index_t b = 0; b < B; ++b)
                        expect_same_fit(got[b], expected[b]);
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
// peek_residuals and change_magnitude
//------------------------------------------------------------------------------

TEST(PeekResiduals, MatchesThePeekLoopForwardAndReversed)
{
    for (const auto& [path, options] : csv_pixels())
    {
        const MaskedPixel p(path, options);
        ASSERT_GE(p.series.size(), index_t{200}) << path;
        const auto dates = p.series.dates();
        const auto spectral = p.series.spectral();
        const index_t B = spectral.extent(0);
        const auto& detection = p.options.DETECTION_BANDS;

        ccd::LassoSolver solver(lasso_options());
        ccd::LassoWorkspace lworkspace(0);
        ccd::HarmonicBasis peek_basis;

        std::vector<index_t> all_bands;
        for (index_t b = 0; b < B; ++b)
            all_bands.push_back(b);
        const std::vector<index_t> detection_bands(detection.begin(), detection.end());

        for (const index_t q : {index_t{4}, index_t{6}, index_t{8}})
        for (const index_t peek : {index_t{6}, index_t{13}})
        for (const index_t start : {index_t{20}, index_t{60}})
        {
            const ccd::Window fit(start, start + 30);
            if (fit.stop + peek > p.series.size())
                continue;
            SCOPED_TRACE(path + " q " + std::to_string(q) + " peek " + std::to_string(peek) +
                         " start " + std::to_string(start));

            std::vector<ccd::LassoResult> models(B);
            ccd::fit_bands(dates.slice(ccd::range(fit.start, fit.stop)),
                           spectral.slice(ccd::all(), ccd::range(fit.start, fit.stop)),
                           q, ccd::KeepResiduals::No, lworkspace, solver, models);

            // Forward (Extend): the observations after the fit window.
            {
                const auto pd = dates.slice(ccd::range(fit.stop, fit.stop + peek));
                const auto ps = spectral.slice(ccd::all(), ccd::range(fit.stop, fit.stop + peek));

                std::vector<std::vector<scalar_t>> expected(B), got(B);
                peek_loop(pd, ps, models, all_bands, lworkspace, expected);

                peek_basis.build(pd, 8);
                ccd::standard::peek_residuals(peek_basis, ps, models, got);
                for (index_t b = 0; b < B; ++b)
                    EXPECT_TRUE(same_bits(got[b], expected[b]));
            }

            // Reversed (Lookback): the observations before the window,
            // nearest first, gathered into contiguous copies.
            {
                std::vector<std::int64_t> rd;
                std::vector<scalar_t> rs(B * (peek - 1));
                for (index_t j = 0; j + 1 < peek; ++j)
                    rd.push_back(dates(fit.start - 1 - j));
                for (index_t b = 0; b < B; ++b)
                    for (index_t j = 0; j + 1 < peek; ++j)
                        rs[b * (peek - 1) + j] = spectral(b, fit.start - 1 - j);
                const auto pd = ccd::ArrayView<const std::int64_t, 1>::contiguous(rd.data(), {peek - 1});
                const auto ps = ccd::ArrayView<const scalar_t, 2>::contiguous(rs.data(), {B, peek - 1});

                std::vector<std::vector<scalar_t>> expected(B), got(B);
                peek_loop(pd, ps, models, detection_bands, lworkspace, expected);

                peek_basis.build(pd, 8);
                ccd::standard::peek_residuals(peek_basis, ps, models, detection, got);
                for (index_t b = 0; b < B; ++b)
                    EXPECT_TRUE(same_bits(got[b], expected[b]));

                // The magnitude over them, against the positional sum, with
                // the models' RMSE and with a stand-in seasonal RMSE.
                const ccd::Variogram variogram{
                    ccd::adjusted_variogram(dates, spectral, p.options.VARIOGRAM_MIN_LAG_DAYS)};
                for (const scalar_t scale : {1.0, 0.37})
                {
                    std::vector<scalar_t> rmse;
                    for (const index_t band : detection)
                        rmse.push_back(models[band].score.rmse * scale);

                    std::vector<scalar_t> magnitude{-1.0};
                    ccd::standard::change_magnitude(got, detection, variogram, rmse, magnitude);
                    EXPECT_TRUE(same_bits(magnitude, magnitude_loop(got, detection, variogram, rmse)));
                }
            }
        }
    }
}

// A band whose norm is not above 0 adds nothing; no detection bands, no
// magnitudes.
TEST(ChangeMagnitude, NonPositiveNormAndNoBands)
{
    const std::vector<std::vector<scalar_t>> residuals{{1.0, 2.0}, {3.0, 4.0}};
    const ccd::Variogram variogram{{0.0, 2.0}};
    std::vector<scalar_t> magnitude;

    ccd::standard::change_magnitude(residuals, ccd::BandList{0, 1}, variogram, {0.0, 1.0}, magnitude);
    EXPECT_EQ(magnitude, (std::vector<scalar_t>{2.25, 4.0}));

    ccd::standard::change_magnitude(residuals, ccd::BandList{}, variogram, {}, magnitude);
    EXPECT_TRUE(magnitude.empty());
}

//------------------------------------------------------------------------------
// catch_segment
//------------------------------------------------------------------------------

TEST(CatchSegment, MatchesTheCatchLoop)
{
    for (const auto& [path, options] : csv_pixels())
    {
        const MaskedPixel p(path, options);
        ASSERT_GE(p.series.size(), index_t{200}) << path;
        const auto dates = p.series.dates();
        const auto spectral = p.series.spectral();
        const index_t n = p.series.size();
        const index_t B = spectral.extent(0);

        ccd::LassoSolver solver(lasso_options());
        ccd::LassoWorkspace expected_ws(0), got_ws(0);

        // A Start catch and an End catch (to the series' end).
        for (const ccd::Window w : {ccd::Window(0, 25), ccd::Window(n - 30, n)})
        for (const ccd::CurveQA qa : {ccd::CurveQA::Start, ccd::CurveQA::End})
        {
            SCOPED_TRACE(path + " window (" + std::to_string(w.start) + ", " + std::to_string(w.stop) + ")");

            std::vector<ccd::LassoResult> expected(B);
            fit_loop(Site::Catch,
                     dates.slice(ccd::range(w.start, w.stop)),
                     spectral.slice(ccd::all(), ccd::range(w.start, w.stop)),
                     p.options.COEFFICIENT_MIN, p.options.DETECTION_BANDS,
                     expected_ws, solver, expected);

            const ccd::ChangeModel got = ccd::catch_segment(
                dates, spectral, w, p.options.COEFFICIENT_MIN, qa, got_ws, solver);

            EXPECT_EQ(got.start_day, dates(w.start));
            EXPECT_EQ(got.end_day, dates(w.stop - 1));
            EXPECT_EQ(got.break_day, w.stop >= n ? dates(n - 1) : dates(w.stop));
            EXPECT_EQ(got.observation_count, w.size());
            EXPECT_EQ(got.change_probability, 0.0);
            EXPECT_EQ(got.curve_qa, qa);
            ASSERT_EQ(got.bands.size(), B);
            for (index_t b = 0; b < B; ++b)
                expect_same_fit(got.bands[b], expected[b]);
        }
    }
}
