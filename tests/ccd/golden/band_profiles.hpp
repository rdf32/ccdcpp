#pragma once

//==============================================================================
//
// The synthetic band profiles of the bands fixture (test_bands.cpp), and the
// cubes it runs them on. Shared by every test that runs that corpus.
//
//==============================================================================

#include <vector>

#include <boost/math/distributions/chi_squared.hpp>

#include "ccd/ccd.hpp"

#include "synthetic.hpp"
#include "synthetic_bands.hpp"

namespace golden
{

namespace bands
{

inline ccd::BandList band_list(const std::vector<ccd::index_t>& bands)
{
    return ccd::BandList(bands);
}

inline std::vector<ccd::index_t> iota(ccd::index_t n)
{
    std::vector<ccd::index_t> out(n);
    for (ccd::index_t i = 0; i < n; ++i)
        out[i] = i;
    return out;
}

inline double chi2_quantile(ccd::index_t df, double p)
{
    return boost::math::quantile(
        boost::math::chi_squared(static_cast<double>(df)), p);
}

//------------------------------------------------------------------------------
// Synthetic profiles
//
// Shaped after sensors ccdcpp may be pointed at, built here only. Values are
// DN as synthetic_bands.hpp makes them.
//------------------------------------------------------------------------------

struct Profile
{
    const char* name;
    std::vector<golden::BandSpec> specs;
    ccd::HarmonicOptions h;
};

inline const golden::BandSpec COASTAL = {400, 100, 25};
inline const golden::BandSpec BLUE    = {450, 120, 25};
inline const golden::BandSpec GREEN   = {750, 180, 30};
inline const golden::BandSpec RED     = {650, 220, 35};
inline const golden::BandSpec RE1     = {1100, 400, 40};
inline const golden::BandSpec RE2     = {2000, 700, 60};
inline const golden::BandSpec RE3     = {2500, 850, 80};
inline const golden::BandSpec NIR     = {2800, 900, 90};
inline const golden::BandSpec NIR8A   = {2900, 920, 90};
inline const golden::BandSpec SWIR1   = {1900, 450, 60};
inline const golden::BandSpec SWIR2   = {1100, 300, 45};
inline const golden::BandSpec THERMAL = {2950, 120, 15, true};

// blue, green, red, re1, re2, re3, nir, nir8a, swir1, swir2
inline const std::vector<golden::BandSpec> S2_SPECS =
    {BLUE, GREEN, RED, RE1, RE2, RE3, NIR, NIR8A, SWIR1, SWIR2};

inline ccd::HarmonicOptions s2_layout()
{
    ccd::HarmonicOptions h;
    h.NUM_BANDS = 10;
    h.DETECTION_BANDS = {1, 2, 6, 8, 9};
    h.TMASK_BANDS = {1, 8};
    h.SATURATION_BANDS = band_list(iota(10));
    h.GREEN_IDX = 1;
    h.THERMAL_IDX = ccd::NO_BAND;
    return h;
}

inline std::vector<Profile> profiles()
{
    std::vector<Profile> out;

    // Sentinel-2-like: 10 bands, no thermal.
    out.push_back({"s2_b10", S2_SPECS, s2_layout()});

    // HLS L30-like: coastal, blue, green, red, nir, swir1, swir2, thermal.
    {
        ccd::HarmonicOptions h;
        h.NUM_BANDS = 8;
        h.DETECTION_BANDS = {2, 3, 4, 5, 6};
        h.TMASK_BANDS = {2, 5};
        h.SATURATION_BANDS = band_list(iota(7));
        h.GREEN_IDX = 2;
        h.THERMAL_IDX = 7;
        out.push_back({"hls_l30_b8",
            {COASTAL, BLUE, GREEN, RED, NIR, SWIR1, SWIR2, THERMAL}, h});
    }

    // HLS-like, 7 bands without thermal.
    {
        ccd::HarmonicOptions h;
        h.DETECTION_BANDS = {2, 3, 4, 5, 6};
        h.TMASK_BANDS = {2, 5};
        h.SATURATION_BANDS = band_list(iota(7));
        h.GREEN_IDX = 2;
        h.THERMAL_IDX = ccd::NO_BAND;
        out.push_back({"hls_b7",
            {COASTAL, BLUE, GREEN, RED, NIR, SWIR1, SWIR2}, h});
    }

    // Sentinel-2-like with the red edge in detection: df = 8, thresholds
    // from chi2(8) at the default probabilities.
    {
        ccd::HarmonicOptions h = s2_layout();
        h.DETECTION_BANDS = {1, 2, 3, 4, 5, 6, 8, 9};
        h.CHANGE_THRESHOLD = chi2_quantile(8, 0.99);
        h.OUTLIER_THRESHOLD = chi2_quantile(8, 1.0 - 1e-6);
        out.push_back({"s2_b10_df8", S2_SPECS, h});
    }

    return out;
}

struct CubeSpec
{
    const char* name;
    std::uint64_t seed;
    golden::Regime regime;
    std::size_t H;
    std::size_t W;
};

inline const CubeSpec CUBES[] = {
    {"c16",    1, golden::Regime::Cadence16, 6, 8},
    {"c8",     2, golden::Regime::Cadence8,  4, 5},
    {"short",  3, golden::Regime::Short,     3, 4},
    {"medium", 4, golden::Regime::Medium,    4, 5},
};

} // namespace bands

} // namespace golden
