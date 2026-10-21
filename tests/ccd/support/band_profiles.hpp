#pragma once

//==============================================================================
//
// The synthetic band profiles of the bands fixture (test_bands.cpp), and the
// cubes it runs them on. Shared by every test that runs that corpus.
//
//==============================================================================

#include <vector>

#include "ccd/ccd.hpp"

#include "config.hpp"
#include "synthetic.hpp"
#include "synthetic_bands.hpp"

namespace golden
{

namespace bands
{

inline std::vector<ccd::index_t> iota(ccd::index_t n)
{
    std::vector<ccd::index_t> out(n);
    for (ccd::index_t i = 0; i < n; ++i)
        out[i] = i;
    return out;
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
    ccd::Input h;
    ccd::config::Config config = golden::pyccd();   // the detection bands

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

// blue, green, red, re1, re2, re3, nir, nir8a, swir1, swir2: the "nir" the
// detection bands read is row 6, so row 7 is an "other".
inline const std::vector<golden::BandSpec> S2_SPECS =
    {BLUE, GREEN, RED, RE1, RE2, RE3, NIR, NIR8A, SWIR1, SWIR2};

inline ccd::Input s2_layout()
{
    ccd::Input h;
    h.roles = {"blue", "green", "red", "red_edge_1", "red_edge_2", "red_edge_3",
               "nir", "other", "swir1", "swir2"};
    return h;
}

// `config` with band_range testing every row of the S2 layout, its "other"
// row 7 too (by row: no role names it).
inline ccd::config::Config s2_config(const ccd::config::Config& config)
{
    std::vector<ccd::index_t> rows = iota(10);
    return golden::with_steps(config, "band_range", "bands",
                              ccd::config::bands(ccd::BandList(rows)));
}

// The default configuration's detection bands (green, red, nir, swir1,
// swir2) and Tmask bands (green, swir1) are each layout's own rows.
inline std::vector<Profile> profiles()
{
    std::vector<Profile> out;

    // Sentinel-2-like: 10 bands, no thermal.
    out.push_back({"s2_b10", S2_SPECS, s2_layout(), s2_config(golden::pyccd())});

    // HLS L30-like: coastal, blue, green, red, nir, swir1, swir2, thermal.
    {
        ccd::Input h;
        h.roles = {"coastal", "blue", "green", "red", "nir", "swir1", "swir2", "thermal"};
        out.push_back({"hls_l30_b8",
            {COASTAL, BLUE, GREEN, RED, NIR, SWIR1, SWIR2, THERMAL}, h, golden::pyccd()});
    }

    // HLS-like, 7 bands without thermal.
    {
        ccd::Input h;
        h.roles = {"coastal", "blue", "green", "red", "nir", "swir1", "swir2"};
        out.push_back({"hls_b7",
            {COASTAL, BLUE, GREEN, RED, NIR, SWIR1, SWIR2}, h, golden::pyccd()});
    }

    // Sentinel-2-like with the red edge in detection: df = 8, the thresholds
    // chi2(8)'s at the default probabilities.
    {
        const ccd::config::Config c = golden::with_shared(
            golden::pyccd(), "detection_bands",
            ccd::config::roles({"green", "red", "red_edge_1", "red_edge_2", "red_edge_3", "nir",
                                "swir1", "swir2"}));
        out.push_back({"s2_b10_df8", S2_SPECS, s2_layout(), s2_config(c)});

    }

    return out;
}

inline const golden::CubeSpec CUBES[] = {
    {"c16",    1, golden::Regime::Cadence16, 6, 8},
    {"c8",     2, golden::Regime::Cadence8,  4, 5},
    {"short",  3, golden::Regime::Short,     3, 4},
    {"medium", 4, golden::Regime::Medium,    4, 5},
};

} // namespace bands

} // namespace golden
