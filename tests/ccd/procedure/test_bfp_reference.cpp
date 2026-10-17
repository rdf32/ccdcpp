//==============================================================================
//
// The bfp-paper method against JuliaCCD's paperbfp, the reference
// implementation of Tollerud et al. 2023's BFP (JuliaCCD 1.1.1, run on the
// same inputs).
//
// bfp_reference_pixels.csv holds twelve pixels of the Atlanta h024v013 cell
// (Landsat Collection 2 ARD, 1984-2023), as Collection 1 DN with QA codes,
// their fill rows left out (both implementations drop them first). They
// were chosen from a 1500-pixel comparison, all 1500 equal in structure, to
// hold every rule that comparison exercised: five changes in one series,
// starts found only with paperbfp's fixed initialization threshold and peek
// and its whole-series statistics, a Lookback drop, a start window reaching
// the last observation, a shortened peek that ends the series undecided,
// Tmask drops undone after a search that finds no start, Start and End
// segments fitted by their intercept, and full ones.
//
// bfp_reference_segments.csv holds JuliaCCD's segments: days, break day,
// change, observation count, the kept observations of the pixel, every
// band's RMSE and magnitude (its reflectance x 1e4, ccdcpp's DN).
//
// The structure is pinned exactly; the RMSE and magnitudes to 1e-6
// relative (the 1500 pixels' largest difference was 2.4e-7 for the RMSE,
// 1.1e-7 for a magnitude): two implementations of least squares, not one
// machine's bits.
//
//==============================================================================

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ccd/ccd.hpp"

namespace
{

using ccd::index_t;
using ccd::scalar_t;

struct Pixel
{
    std::vector<std::int64_t> dates;
    std::vector<scalar_t> spectral;   // filled band-major after reading
    std::vector<std::array<scalar_t, 7>> rows;
    std::vector<std::uint8_t> qas;
};

struct Segment
{
    index_t kept;
    std::int64_t start_day, end_day, break_day;
    bool change;
    index_t observation_count;
    scalar_t rmse[7], magnitude[7];
};

std::vector<std::string> fields(const std::string& line)
{
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string f;
    while (std::getline(ss, f, ','))
        out.push_back(f);
    return out;
}

std::map<std::string, Pixel> read_pixels()
{
    std::ifstream in(CCD_BFP_REFERENCE_PIXELS);
    std::string line;
    std::getline(in, line);
    std::map<std::string, Pixel> out;
    while (std::getline(in, line))
    {
        const auto f = fields(line);
        Pixel& p = out[f[0]];
        p.dates.push_back(std::stoll(f[1]));
        std::array<scalar_t, 7> row;
        for (int b = 0; b < 7; ++b)
            row[b] = std::stod(f[2 + b]);
        p.rows.push_back(row);
        p.qas.push_back(static_cast<std::uint8_t>(std::stoi(f[9])));
    }
    for (auto& [id, p] : out)
    {
        const std::size_t T = p.dates.size();
        p.spectral.resize(7 * T);
        for (std::size_t t = 0; t < T; ++t)
            for (int b = 0; b < 7; ++b)
                p.spectral[b * T + t] = p.rows[t][b];
    }
    return out;
}

std::map<std::string, std::vector<Segment>> read_segments()
{
    std::ifstream in(CCD_BFP_REFERENCE_SEGMENTS);
    std::string line;
    std::getline(in, line);
    std::map<std::string, std::vector<Segment>> out;
    while (std::getline(in, line))
    {
        const auto f = fields(line);
        Segment s;
        s.kept = std::stoul(f[1]);
        s.start_day = std::stoll(f[2]);
        s.end_day = std::stoll(f[3]);
        s.break_day = std::stoll(f[4]);
        s.change = f[5] == "1";
        s.observation_count = std::stoul(f[6]);
        for (int b = 0; b < 7; ++b)
        {
            s.rmse[b] = std::stod(f[7 + b]);
            s.magnitude[b] = std::stod(f[14 + b]);
        }
        out[f[0]].push_back(s);
    }
    return out;
}

bool near(scalar_t a, scalar_t b, scalar_t rel)
{
    return std::abs(a - b) <= rel * std::max(std::abs(a), std::abs(b)) + 1e-9;
}

} // namespace

TEST(BfpReference, EqualsJuliaCcdPaperbfp)
{
    const auto pixels = read_pixels();
    const auto expected = read_segments();
    ASSERT_EQ(pixels.size(), 12u);
    ASSERT_EQ(expected.size(), 12u);

    const ccd::HarmonicOptions h =
        ccd::with_method(ccd::HarmonicOptions{}, ccd::ChangeMethod::BfpPaper);
    const ccd::LassoOptions l;

    std::size_t segments = 0, changes = 0;
    for (const auto& [id, p] : pixels)
    {
        const std::size_t T = p.dates.size();
        const ccd::FitResult r = ccd::detect(
            ccd::ArrayView<const std::int64_t, 1>::contiguous(p.dates.data(), {T}),
            ccd::ArrayView<const scalar_t, 2>::contiguous(p.spectral.data(), {7, T}),
            ccd::ArrayView<const std::uint8_t, 1>::contiguous(p.qas.data(), {T}), h, l);

        const auto& want = expected.at(id);
        ASSERT_EQ(r.models.size(), want.size()) << id;
        EXPECT_EQ(r.mask.count(), want[0].kept) << id;
        for (std::size_t k = 0; k < want.size(); ++k)
        {
            const ccd::ChangeModel& m = r.models[k];
            const Segment& w = want[k];
            const std::string at = id + " segment " + std::to_string(k);
            EXPECT_EQ(m.start_day, w.start_day) << at;
            EXPECT_EQ(m.end_day, w.end_day) << at;
            EXPECT_EQ(m.change_probability == 1.0, w.change) << at;
            EXPECT_EQ(m.observation_count, w.observation_count) << at;
            if (w.change)
                EXPECT_EQ(m.break_day, w.break_day) << at;
            for (int b = 0; b < 7; ++b)
            {
                EXPECT_TRUE(near(m.bands[b].rmse, w.rmse[b], 1e-6))
                    << at << " band " << b << " rmse " << m.bands[b].rmse << " vs " << w.rmse[b];
                if (w.change)
                    EXPECT_TRUE(near(m.bands[b].magnitude, w.magnitude[b], 1e-6))
                        << at << " band " << b << " magnitude " << m.bands[b].magnitude << " vs "
                        << w.magnitude[b];
            }
            ++segments;
            changes += w.change;
        }
    }
    EXPECT_EQ(segments, 31u);
    EXPECT_GT(changes, 15u);
}
