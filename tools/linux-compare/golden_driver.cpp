// The golden driver: generates the synthetic corpus, and dumps every output of
// detect() and detect_cube() (threads 1 and 8) on it and on the CSV pixels as
// hexfloat text, one file per input and option set.
//
//   golden_driver gen <corpus_dir>
//   golden_driver run <corpus_dir> <csv_dir> <out_dir> [bench]
//
// <csv_dir> is a tree's tests/ccd/machine. Build it against a tree's
// include/, tests/ccd/golden/ and libccd_core.a. Every tree's dumps compare
// file for file with every other's: v0.6.1 to v0.10.0 ran it through their
// option-argument detect() and detect_cube() (this driver as of v0.10.0), 1.0
// on through the RunSpec ones. This is the driver of the refactor's golden
// check, kept here for linux-compare.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "ccd/ccd.hpp"
#include "synthetic.hpp"
#include "golden.hpp"
#include "specs.hpp"

struct CubeSpec { const char* name; std::uint64_t seed; golden::Regime regime; std::size_t H, W; };

static const CubeSpec SPECS[] = {
    {"c16",    1, golden::Regime::Cadence16,  40, 50},
    {"c8",     2, golden::Regime::Cadence8,   16, 25},
    {"dup",    3, golden::Regime::Duplicates, 10, 20},
    {"short",  4, golden::Regime::Short,      10, 10},
    {"medium", 5, golden::Regime::Medium,     10, 20},
};

static void write_cube(const std::string& path, const golden::Cube& c)
{
    FILE* f = std::fopen(path.c_str(), "wb");
    std::uint64_t hdr[3] = {c.H, c.W, c.T};
    std::fwrite(hdr, sizeof(hdr), 1, f);
    std::fwrite(c.dates.data(), 8, c.T, f);
    std::fwrite(c.spectral.data(), 8, c.spectral.size(), f);
    std::fwrite(c.qas.data(), 1, c.qas.size(), f);
    std::fclose(f);
}

static golden::Cube read_cube(const std::string& path)
{
    golden::Cube c;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::perror(path.c_str()); std::exit(1); }
    std::uint64_t hdr[3];
    if (std::fread(hdr, sizeof(hdr), 1, f) != 1) std::exit(1);
    c.H = hdr[0]; c.W = hdr[1]; c.T = hdr[2];
    c.dates.resize(c.T);
    c.spectral.resize(c.H * c.W * 7 * c.T);
    c.qas.resize(c.H * c.W * c.T);
    if (std::fread(c.dates.data(), 8, c.T, f) != c.T) std::exit(1);
    if (std::fread(c.spectral.data(), 8, c.spectral.size(), f) != c.spectral.size()) std::exit(1);
    if (std::fread(c.qas.data(), 1, c.qas.size(), f) != c.qas.size()) std::exit(1);
    std::fclose(f);
    return c;
}

struct Csv { std::vector<std::int64_t> dates; std::vector<double> spectral; std::vector<std::uint8_t> qas; std::size_t T; };

static Csv read_csv(const std::string& path)
{
    std::ifstream file(path);
    if (!file) { std::perror(path.c_str()); std::exit(1); }
    std::string line;
    std::array<std::vector<double>, 7> bands;
    Csv out;
    std::getline(file, line);
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string v;
        std::getline(ss, v, ',');
        out.dates.push_back(static_cast<std::int64_t>(std::stod(v)));
        for (int b = 0; b < 7; ++b) { std::getline(ss, v, ','); bands[b].push_back(std::stod(v)); }
        std::getline(ss, v, ',');
        out.qas.push_back(static_cast<std::uint8_t>(std::stod(v)));
    }
    out.T = out.dates.size();
    out.spectral.resize(7 * out.T);
    for (int b = 0; b < 7; ++b)
        for (std::size_t t = 0; t < out.T; ++t)
            out.spectral[b * out.T + t] = bands[b][t];
    return out;
}

static void write_text(const std::string& path, const std::string& s)
{
    FILE* f = std::fopen(path.c_str(), "wb");
    std::fwrite(s.data(), 1, s.size(), f);
    std::fclose(f);
}

// Each runs the "ccdc" preset on its input.
struct OptSet { const char* name; ccd::Input h; int transform; }; // 0 none, 1 phys, 2 frac

static void apply_transform(golden::Cube& c, int transform)
{
    const std::size_t T = c.T;
    if (transform == 1)
    {
        for (std::size_t p = 0; p < c.H * c.W; ++p)
            golden::to_physical(c.spectral.data() + p * 7 * T, T);
    }
    else if (transform == 2)
    {
        golden::Rng rng(77);
        for (auto& v : c.spectral)
            if (v != -9999.0) v = v + (rng.uniform() - 0.5) * 0.999;
    }
}

static double now_ms()
{
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv)
{
    if (argc < 3) { std::fprintf(stderr, "usage\n"); return 2; }
    const std::string mode = argv[1];

    if (mode == "gen")
    {
        for (const auto& s : SPECS)
        {
            auto c = golden::make_cube(s.seed, s.regime, s.H, s.W);
            write_cube(std::string(argv[2]) + "/" + s.name + ".bin", c);
            std::printf("%s H=%zu W=%zu T=%zu\n", s.name, c.H, c.W, c.T);
        }
        return 0;
    }

    const std::string corpus = argv[2];
    const std::string csvdir = argv[3];
    const std::string out = argv[4];
    const bool bench = argc > 5;

    std::vector<OptSet> opts = {
        {"dn", ccd::Input{}, 0},
        {"phys", golden::physical_input(), 1},
        {"frac", ccd::Input{}, 2},
    };

    double detect_ms = 0.0;
    std::size_t detect_px = 0;

    //--------------------------------------------------------------------------
    // CSV pixels
    //--------------------------------------------------------------------------
    if (!bench)
    {
        std::string s;
        const char* files[] = {"test_3657_3610_observations.csv", "test_c2.csv", "test002_c2.csv"};
        for (const char* fn : files)
        {
            auto csv = read_csv(csvdir + "/" + fn);
            struct Set { std::string name; ccd::Input input; ccd::config::Config config; };
            const std::vector<Set> sets = {
                {"dn", ccd::Input{}, golden::pyccd()},
                {"c2", golden::c2_input(), golden::c2_config()},
                {"phys", golden::physical_input(), golden::pyccd()},
            };
            for (auto& set : sets)
            {
                std::vector<double> spec = csv.spectral;
                if (set.name == "phys") golden::to_physical(spec.data(), csv.T);
                auto r = ccd::detect(golden::spec_of(set.input, set.config),
                    ccd::ArrayView<const std::int64_t, 1>::contiguous(csv.dates.data(), {csv.T}),
                    ccd::ArrayView<const double, 2>::contiguous(spec.data(), {7, csv.T}),
                    ccd::ArrayView<const std::uint8_t, 1>::contiguous(csv.qas.data(), {csv.T})).result;
                golden::dump_pixel(s, std::string(fn) + ":" + set.name, r);
            }
        }
        write_text(out + "/csv.txt", s);
    }

    //--------------------------------------------------------------------------
    // Synthetic cubes
    //--------------------------------------------------------------------------
    for (const auto& spec : SPECS)
    {
        const auto base = read_cube(corpus + "/" + spec.name + ".bin");
        for (auto& o : opts)
        {
            if (o.transform == 2 && std::string(spec.name) != "c16") continue;
            if (bench && o.transform != 0) continue;

            golden::Cube c = base;
            apply_transform(c, o.transform);
            const std::size_t T = c.T, N = c.H * c.W;

            auto dates = ccd::ArrayView<const std::int64_t, 1>::contiguous(c.dates.data(), {T});

            // detect, per pixel, fresh copy each
            std::string s;
            std::vector<double> px(7 * T);
            const double t0 = now_ms();
            for (std::size_t p = 0; p < N; ++p)
            {
                std::copy(c.spectral.begin() + p * 7 * T, c.spectral.begin() + (p + 1) * 7 * T, px.begin());
                auto r = ccd::detect(golden::spec_of(o.h),
                    dates,
                    ccd::ArrayView<const double, 2>::contiguous(px.data(), {7, T}),
                    ccd::ArrayView<const std::uint8_t, 1>::contiguous(c.qas.data() + p * T, {T})).result;
                if (!bench)
                    golden::dump_pixel(s, std::to_string(p / c.W) + "," + std::to_string(p % c.W), r);
            }
            const double t1 = now_ms();
            detect_ms += t1 - t0;
            detect_px += N;
            if (!bench)
                write_text(out + "/detect_" + spec.name + "_" + o.name + ".txt", s);

            for (int threads : {1, 8})
            {
                std::vector<double> cube_spec = c.spectral;
                const double c0 = now_ms();
                auto cube = ccd::detect_cube(golden::spec_of(o.h),
                    dates,
                    ccd::ArrayView<const double, 4>::contiguous(cube_spec.data(), {c.H, c.W, 7, T}),
                    ccd::ArrayView<const std::uint8_t, 3>::contiguous(c.qas.data(), {c.H, c.W, T}),
                    threads);
                const double c1 = now_ms();
                std::fprintf(stderr, "cube %s %s threads=%d: %.1f ms (%.3f ms/px)\n",
                    spec.name, o.name, threads, c1 - c0, (c1 - c0) / N);
                if (bench) continue;

                std::sort(cube.pixels.begin(), cube.pixels.end(),
                    [](const ccd::PixelResult& a, const ccd::PixelResult& b) {
                        return a.row != b.row ? a.row < b.row : a.col < b.col; });
                std::string cs;
                for (const auto& p : cube.pixels)
                    golden::dump_pixel(cs, std::to_string(p.row) + "," + std::to_string(p.col), cube, p);
                for (const auto& f : cube.failures)
                    cs += "F " + std::to_string(f.row) + "," + std::to_string(f.col) + " " + f.message + "\n";
                write_text(out + "/cube_" + spec.name + "_" + o.name + "_t" + std::to_string(threads) + ".txt", cs);

                // cube vs detect must agree too
                if (cs != s) std::fprintf(stderr, "  NOTE: cube dump differs from detect dump for %s %s t%d\n", spec.name, o.name, threads);
            }
        }
    }

    std::fprintf(stderr, "detect: %zu px, %.1f ms, %.4f ms/px\n", detect_px, detect_ms, detect_ms / detect_px);
    return 0;
}
