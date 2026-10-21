// The golden driver: generates the synthetic corpus, and dumps every output of
// detect() and detect_cube() (threads 1 and 8) on it and on the CSV pixels as
// hexfloat text, one file per input and option set.
//
//   golden_driver gen <corpus_dir>
//   golden_driver run <corpus_dir> <csv_dir> <out_dir> [bench]
//
// gen writes one file per cube; c16.bin is also tools/bench's corpus. The
// format, little-endian: uint64 H, W, T; int64 dates[T]; float64
// spectral[H][W][7][T]; uint8 qas[H][W][T]. run exits 1 if a cube's dump
// differs from the detect() dump of the same input; with bench it only times.
//
// <csv_dir> is a tree's tests/ccd/data. Build it against a tree's
// include/, tests/ccd/support/ and libccd_core.a. Every tree's dumps compare
// file for file with every other's: v0.6.1 to v0.10.0 ran it through their
// option-argument detect() and detect_cube() (this driver as of v0.10.0), 1.0
// on through the RunSpec ones. It is the golden check's driver and
// linux-compare's.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include <sys/resource.h>

#include "ccd/ccd.hpp"
#include "synthetic.hpp"
#include "dump.hpp"
#include "units.hpp"
#include "config.hpp"
#include "series.hpp"

struct CubeSpec { const char* name; std::uint64_t seed; golden::Regime regime; std::size_t H, W; };

static const CubeSpec SPECS[] = {
    {"c16",    1, golden::Regime::Cadence16,  40, 50},
    {"c8",     2, golden::Regime::Cadence8,   16, 25},
    {"dup",    3, golden::Regime::Duplicates, 10, 20},
    {"short",  4, golden::Regime::Short,      10, 10},
    {"medium", 5, golden::Regime::Medium,     10, 20},
};

static FILE* open_or_exit(const std::string& path, const char* mode)
{
    FILE* f = std::fopen(path.c_str(), mode);
    if (!f) { std::perror(path.c_str()); std::exit(1); }
    return f;
}

static void close_or_exit(const std::string& path, FILE* f)
{
    if (std::fclose(f) != 0) { std::perror(path.c_str()); std::exit(1); }
}

static void write_cube(const std::string& path, const golden::Cube& c)
{
    FILE* f = open_or_exit(path, "wb");
    std::uint64_t hdr[3] = {c.H, c.W, c.T};
    std::fwrite(hdr, sizeof(hdr), 1, f);
    std::fwrite(c.dates.data(), 8, c.T, f);
    std::fwrite(c.spectral.data(), 8, c.spectral.size(), f);
    std::fwrite(c.qas.data(), 1, c.qas.size(), f);
    close_or_exit(path, f);
}

static golden::Cube read_cube(const std::string& path)
{
    golden::Cube c;
    FILE* f = open_or_exit(path, "rb");
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

static void write_text(const std::string& path, const std::string& s)
{
    FILE* f = open_or_exit(path, "wb");
    std::fwrite(s.data(), 1, s.size(), f);
    close_or_exit(path, f);
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

// The process's peak resident set size in MB of 10^6 bytes: ru_maxrss is bytes
// on macOS, kilobytes on Linux.
static double peak_mb()
{
    struct rusage u {};
    getrusage(RUSAGE_SELF, &u);
#ifdef __APPLE__
    return static_cast<double>(u.ru_maxrss) / 1e6;
#else
    return static_cast<double>(u.ru_maxrss) * 1024.0 / 1e6;
#endif
}

static int usage()
{
    std::fprintf(stderr,
        "usage: golden_driver gen <corpus_dir>\n"
        "       golden_driver run <corpus_dir> <csv_dir> <out_dir> [bench]\n");
    return 2;
}

int main(int argc, char** argv)
{
    const std::string mode = argc > 1 ? argv[1] : "";
    const bool gen = mode == "gen" && argc == 3;
    const bool run = mode == "run" && (argc == 5 || (argc == 6 && std::string(argv[5]) == "bench"));
    if (!gen && !run) return usage();

    if (gen)
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
    const bool bench = argc == 6;
    bool mismatch = false;

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
            // The first line skipped, as the golden fixtures read these files.
            golden::Series csv;
            try
            {
                csv = golden::read_csv(csvdir + "/" + fn, true);
            }
            catch (const std::exception& e)
            {
                std::fprintf(stderr, "%s\n", e.what());
                return 1;
            }
            const std::size_t T = csv.size();
            struct Set { std::string name; ccd::Input input; ccd::config::Config config; };
            const std::vector<Set> sets = {
                {"dn", ccd::Input{}, golden::pyccd()},
                {"c2", golden::c2_input(), golden::c2_config()},
                {"phys", golden::physical_input(), golden::pyccd()},
            };
            for (auto& set : sets)
            {
                std::vector<double> spec = csv.spectral;
                if (set.name == "phys") golden::to_physical(spec.data(), T);
                auto r = ccd::detect(golden::spec_of(set.input, set.config),
                    ccd::ArrayView<const std::int64_t, 1>::contiguous(csv.dates.data(), {T}),
                    ccd::ArrayView<const double, 2>::contiguous(spec.data(), {7, T}),
                    ccd::ArrayView<const std::uint8_t, 1>::contiguous(csv.qas.data(), {T})).result;
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
                    [](const ccd::CubePixel& a, const ccd::CubePixel& b) {
                        return a.row != b.row ? a.row < b.row : a.col < b.col; });
                std::string cs;
                for (const auto& p : cube.pixels)
                    golden::dump_pixel(cs, std::to_string(p.row) + "," + std::to_string(p.col), cube, p);
                for (const auto& f : cube.failures)
                    cs += "F " + std::to_string(f.row) + "," + std::to_string(f.col) + " " + f.message + "\n";
                write_text(out + "/cube_" + spec.name + "_" + o.name + "_t" + std::to_string(threads) + ".txt", cs);

                // cube vs detect must agree too
                if (cs != s)
                {
                    std::fprintf(stderr, "cube dump differs from detect dump for %s %s t%d\n", spec.name, o.name, threads);
                    mismatch = true;
                }
            }
        }
    }

    std::fprintf(stderr, "detect: %zu px, %.1f ms, %.4f ms/px; peak %.0f MB\n",
        detect_px, detect_ms, detect_ms / detect_px, peak_mb());
    return mismatch ? 1 : 0;
}
