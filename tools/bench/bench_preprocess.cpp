// bench_preprocess [hand | apply]
//
// apply() of each "ccdc" chain, its plan built once, against the chain written
// out by hand from the filter kernels (tests/ccd/support/preprocess_reference.hpp),
// on the pyccd reference pixel. Each side runs in its own process: without an
// argument the bench checks that every chain's mask is the hand-written one's,
// then runs itself once per side and prints the best of 5 rounds of REPS
// masks, in ns per mask, and each side's peak memory in MB of 10^6 bytes, with
// how far it is above the peak once the pixel is read and the plan built. With
// `hand` or `apply` it times that side alone. Exits 1 when a chain's mask
// differs from the hand-written one. Built with the tests (target
// bench_preprocess), not run by ctest.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <sys/resource.h>

#include "ccd/ccd.hpp"

#include "preprocess_reference.hpp"
#include "series.hpp"

using namespace golden::chains;

namespace
{

constexpr int REPS = 20000;

// The process's peak resident set size in MB of 10^6 bytes: ru_maxrss is bytes
// on macOS, kilobytes on Linux.
double peak_mb()
{
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
#if defined(__APPLE__)
    return static_cast<double>(usage.ru_maxrss) / 1e6;
#else
    return static_cast<double>(usage.ru_maxrss) * 1024.0 / 1e6;
#endif
}

struct Row
{
    const char* name;
    const pp::Chain& chain;
    ccd::ProcessingMask (*reference)(const golden::PixelView&, const Options&);
};

// One side, timed: a line "<chain> <ns per mask>" per chain, then
// "peak <MB> <MB above the pixel and plan>". Returns 1 if a mask kept nothing.
int time_side(bool hand)
{
    const golden::Series p =
        golden::read_csv(CCD_TEST_DATA_DIR "/test_3657_3610_observations.csv", true);
    const golden::PixelView px = p.view();
    const Options h;
    const pp::PreprocessPlan plan = plan_of(h);
    const Row rows[] = {
        {"standard", plan.standard.chain, reference_standard},
        {"snow", plan.snow.chain, reference_snow},
        {"insufficient_clear", plan.insufficient_clear.chain, reference_insufficient_clear},
    };
    const double input_mb = peak_mb();

    using clock = std::chrono::steady_clock;
    bool kept = true;
    auto time_ns = [&](auto&& build)
    {
        std::size_t sink = 0;
        double best = std::numeric_limits<double>::infinity();
        for (int round = 0; round < 5; ++round)
        {
            const auto start = clock::now();
            for (int i = 0; i < REPS; ++i)
                sink += build().count();
            const auto stop = clock::now();
            best = std::min(best,
                std::chrono::duration<double, std::nano>(stop - start).count() / REPS);
        }
        kept = kept && sink > 0;
        return best;
    };
    for (const Row& r : rows)
    {
        const double best = hand ? time_ns([&] { return r.reference(px, h); })
                                 : time_ns([&] { return run_chain(r.chain, px); });
        std::printf("%s %.1f\n", r.name, best);
    }
    std::printf("peak %.1f %.1f\n", peak_mb(), peak_mb() - input_mb);
    return kept ? 0 : 1;
}

// Every chain's mask is the hand-written one's.
bool masks_agree()
{
    const golden::Series p =
        golden::read_csv(CCD_TEST_DATA_DIR "/test_3657_3610_observations.csv", true);
    const golden::PixelView px = p.view();
    const Options h;
    const pp::PreprocessPlan plan = plan_of(h);
    return run_chain(plan.standard.chain, px).data() == reference_standard(px, h).data()
        && run_chain(plan.snow.chain, px).data() == reference_snow(px, h).data()
        && run_chain(plan.insufficient_clear.chain, px).data()
               == reference_insufficient_clear(px, h).data();
}

// One side's output, from its own process.
struct Side
{
    double ns[3] = {0.0, 0.0, 0.0};
    double peak = 0.0, above = 0.0;
};

bool run_side(const char* self, const char* side, Side& out)
{
    const std::string command = std::string("'") + self + "' " + side;
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe)
        return false;
    char name[64];
    double value = 0.0;
    int chain = 0;
    while (std::fscanf(pipe, "%63s %lf", name, &value) == 2)
    {
        if (std::strcmp(name, "peak") == 0)
        {
            out.peak = value;
            if (std::fscanf(pipe, "%lf", &out.above) != 1)
                break;
        }
        else if (chain < 3)
            out.ns[chain++] = value;
    }
    return pclose(pipe) == 0 && chain == 3;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc == 2 && (std::strcmp(argv[1], "hand") == 0 || std::strcmp(argv[1], "apply") == 0))
        return time_side(std::strcmp(argv[1], "hand") == 0);
    if (argc != 1)
    {
        std::fprintf(stderr, "usage: bench_preprocess [hand | apply]\n");
        return 2;
    }

    if (!masks_agree())
    {
        std::fprintf(stderr, "bench_preprocess: a chain's mask differs from the hand-written one\n");
        return 1;
    }
    Side hand, apply;
    if (!run_side(argv[0], "hand", hand) || !run_side(argv[0], "apply", apply))
    {
        std::fprintf(stderr, "bench_preprocess: a side failed (or a mask kept no observation)\n");
        return 1;
    }

    const char* names[] = {"standard", "snow", "insufficient_clear"};
    std::printf("best of 5 x %d, ns per mask, each side in its own process\n", REPS);
    for (int i = 0; i < 3; ++i)
        std::printf("%s: hand-written %.1f, apply(ccdc chain) %.1f (%.2fx)\n",
                    names[i], hand.ns[i], apply.ns[i], apply.ns[i] / hand.ns[i]);
    std::printf("peak memory: hand-written %.1f MB (%.1f above the input), "
                "apply %.1f MB (%.1f above the input)\n",
                hand.peak, hand.above, apply.peak, apply.above);
    return 0;
}
