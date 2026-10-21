// bench_preprocess
//
// apply() of each "ccdc" chain, its plan built once, against the chain written
// out by hand from the filter kernels (tests/ccd/support/preprocess_reference.hpp),
// on the pyccd reference pixel. Prints the best of 5 rounds of REPS masks, in ns
// per mask, and the process's peak memory; exits 1 when a chain's mask differs
// from the hand-written one. Built with the tests (target bench_preprocess),
// not run by ctest.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <sys/resource.h>

#include "ccd/ccd.hpp"

#include "preprocess_reference.hpp"
#include "series.hpp"

using namespace golden::chains;

int main()
{
    const golden::Series p =
        golden::read_csv(CCD_TEST_DATA_DIR "/test_3657_3610_observations.csv", true);
    const golden::PixelView px = p.view();
    const Options h;
    const pp::PreprocessPlan plan = plan_of(h);

    constexpr int REPS = 20000;
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

    struct Row
    {
        const char* name;
        const pp::Chain& chain;
        ccd::ProcessingMask (*reference)(const golden::PixelView&, const Options&);
    };

    const Row rows[] = {
        {"standard", plan.standard.chain, reference_standard},
        {"snow", plan.snow.chain, reference_snow},
        {"insufficient_clear", plan.insufficient_clear.chain, reference_insufficient_clear},
    };

    std::printf("T = %zu, best of 5 x %d, ns per mask\n", p.size(), REPS);
    bool same = true;
    for (const Row& r : rows)
    {
        const double reference = time_ns([&] { return r.reference(px, h); });
        const double chain = time_ns([&] { return run_chain(r.chain, px); });
        std::printf("%s: hand-written %.1f, apply(ccdc chain) %.1f (%.2fx)\n",
                    r.name, reference, chain, chain / reference);
        same = same && run_chain(r.chain, px).data() == r.reference(px, h).data();
    }

    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
#if defined(__APPLE__)
    const double peak_mb = static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);
#else
    const double peak_mb = static_cast<double>(usage.ru_maxrss) / 1024.0;
#endif
    std::printf("peak memory %.1f MB\n", peak_mb);

    if (!kept || !same)
    {
        std::fprintf(stderr, "bench_preprocess: %s\n",
                     !same ? "a chain's mask differs from the hand-written one"
                           : "a mask kept no observation");
        return 1;
    }
    return 0;
}
