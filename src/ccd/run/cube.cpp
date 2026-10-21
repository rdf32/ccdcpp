// The cube loop: every pixel of a cube through Run::pixel(), on OpenMP
// threads, collected in raster order (detect_cube()).

#include "run_pixel.hpp"

#ifdef _OPENMP
#include <omp.h>
#else
// A build without OpenMP runs the cube on one thread: the pragmas are
// ignored, and these are the runtime's answers for a single thread.
namespace
{
int omp_get_max_threads() { return 1; }
int omp_get_num_threads() { return 1; }
int omp_get_thread_num() { return 0; }
} // namespace
#endif

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "ccd/run/run.hpp"
#include "../core/logger.hpp"

namespace ccd
{

namespace
{

// The T bytes of `bytes` (a pixel's mask or reason codes) into its place `at`
// of the cube's (H, W, T) array.
void place(const std::vector<std::uint8_t>& bytes, index_t T, std::size_t at,
           std::vector<std::uint8_t>& cube)
{
    if (bytes.size() != T)
        throw std::logic_error("the pixel's processing mask or reason codes are not T long");
    std::copy(bytes.begin(), bytes.end(), cube.begin() + at * T);
}

// The pixel at (row, col) of a cube, run with the thread's `worker`. With
// RunSpec::masks its mask goes straight into its T bytes of `masks`, the
// cube's (H, W, T) array, and the pixel keeps none; with RunSpec::reasons
// its reason codes into `reasons` the same way.
template<typename Value, typename Qa>
CubePixel cube_pixel(
    const Run& run,
    ArrayView<const Value, 4> spectral,       // (H,W,B,T)
    ArrayView<const Qa, 3> qas,               // (H,W,T)
    index_t row,
    index_t col,
    Worker& worker,
    std::vector<std::uint8_t>& masks,
    std::vector<std::uint8_t>& reasons
)
{
    const index_t T = spectral.extent(3);
    const std::size_t at =
        static_cast<std::size_t>(row) * static_cast<std::size_t>(spectral.extent(1)) +
        static_cast<std::size_t>(col);

    auto spect_view =
        spectral.slice(fixed(row), fixed(col), all(), all()); // B, T

    auto qa_view =
        qas.slice(fixed(row), fixed(col), all()); // T

    worker.observe = run.observed(at);
    PixelOutput outcome =
        run.pixel<Value, Qa>(spect_view, qa_view, static_cast<index_t>(at), worker);

    if (run.spec().masks)
    {
        place(outcome.result.mask.data(), T, at, masks);
        outcome.result.mask = ProcessingMask();
    }
    if (run.spec().reasons)
    {
        place(outcome.result.reasons, T, at, reasons);
        outcome.result.reasons = std::vector<std::uint8_t>();
    }

    return CubePixel{row, col, std::move(outcome.result), std::move(outcome.record),
                     outcome.status};
}

// The cube's pixels that succeeded into `merged`, in raster order, and every
// thread's failures, sorted into raster order.
void collect(
    CubeOutput& merged,
    std::vector<CubePixel>& by_pixel,
    const std::vector<std::uint8_t>& succeeded,
    std::vector<std::vector<CubeFailure>>& thread_failures
)
{
    std::size_t total = 0;
    for (const auto ok : succeeded)
        total += ok;

    merged.pixels.reserve(total);

    for (std::size_t pixel = 0; pixel < by_pixel.size(); ++pixel)
    {
        if (succeeded[pixel])
            merged.pixels.push_back(std::move(by_pixel[pixel]));
    }

    by_pixel = std::vector<CubePixel>();

    std::size_t total_failed = 0;
    for (const auto& v : thread_failures)
        total_failed += v.size();

    merged.failures.reserve(total_failed);

    for (auto& v : thread_failures)
    {
        std::move(
            v.begin(),
            v.end(),
            std::back_inserter(merged.failures)
        );
    }

    std::sort(
        merged.failures.begin(),
        merged.failures.end(),
        [](const CubeFailure& a, const CubeFailure& b)
        {
            return a.row != b.row ? a.row < b.row : a.col < b.col;
        });
}

// The cube loop for one pair of element types: every pixel through
// cube_pixel(), on `threads` threads, then collect().
template<typename Value, typename Qa>
CubeOutput cube_loop(
    const Run& run,
    ArrayView<const Value, 4> spectral,       // (H,W,B,T)
    ArrayView<const Qa, 3> qas,               // (H,W,T)
    int threads
)
{
    const index_t H = spectral.extent(0);
    const index_t W = spectral.extent(1);
    const index_t T = spectral.extent(3);

    // threads == 0 keeps the historical behaviour: let the OpenMP runtime
    // decide, which is OMP_NUM_THREADS or the core count. A positive value is
    // a request, not a guarantee -- the runtime may still hand back fewer, so
    // the count reported below is read from inside the region.
    const int requested =
        threads > 0 ? threads : omp_get_max_threads();

    //----------------------------------------------------------------------
    // One slot per pixel, written by whichever thread runs it, so the
    // result comes back in raster order whatever the schedule did.
    //----------------------------------------------------------------------
    const std::int64_t npixels =
        static_cast<std::int64_t>(H) *
        static_cast<std::int64_t>(W);

    std::vector<CubePixel> by_pixel(static_cast<std::size_t>(npixels));
    std::vector<std::uint8_t> succeeded(static_cast<std::size_t>(npixels), 0);
    std::vector<std::vector<CubeFailure>> thread_failures(requested);

    CubeOutput merged;
    merged.height = H;
    merged.width = W;
    merged.timesteps = T;
    if (run.spec().masks)
        merged.masks.assign(static_cast<std::size_t>(npixels) * T, std::uint8_t{0});
    if (run.spec().reasons)
        merged.reasons.assign(static_cast<std::size_t>(npixels) * T, REASON_COUNT);

    int used = requested;

    #pragma omp parallel num_threads(requested)
    {
        const int tid = omp_get_thread_num();

        #pragma omp single
        used = omp_get_num_threads();

        auto& failures = thread_failures[tid];

        Worker worker;

        #pragma omp for schedule(dynamic)
        for (std::int64_t pixel = 0; pixel < npixels; ++pixel)
        {
            const index_t row =
                static_cast<index_t>(pixel) / W;

            const index_t col =
                static_cast<index_t>(pixel) % W;

            LOG_DEBUG(
                "thread "
                << omp_get_thread_num()
                << " pixel "
                << row << "," << col
            );

            //------------------------------------------------------------------
            // The try/catch is load-bearing, not defensive habit: an
            // exception that escapes an OpenMP region terminates the process,
            // so without this one pathological time series destroys every
            // other pixel's result in the cube and reports nothing about
            // which pixel did it. Catching per pixel turns that into a
            // recorded address plus a message, and the other 249,999 pixels
            // still come back.
            //
            // Nothing is swallowed -- every catch lands in
            // CubeOutput::failures, and it is the caller's job to surface the
            // count.
            //------------------------------------------------------------------
            try
            {
                const std::size_t at = static_cast<std::size_t>(pixel);
                by_pixel[at] =
                    cube_pixel<Value, Qa>(run, spectral, qas, row, col, worker, merged.masks,
                                          merged.reasons);
                succeeded[at] = 1;
            }
            catch (const std::exception& e)
            {
                failures.emplace_back(
                    CubeFailure{
                        row,
                        col,
                        std::string(e.what())
                    }
                );
            }
            catch (...)
            {
                failures.emplace_back(
                    CubeFailure{
                        row,
                        col,
                        std::string("unknown non-std exception")
                    }
                );
            }
        }
    }

    merged.threads = used;
    collect(merged, by_pixel, succeeded, thread_failures);
    merged.header = run.header();
    return merged;
}

} // namespace

CubeOutput detect_cube(
    const RunSpec& spec,
    ArrayView<const std::int64_t, 1> dates,
    const CubeSpectral& spectral,
    const CubeQa& qas,
    int threads,
    const Resume* resume
)
{
    return std::visit(
        [&](auto s, auto q) -> CubeOutput
        {
            using Value = std::remove_const_t<typename decltype(s)::value_type>;
            using Qa = std::remove_const_t<typename decltype(q)::value_type>;

            const index_t H = s.extent(0);
            const index_t W = s.extent(1);
            const index_t T = s.extent(3);
            if (q.extent(0) != H || q.extent(1) != W || q.extent(2) != T)
                throw std::invalid_argument("qas shape must be (height, width, timesteps)");

            const Run run(spec, dates, s.extent(2), T, q.extent(2),
                          std::is_same_v<Qa, std::uint16_t>, resume);

            if (resume)
            {
                const std::size_t cells =
                    static_cast<std::size_t>(H) * static_cast<std::size_t>(W);
                if (resume->records.size() != cells || resume->segments.size() != cells)
                    throw std::invalid_argument(
                        "the resume records hold " + std::to_string(resume->records.size()) +
                        " pixels and " + std::to_string(resume->segments.size()) +
                        " segment lists; the cube has " + std::to_string(cells));
            }

            return cube_loop<Value, Qa>(run, s, q, threads);
        },
        spectral, qas);
}

} // namespace ccd
