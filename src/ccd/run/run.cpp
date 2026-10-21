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
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "ccd/ccd.hpp"
#include "ccd/input/qa_statistics.hpp"
#include "../core/logger.hpp"
#include "ccd/strategies/library.hpp"

#include "../core/fnv.hpp"

namespace ccd
{

namespace
{

//==============================================================================
// The checks a Run makes before any pixel
//==============================================================================

// The input against the band count, one timestep count shared by dates,
// spectral and the QA, and dates in non-decreasing order.
void check_shapes(
    const Input& input,
    index_t bands,
    index_t timesteps,
    ArrayView<const std::int64_t, 1> dates,
    index_t qa_timesteps
)
{
    validate(input, bands);

    if (dates.size() != timesteps)
    {
        throw std::invalid_argument(
            "dates has " + std::to_string(dates.size()) +
            " timesteps; spectral has " + std::to_string(timesteps)
        );
    }

    if (qa_timesteps != timesteps)
    {
        throw std::invalid_argument(
            "qas has " + std::to_string(qa_timesteps) +
            " timesteps; spectral has " + std::to_string(timesteps)
        );
    }

    validate_dates(dates);
}

// Raw (uint16) QA has no meaning without a table.
void check_qa(bool raw_qa, const QaTable* qa_table)
{
    if (raw_qa && qa_table == nullptr)
    {
        throw std::invalid_argument(
            "qas is uint16, which holds raw QA values; give the input a QaTable "
            "(qa_table) to decode them, or pass uint8 QA codes"
        );
    }
}

// A resume's refusals: another record format or results version, another
// input (its QA table included) or configuration, a series that ends
// before the record's.
void check_resume(
    const ResumeHeader& from,
    std::uint64_t digest,
    ArrayView<const std::int64_t, 1> dates
)
{
    check_version(from);
    if (from.digest != digest)
        throw std::invalid_argument(
            "the resume record was made with another input or configuration");
    if (dates.size() == 0 || dates(dates.size() - 1) < from.series_end)
        throw std::invalid_argument(
            "the series ends before the resume record's, on " +
            std::to_string(from.series_end));
}

// A resumed pixel's refusal: fewer stored segments than its record's final
// ones.
void check_stored(const PixelRecord& from, const std::vector<ChangeModel>& stored)
{
    if (from.procedure == PixelProcedure::Standard &&
        stored.size() < from.standard.final_segments)
        throw std::invalid_argument(
            "the resume record has " + std::to_string(from.standard.final_segments) +
            " final segments; " + std::to_string(stored.size()) + " were given");
}

//==============================================================================
// Resume: the cutoffs of a run resumed from one whose series ended at `end`
//==============================================================================

// statistics_end capped at `end`.
config::Shared capped(const config::Shared& shared, std::int64_t end)
{
    config::Shared out = shared;
    out.statistics_end = std::min(out.statistics_end, end);
    return out;
}

// Every GreenMedian step's date capped at `end`.
preprocess::PreprocessPlan capped(const preprocess::PreprocessPlan& plan, std::int64_t end)
{
    preprocess::PreprocessPlan out = plan;
    for (PixelProcedure procedure : PROCEDURES)
    {
        preprocess::Chain& chain = out.of(procedure).chain;
        for (index_t i = 0; i < chain.size(); ++i)
        {
            preprocess::Step& step = chain.steps[i];
            if (step.kind == preprocess::StepKind::GreenMedian)
                step.max_date = std::min(step.max_date, end);
        }
    }
    return out;
}

// The exits a resume continues the Standard machine from; the others (the
// early exits) are rerun, as are the other procedures.
bool resumable(standard::Exit exit)
{
    using standard::Exit;
    return exit == Exit::NoRoomToInitialize || exit == Exit::InitializeExhausted ||
        exit == Exit::PeekBeyondEnd || exit == Exit::SeriesEnd;
}

//==============================================================================
// The steps of one pixel
//==============================================================================

// Step 1. The pixel's QA codes: a uint8 QA without a table is the codes and
// is returned as it is; otherwise every value is decoded into `buffer`, which
// is returned. Raw QA always has a table here (check_qa()).
template<typename Qa>
ArrayView<const std::uint8_t, 1> decode_qa(
    ArrayView<const Qa, 1> qas,
    const QaTable* qa_table,
    std::vector<std::uint8_t>& buffer
)
{
    if constexpr (std::is_same_v<Qa, std::uint8_t>)
    {
        if (qa_table == nullptr)
            return qas;
    }

    const index_t T = qas.size();
    buffer.resize(T);

    const auto& code = qa_table->code;
    for (index_t t = 0; t < T; ++t)
        buffer[t] = code[qas(t)];

    return ArrayView<const std::uint8_t, 1>::contiguous(buffer.data(), {T});
}

// Step 2. The pixel's procedure and mode, from its QA statistics up to
// statistics_end.
PixelMode choose_procedure(
    ArrayView<const std::uint8_t, 1> qas,
    ArrayView<const std::int64_t, 1> dates,
    const Input& input,
    std::int64_t statistics_end,
    const ModeRule& mode
)
{
    return mode.choose(qa_statistics(qas, dates, input, statistics_end));
}

// Step 3. The pixel's (B, T) as contiguous scalar_t with the procedure's
// transform applied: read in place when it already is that and there is no
// transform, otherwise converted into `buffer`. The caller's buffer is never
// written.
template<typename Value>
ArrayView<const scalar_t, 2> pixel_values(
    ArrayView<const Value, 2> spectral,
    const std::optional<preprocess::Affine>& transform,
    std::vector<scalar_t>& buffer
)
{
    const index_t B = spectral.extent(0);
    const index_t T = spectral.extent(1);

    if constexpr (std::is_same_v<Value, scalar_t>)
    {
        if (!transform && spectral.is_contiguous())
            return spectral;
    }

    buffer.resize(B * T);

    for (index_t b = 0; b < B; ++b)
    {
        for (index_t t = 0; t < T; ++t)
        {
            buffer[b * T + t] =
                static_cast<scalar_t>(spectral(b, t));
        }
    }

    if (transform)
    {
        const preprocess::Affine& affine = *transform;
        const index_t band = affine.band;
        for (index_t t = 0; t < T; ++t) {
            buffer[band * T + t] = preprocess::apply(affine, buffer[band * T + t]);
        }
    }

    return ArrayView<const scalar_t, 2>::contiguous(buffer.data(), {B, T});
}

// Whether a stored record's drops can be replayed on this input: a stored
// drop on a date the mask keeps twice (a plan without OnePerDate) does not say
// which of the two it was, so such a pixel is rerun.
bool drops_unambiguous(
    const std::vector<std::int64_t>& days,
    ArrayView<const std::int64_t, 1> dates,
    const ProcessingMask& mask
)
{
    bool unambiguous = true;
    std::size_t next = 0;
    index_t kept_on_day = 0;
    for (index_t t = 0; t < dates.size() && next < days.size(); ++t)
    {
        while (next < days.size() && days[next] < dates(t))
        {
            unambiguous = unambiguous && kept_on_day <= 1;
            kept_on_day = 0;
            ++next;
        }
        if (next < days.size() && dates(t) == days[next] && mask[t])
            ++kept_on_day;
    }
    return unambiguous && kept_on_day <= 1;
}

// Step 6. The record for the next resume: where the Standard machine stopped
// and the days it dropped from `filtered` (the chain's mask).
void fill_record(
    PixelRecord& record,
    PixelProcedure procedure,
    bool plain_mask,
    const standard::Detector& detector,
    const ProcessingMask& filtered,
    const FitResult& result,
    ArrayView<const std::int64_t, 1> dates
)
{
    record.standard = standard::Suspended{};
    record.excluded_days.clear();
    // A run the record cannot continue (a mode's extra mask, or a route the
    // suspension does not keep) stores none: a resume reruns it.
    if (procedure == PixelProcedure::Standard && plain_mask &&
        detector.suspended().exit != standard::Exit::None)
    {
        record.standard = detector.suspended();
        for (index_t t = 0; t < filtered.size(); ++t)
            if (filtered[t] && !result.mask[t])
                record.excluded_days.push_back(dates(t));
    }
}

} // namespace

//==============================================================================
// Run
//==============================================================================

Run::Run(
    const RunSpec& spec,
    ArrayView<const std::int64_t, 1> dates,
    index_t bands,
    index_t timesteps,
    index_t qa_timesteps,
    bool raw_qa,
    const Resume* resume
)
    : spec_(spec), dates_(dates), bands_(bands), timesteps_(timesteps), raw_qa_(raw_qa),
      resume_(resume)
{
    check_shapes(spec_.input, bands, timesteps, dates, qa_timesteps);
    check_qa(raw_qa, spec_.input.qa_table.get());

    if (!spec_.algorithm)
        throw std::invalid_argument("the run has no algorithm");
    resolved_ = spec_.algorithm->for_input(spec_.input);


    if (spec_.record || resume)
    {
        header_.digest = resolved_->digest;
        header_.series_end = dates.size() > 0
            ? dates(dates.size() - 1)
            : std::numeric_limits<std::int64_t>::min();
        header_.statistics_end = header_.series_end;
    }

    if (resume)
    {
        check_resume(resume->header, header_.digest, dates);
        header_.statistics_end = resume->header.statistics_end;
        // A resumed run reruns what it does not continue under the stored
        // run's statistics cutoff.
        shared_ = capped(resolved_->shared, header_.statistics_end);
        run_plan_ = capped(resolved_->plan, header_.statistics_end);
    }
    else
    {
        shared_ = resolved_->shared;
        run_plan_ = resolved_->plan;
    }
}

Observe Run::observed(index_t at) const
{
    if (!spec_.observer)
        return Observe{};
    const auto& list = spec_.observe;
    if (!list.empty() && std::find(list.begin(), list.end(), at) == list.end())
        return Observe{};
    return Observe{spec_.observer.get(), at};
}

// Steps 1 to 4. The mask is the procedure's chain over the pixel, then the
// mode's extra mask steps.
template<typename Value, typename Qa>
PreparedPixel Run::prepare(
    ArrayView<const Value, 2> spectral,
    ArrayView<const Qa, 1> qa,
    Worker& worker
) const
{
    const auto qas = decode_qa<Qa>(qa, spec_.input.qa_table.get(), worker.qas);
    PixelMode mode = choose_procedure(qas, dates_, spec_.input, shared_.statistics_end,
                                      *resolved_->strategies.mode);

    const preprocess::ProcedurePlan& steps = run_plan_.of(mode.procedure);
    const auto values = pixel_values<Value>(spectral, steps.transform, worker.spectral);
    PreparedPixel pixel{std::move(mode), PixelSeries(dates_, values, qas), ProcessingMask()};
    const PixelSeries& series = pixel.series;
    LOG_DEBUG("initial spectral: " << series.spectral());

    preprocess::apply(steps.chain, series.dates(), series.spectral(), series.qas(), pixel.mask);
    if (!pixel.mode.extra_mask.empty())
        preprocess::apply_steps(pixel.mode.extra_mask, series.dates(), series.spectral(),
                                series.qas(), pixel.mask);
    return pixel;
}

// Step 5 and, with RunSpec::record, step 6.
PixelOutput Run::fresh(PreparedPixel& pixel, Worker& worker) const
{
    PixelOutput out;
    standard::Detector& detector = worker.standard;
    const Strategies& strategies = resolved_->strategies;

    if (!spec_.record)
    {
        out.result = detector.fit(pixel.mode, pixel.series, std::move(pixel.mask), shared_,
                                  strategies, worker.observe);
        return out;
    }

    const ProcessingMask filtered = pixel.mask;
    out.result = detector.fit(pixel.mode, pixel.series, std::move(pixel.mask), shared_,
                              strategies, worker.observe);
    fill_record(out.record, pixel.mode.procedure, pixel.mode.extra_mask.empty(), detector,
                filtered, out.result, dates_);
    return out;
}

// Steps 5 and 6 of a resume: continued from the record when the input up to
// the stored series end is the stored run's and the machine stopped where it
// can continue; rerun under the capped cutoffs otherwise.
PixelOutput Run::resumed(
    PreparedPixel& pixel,
    Worker& worker,
    index_t at,
    std::uint64_t fingerprint
) const
{
    const PixelRecord& from = resume_->records[at];
    const std::vector<ChangeModel>& stored = resume_->segments[at];
    const PixelMode& mode = pixel.mode;

    PixelOutput out;
    out.status = fingerprint == from.input_fingerprint
        ? ResumeStatus::Resumed
        : ResumeStatus::RerunInputChanged;

    // A pixel its mode filters or caps is fitted again, as a whole-series
    // one is; so is one whose stored drops this mask cannot replay. Then its
    // result is a whole rerun, without the stored segments.
    const bool continues =
        out.status == ResumeStatus::Resumed &&
        from.procedure == PixelProcedure::Standard &&
        resumable(from.standard.exit) &&
        mode.procedure == PixelProcedure::Standard &&
        mode.extra_mask.empty() && mode.max_coefficients == 0 &&
        drops_unambiguous(from.excluded_days, dates_, pixel.mask);

    const index_t carried = from.standard.final_segments;
    const bool carried_change = carried > 0 && stored[carried - 1].change_probability == 1.0;

    standard::Detector& detector = worker.standard;
    const Strategies& strategies = resolved_->strategies;
    const ProcessingMask filtered = pixel.mask;
    out.result = continues
        ? detector.resume(pixel.series, std::move(pixel.mask), shared_, strategies,
                          from.standard, from.excluded_days, worker.observe, carried_change)
        : detector.fit(mode, pixel.series, std::move(pixel.mask), shared_, strategies,
                       worker.observe);
    fill_record(out.record, mode.procedure, mode.extra_mask.empty(), detector, filtered,
                out.result, dates_);

    if (continues)
    {
        out.result.models.insert(
            out.result.models.begin(),
            stored.begin(),
            stored.begin() + carried);
    }
    return out;
}

template<typename Value, typename Qa>
PixelOutput Run::pixel(
    ArrayView<const Value, 2> spectral,
    ArrayView<const Qa, 1> qa,
    index_t at,
    Worker& worker
) const
{
    if (spectral.extent(0) != bands_ || spectral.extent(1) != timesteps_ ||
        qa.size() != timesteps_)
        throw std::invalid_argument(
            "the pixel has " + std::to_string(spectral.extent(0)) + " bands, " +
            std::to_string(spectral.extent(1)) + " timesteps and " + std::to_string(qa.size()) +
            " QA values; the run has " + std::to_string(bands_) + " bands and " +
            std::to_string(timesteps_) + " timesteps");
    if (std::is_same_v<Qa, std::uint16_t> && !raw_qa_)
        throw std::invalid_argument("the pixel's QA is raw (uint16); the run was built for codes");
    if (resume_)
        check_stored(resume_->records[at], resume_->segments[at]);

    PreparedPixel prepared = prepare<Value, Qa>(spectral, qa, worker);

    // The pixel's input up to `through`, as a record holds it.
    const auto fingerprint = [&](std::int64_t through)
    {
        return detail::fingerprint_of<Value>(dates_, spectral, prepared.series.qas(),
                                             spec_.input.qa_fill, through);
    };

    PixelOutput out = resume_
        ? resumed(prepared, worker, at, fingerprint(resume_->header.series_end))
        : fresh(prepared, worker);
    if (spec_.record || resume_)
    {
        out.record.input_fingerprint = fingerprint(header_.series_end);
        out.record.procedure = prepared.mode.procedure;
    }
    if (!spec_.masks)
        out.result.mask = ProcessingMask();
    return out;
}

// Every supported pair of element types.
template PixelOutput Run::pixel<double, std::uint8_t>(
    ArrayView<const double, 2>, ArrayView<const std::uint8_t, 1>, index_t, Worker&) const;
template PixelOutput Run::pixel<double, std::uint16_t>(
    ArrayView<const double, 2>, ArrayView<const std::uint16_t, 1>, index_t, Worker&) const;
template PixelOutput Run::pixel<float, std::uint8_t>(
    ArrayView<const float, 2>, ArrayView<const std::uint8_t, 1>, index_t, Worker&) const;
template PixelOutput Run::pixel<float, std::uint16_t>(
    ArrayView<const float, 2>, ArrayView<const std::uint16_t, 1>, index_t, Worker&) const;
template PixelOutput Run::pixel<std::uint16_t, std::uint8_t>(
    ArrayView<const std::uint16_t, 2>, ArrayView<const std::uint8_t, 1>, index_t,
    Worker&) const;
template PixelOutput Run::pixel<std::uint16_t, std::uint16_t>(
    ArrayView<const std::uint16_t, 2>, ArrayView<const std::uint16_t, 1>, index_t,
    Worker&) const;

//==============================================================================
// Entry points
//==============================================================================

PixelOutput detect(
    const RunSpec& spec,
    ArrayView<const std::int64_t, 1> dates,
    const PixelSpectral& spectral,
    const PixelQa& qas,
    const Resume* resume
)
{
    const auto [bands, timesteps] = std::visit(
        [](auto s) { return std::pair<index_t, index_t>(s.extent(0), s.extent(1)); }, spectral);
    const index_t qa_timesteps = std::visit([](auto q) { return q.size(); }, qas);
    const bool raw_qa = std::holds_alternative<ArrayView<const std::uint16_t, 1>>(qas);

    const Run run(spec, dates, bands, timesteps, qa_timesteps, raw_qa, resume);

    if (resume && (resume->records.size() != 1 || resume->segments.size() != 1))
        throw std::invalid_argument(
            "the resume holds " + std::to_string(resume->records.size()) + " records and " +
            std::to_string(resume->segments.size()) + " segment lists; detect resumes one pixel");

    Worker worker;
    worker.observe = Observe{spec.observer.get(), 0};
    PixelOutput out = std::visit(
        [&](auto s, auto q)
        {
            using Value = std::remove_const_t<typename decltype(s)::value_type>;
            using Qa = std::remove_const_t<typename decltype(q)::value_type>;
            return run.pixel<Value, Qa>(s, q, 0, worker);
        },
        spectral, qas);

    out.header = run.header();
    return out;
}


// The cube loop for one pair of element types.
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

    // With RunSpec::masks, each pixel's mask goes straight into its T bytes
    // of the result's (H, W, T) array, and the pixel keeps none.
    CubeOutput merged;
    merged.height = H;
    merged.width = W;
    merged.timesteps = T;
    const bool masks = run.spec().masks;
    if (masks)
        merged.masks.assign(static_cast<std::size_t>(npixels) * T, std::uint8_t{0});

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

            auto spect_view =
                spectral.slice(fixed(row), fixed(col), all(), all()); // B, T

            auto qa_view =
                qas.slice(fixed(row), fixed(col), all()); // T

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

                worker.observe = run.observed(at);
                PixelOutput outcome =
                    run.pixel<Value, Qa>(spect_view, qa_view, static_cast<index_t>(at), worker);

                if (masks)
                {
                    const std::vector<std::uint8_t>& bits = outcome.result.mask.data();
                    if (bits.size() != T)
                        throw std::logic_error("the pixel's processing mask is not T long");
                    std::copy(bits.begin(), bits.end(), merged.masks.begin() + at * T);
                    outcome.result.mask = ProcessingMask();
                }

                by_pixel[at] = CubePixel{row, col, std::move(outcome.result),
                                           std::move(outcome.record), outcome.status};

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

    //----------------------------------------------------------------------
    // Collect, in raster order
    //----------------------------------------------------------------------
    merged.threads = used;

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

    merged.header = run.header();
    return merged;
}



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
                throw std::runtime_error("qas shape must be (height, width, timesteps)");

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
