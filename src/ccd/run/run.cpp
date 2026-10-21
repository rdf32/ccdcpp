#include "run_pixel.hpp"
#include "resume.hpp"

#include <algorithm>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "ccd/ccd.hpp"
#include "ccd/input/qa_statistics.hpp"
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

// Step 6, with RunSpec::diagnostics: the pixel's, its procedure's, of a
// result whose preprocessing kept `clear` observations, the machine's from
// `detector` (a Standard pixel's); its segments' work the detector's, after
// zeros for the segments it did not write (a resume's carried ones, or the
// whole-series one).
PixelDiagnostics diagnose(PixelProcedure procedure, index_t clear, const FitResult& result,
                          const machine::Detector& detector)
{
    PixelDiagnostics d;
    d.procedure = procedure;
    d.clear = clear;
    d.kept = result.mask.count();
    const bool standard = procedure == PixelProcedure::Standard;
    if (standard)
    {
        d.exit = detector.exit();
        d.peek = detector.peek();
    }
    const std::vector<SegmentDiagnostics> none;
    const std::vector<SegmentDiagnostics>& written = standard ? detector.segment_work() : none;
    assert(written.size() <= result.models.size());
    d.segments.assign(result.models.size() - written.size(), SegmentDiagnostics{});
    d.segments.insert(d.segments.end(), written.begin(), written.end());
    return d;
}

// Step 6, with RunSpec::reasons: the code of each kept observation of
// `result`, Reason::Kept inside a segment's days, Reason::NoFit after the
// last segment of a pixel whose machine ended in Exit::NoFit (`no_fit`),
// Reason::NoSegment otherwise. The segments are in date order.
void settle_reasons(FitResult& result, ArrayView<const std::int64_t, 1> dates, bool no_fit)
{
    const std::vector<ChangeModel>& models = result.models;
    const auto kept = static_cast<std::uint8_t>(Reason::Kept);
    std::size_t m = 0;
    for (index_t t = 0; t < dates.size(); ++t)
    {
        if (result.reasons[t] != kept)
            continue;
        const std::int64_t day = dates(t);
        while (m < models.size() && models[m].end_day < day)
            ++m;
        if (m < models.size() && models[m].start_day <= day)
            continue;
        result.reasons[t] = static_cast<std::uint8_t>(
            no_fit && m == models.size() ? Reason::NoFit : Reason::NoSegment);
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
    // Step 2. The pixel's procedure and mode, from its QA statistics up to
    // statistics_end.
    PixelMode mode = resolved_->strategies.mode->choose(
        qa_statistics(qas, dates_, spec_.input, shared_.statistics_end));

    const preprocess::ProcedurePlan& steps = run_plan_.of(mode.procedure);
    const auto values = pixel_values<Value>(spectral, steps.transform, worker.spectral);
    PreparedPixel pixel{std::move(mode), PixelSeries(dates_, values, qas), ProcessingMask(), {}};
    const PixelSeries& series = pixel.series;

    std::vector<std::uint8_t>* reasons = spec_.reasons ? &pixel.reasons : nullptr;
    preprocess::apply(steps.chain, series.dates(), series.spectral(), series.qas(), pixel.mask,
                      reasons);
    if (!pixel.mode.extra_mask.empty())
        preprocess::apply_steps(pixel.mode.extra_mask, series.dates(), series.spectral(),
                                series.qas(), pixel.mask, reasons);
    return pixel;
}

// Step 5 and, with RunSpec::record, step 6.
PixelOutput Run::fresh(PreparedPixel& pixel, Worker& worker) const
{
    PixelOutput out;
    machine::Detector& detector = worker.detector;

    // The record keeps the mask the chain built; the fit takes it.
    const ProcessingMask filtered = spec_.record ? pixel.mask : ProcessingMask();
    out.result = detector.fit(pixel.mode, pixel.series, std::move(pixel.mask), shared_,
                              resolved_->strategies, worker.observe, std::move(pixel.reasons));
    if (spec_.record)
        fill_record(out.record, pixel.mode.procedure, pixel.mode.extra_mask.empty(), detector,
                    filtered, out.result, dates_);
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
    const index_t clear = spec_.diagnostics ? prepared.mask.count() : 0;

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
    if (spec_.reasons)
        settle_reasons(out.result, dates_,
                       prepared.mode.procedure == PixelProcedure::Standard &&
                           worker.detector.exit() == machine::Exit::NoFit);
    if (spec_.diagnostics)
        out.result.diagnostics =
            diagnose(prepared.mode.procedure, clear, out.result, worker.detector);
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

} // namespace ccd
