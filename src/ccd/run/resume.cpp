#include "resume.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "run_pixel.hpp"

namespace ccd
{

//==============================================================================
//
// Resume, on the run's side
//
// A resumed call refuses a stored call it cannot continue (check_resume(),
// check_stored()) and reruns what it does not continue under the stored run's
// statistics cutoff (capped()). Run::resumed() continues each pixel its record
// lets it continue, and every run with a record stores where its machine
// stopped (fill_record()). The machine's side is Detector::resume()
// (machine/detector.cpp).
//
//==============================================================================

namespace
{

// The exits a resume continues the machine from; the others (the
// early exits) are rerun, as are the other procedures.
bool resumable(machine::Exit exit)
{
    using machine::Exit;
    return exit == Exit::NoRoomToInitialize || exit == Exit::InitializeExhausted ||
        exit == Exit::PeekBeyondEnd || exit == Exit::SeriesEnd;
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

} // namespace

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

void check_stored(const PixelRecord& from, const std::vector<ChangeModel>& stored)
{
    if (from.procedure == PixelProcedure::Standard &&
        stored.size() < from.standard.final_segments)
        throw std::invalid_argument(
            "the resume record has " + std::to_string(from.standard.final_segments) +
            " final segments; " + std::to_string(stored.size()) + " were given");
}

config::Shared capped(const config::Shared& shared, std::int64_t end)
{
    config::Shared out = shared;
    out.statistics_end = std::min(out.statistics_end, end);
    return out;
}

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

void fill_record(
    PixelRecord& record,
    PixelProcedure procedure,
    bool plain_mask,
    const machine::Detector& detector,
    const ProcessingMask& filtered,
    const FitResult& result,
    ArrayView<const std::int64_t, 1> dates
)
{
    record.standard = machine::Suspended{};
    record.excluded_days.clear();
    // A run the record cannot continue (a mode's extra mask, or a route the
    // suspension does not keep) stores none: a resume reruns it.
    if (procedure == PixelProcedure::Standard && plain_mask &&
        detector.suspended().exit != machine::Exit::None)
    {
        record.standard = detector.suspended();
        for (index_t t = 0; t < filtered.size(); ++t)
            if (filtered[t] && !result.mask[t])
                record.excluded_days.push_back(dates(t));
    }
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

    machine::Detector& detector = worker.detector;
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

} // namespace ccd
