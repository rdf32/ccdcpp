#include "resume.hpp"

#include <algorithm>
#include <cstdio>
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
// check_stored()) and a configuration whose statistics cutoff it would change
// unless told to (check_statistics()), and reruns what it does not continue
// under the stored run's statistics cutoff (capped()). Run::resumed() continues each pixel its record
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

// An ordinal day (0001-01-01 is 1) as "YYYY-MM-DD" (H. Hinnant's
// civil_from_days, from 1970-01-01's ordinal, 719163); a day outside years 1
// to 9999 as its ordinal.
std::string iso_date(std::int64_t ordinal)
{
    if (ordinal < 1 || ordinal > 3652059)
        return std::to_string(ordinal);
    const std::int64_t z = ordinal - 719163 + 719468;
    const std::int64_t era = z / 146097;
    const std::int64_t doe = z - era * 146097;
    const std::int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const std::int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const std::int64_t mp = (5 * doy + 2) / 153;
    const std::int64_t d = doy - (153 * mp + 2) / 5 + 1;
    const std::int64_t m = mp < 10 ? mp + 3 : mp - 9;
    const std::int64_t y = yoe + era * 400 + (m <= 2 ? 1 : 0);
    char text[16];
    std::snprintf(text, sizeof text, "%04lld-%02lld-%02lld", static_cast<long long>(y),
                  static_cast<long long>(m), static_cast<long long>(d));
    return text;
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

void check_statistics(const config::Shared& shared, std::int64_t end)
{
    if (shared.statistics_end <= end)
        return;
    const std::string statistics = shared.statistics_end == config::NO_STATISTICS_END
        ? "cover the whole series"
        : "run to " + iso_date(shared.statistics_end);
    throw std::invalid_argument(
        "this configuration's statistics " + statistics +
        "; a resume keeps them at the stored run's end (" + iso_date(end) +
        "). Pass freeze_statistics=True to resume that way, or rerun in full to use "
        "the new dates in the statistics");
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
    std::vector<std::uint8_t>* reasons = spec_.reasons ? &pixel.reasons : nullptr;
    out.result = continues
        ? detector.resume(pixel.series, std::move(pixel.mask), shared_, strategies,
                          from.standard, from.excluded_days, worker.observe, carried_change,
                          reasons)
        : detector.fit(mode, pixel.series, std::move(pixel.mask), shared_, strategies,
                       worker.observe, reasons);
    out.reasons = std::move(pixel.reasons);
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
