#pragma once

// Internal: the run's side of a resume (run/resume.cpp), which Run reads.

#include <cstdint>
#include <vector>

#include "ccd/config/shared.hpp"
#include "ccd/core/array_view.hpp"
#include "ccd/input/mask/chain.hpp"
#include "ccd/input/processing_mask.hpp"
#include "ccd/results/record.hpp"
#include "ccd/results/segment.hpp"
#include "../machine/detector.hpp"

namespace ccd
{

// A resume's refusals: another record format or results version, another
// input (its QA table included) or configuration, a series that ends
// before the record's.
void check_resume(
    const ResumeHeader& from,
    std::uint64_t digest,
    ArrayView<const std::int64_t, 1> dates
);

// A resumed pixel's refusal: fewer stored segments than its record's final
// ones.
void check_stored(const PixelRecord& from, const std::vector<ChangeModel>& stored);

// A resume's refusal of a statistics cutoff it would change: the
// configuration's statistics_end (which its GreenMedian steps take) after
// `end`, the stored run's statistics end. Run calls it unless the caller
// froze the statistics there (Resume::freeze_statistics).
void check_statistics(const config::Shared& shared, std::int64_t end);

// The cutoffs of a run resumed from one whose series ended at `end`:
// statistics_end capped at `end`,
config::Shared capped(const config::Shared& shared, std::int64_t end);

// and every GreenMedian step's date.
preprocess::PreprocessPlan capped(const preprocess::PreprocessPlan& plan, std::int64_t end);

// Step 6 of a pixel (run_pixel.hpp), the record for the next resume: where
// the machine stopped and the days it dropped from `filtered` (the
// chain's mask).
void fill_record(
    PixelRecord& record,
    PixelProcedure procedure,
    bool plain_mask,
    const machine::Detector& detector,
    const ProcessingMask& filtered,
    const FitResult& result,
    ArrayView<const std::int64_t, 1> dates
);

} // namespace ccd
