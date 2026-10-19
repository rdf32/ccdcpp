#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "ccd/array_view.hpp"
#include "ccd/band_list.hpp"
#include "ccd/types.hpp"

#include "ccd/procedure/fit.hpp"

namespace ccd
{

class Model;   // strategies/model.hpp

//==============================================================================
//
// The stages of the algorithm
//
// The machine (procedure/standard.hpp) is the CCDC family's fixed skeleton:
// its phases and routes. What each phase decides is a stage, decided by the
// strategy the run's configuration names for it (config/registry.hpp). A
// strategy returns its decision; only the machine routes.
//
// Each stage is an interface with its name in `stage`; its strategies live
// one per file under strategies/<stage>/. Strategy objects are immutable and
// shared by every thread of a call.
//
// The stages so far (the others are still decided inside the machine):
//
//   model         the fit Extend tests each peek against
//   refit         when Extend refits it
//   change_test   the pixel's peek and change threshold, and whether a peek
//                 is a change
//
// What a strategy changes from one call to the next (a model's fits, a
// test's buffers) is not in the strategy: it makes one per thread
// (make_model(), make_scratch()), which the thread keeps in a PerThread
// (strategies/per_thread.hpp).
//
//==============================================================================

//------------------------------------------------------------------------------
// refit: whether Extend refits the model before this peek test. The first
// test after a start always fits; this decides the tests after it.
//
//   window_size   observations in the model window now
//   window_span   days the model window spans now
//   fit_span      days it spanned when the model was last fitted
//------------------------------------------------------------------------------
struct RefitSchedule
{
    static constexpr const char* stage = "refit";

    virtual ~RefitSchedule() = default;
    virtual bool due(index_t window_size, scalar_t window_span, scalar_t fit_span) const = 0;
};

//------------------------------------------------------------------------------
// model: the monitoring model, one pixel's fits that Extend tests each peek
// against (strategies/model.hpp). make_model() makes one thread's Model,
// which the thread keeps across pixels.
//------------------------------------------------------------------------------
struct MonitoringModel
{
    static constexpr const char* stage = "model";

    virtual ~MonitoringModel() = default;
    virtual std::unique_ptr<Model> make_model() const = 0;
};

//------------------------------------------------------------------------------
// change_test: the pixel's peek, whether a peek is a change, and what the
// segment a change ends reports.
//
// prepare() is called once per pixel, with its statistics-window dates and
// its detection band count: it returns the peek size, which the machine's
// guards and Lookback also read, and keeps what it derives for the pixel
// (chi2's threshold) in the thread's scratch. test() decides one Extend
// peek, writing each peek observation's statistic into `magnitude`, which
// the outlier test reads after it. break_at() and magnitude() are the
// segment's break observation and each band's change magnitude, from the
// segment's last test.
//------------------------------------------------------------------------------

enum class TestOutcome : std::uint8_t
{
    Change,
    NoChange
};

// What a change test reads at one Extend step. Windows are positions in the
// masked series, whose dates are `dates`.
struct PeekTest
{
    const Model& model;                                    // fitted on fit_window
    ArrayView<const std::int64_t, 1> dates;
    Window window;                                         // the model window
    Window fit_window;                                     // the last refit's
    index_t fit;                                           // that refit's number: a new one per refit
    Window peek;
    const std::vector<std::vector<scalar_t>>& residuals;   // by band, over the peek
    const Variogram& variogram;
    const BandList& bands;                                 // the detection bands
};

// A change test's buffers and per-pixel values for one thread
// (make_scratch()).
class TestScratch
{
public:
    virtual ~TestScratch() = default;
};

struct ChangeTest
{
    static constexpr const char* stage = "change_test";

    virtual ~ChangeTest() = default;
    virtual std::unique_ptr<TestScratch> make_scratch() const = 0;
    virtual index_t prepare(ArrayView<const std::int64_t, 1> statistics_dates,
                            index_t detection_bands, TestScratch& scratch) const = 0;
    virtual TestOutcome test(const PeekTest& peek, TestScratch& scratch,
                             std::vector<scalar_t>& magnitude) const = 0;
    virtual index_t break_at(const PeekTest& peek) const = 0;
    virtual scalar_t magnitude(const PeekTest& peek, index_t band) const = 0;
};

// One call's strategies, one per stage, built from its configuration.
struct Strategies
{
    std::shared_ptr<const MonitoringModel> model;
    std::shared_ptr<const RefitSchedule> refit;
    std::shared_ptr<const ChangeTest> change_test;
};

} // namespace ccd
