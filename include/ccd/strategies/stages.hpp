#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "ccd/array_view.hpp"
#include "ccd/band_list.hpp"
#include "ccd/pmask.hpp"
#include "ccd/types.hpp"

#include "ccd/procedure/fit.hpp"
#include "ccd/strategies/capabilities.hpp"
#include "ccd/strategies/per_thread.hpp"

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
//   start_search     where the search for a stable start window begins, and
//                    when it has no candidate left
//   window_screen    the outliers of a candidate start window, dropped before
//                    it is fitted
//   stability_fit    the fit Initialize judges a candidate start window on,
//                    which Lookback tests against
//   stability_rule   whether that candidate is stable
//   model            the fit Extend tests each peek against
//   refit            when Extend refits it
//   change_test      the pixel's peek, whether a peek is a change, and the
//                    segment's break and magnitudes
//   outlier          whether the first peek observation is an outlier
//
// At an Extend peek the stages read one Peek, and the change test's results
// (PeekResult) are what the outlier rule reads; which stage provides what
// each needs is declared (Capabilities, strategies/capabilities.hpp) and
// checked when the strategies are built.
//
// What a strategy changes from one call to the next (a model's fits, a
// test's buffers) is not in the strategy: it makes one per thread
// (make_model(), make_scratch()), which the thread keeps in a PerThread
// (strategies/per_thread.hpp).
//
//==============================================================================

//------------------------------------------------------------------------------
// start_search: the search for a stable start window. Its decisions, in the
// order the machine asks them:
//
//   too_few(n)        a series of n observations has no start: the pixel
//                     ends without segments
//   first(at)         the first candidate window at observation `at` (the
//                     series start, or a segment's end)
//   room(w, n)        whether candidate w fits in n observations: Initialize
//                     may begin; otherwise the series ends
//   exhausted(w, n)   at an Initialize step, whether no candidate is left
//   short_span(d)     whether a candidate with dates d spans too few days:
//                     it grows
//   usable(d)         whether a candidate shortened by the window screen's
//                     drops, with dates d, is still one; otherwise it grows
//                     with nothing dropped
//   min_window()      the fewest observations a candidate holds
//------------------------------------------------------------------------------
struct StartSearch
{
    static constexpr const char* stage = "start_search";

    virtual ~StartSearch() = default;
    virtual bool too_few(index_t observations) const = 0;
    virtual Window first(index_t at) const = 0;
    virtual bool room(const Window& window, index_t observations) const = 0;
    virtual bool exhausted(const Window& window, index_t observations) const = 0;
    virtual bool short_span(ArrayView<const std::int64_t, 1> dates) const = 0;
    virtual bool usable(ArrayView<const std::int64_t, 1> dates) const = 0;
    virtual index_t min_window() const = 0;
};

//------------------------------------------------------------------------------
// window_screen: the observations of a candidate start window that are
// outliers, set in the returned mask (one entry per window observation).
// Initialize drops them before fitting the window when what is left is still
// long enough, and grows the window when every observation is one.
//
//   dates, spectral   the window's observations (spectral is (B, n))
//   variogram         the pixel's noise floor
//------------------------------------------------------------------------------
struct WindowScreen
{
    static constexpr const char* stage = "window_screen";

    virtual ~WindowScreen() = default;
    virtual ProcessingMask screen(ArrayView<const std::int64_t, 1> dates,
                                  ArrayView<const scalar_t, 2> spectral,
                                  const Variogram& variogram) const = 0;
};

//------------------------------------------------------------------------------
// stability_fit: the fit of a candidate start window's detection bands
// (strategies/model.hpp), which the stability rule judges and, once the
// window is stable, Lookback tests against. make_model() makes one thread's
// Model, which the thread keeps across pixels.
//------------------------------------------------------------------------------
struct StabilityFit
{
    static constexpr const char* stage = "stability_fit";

    virtual ~StabilityFit() = default;
    virtual Capabilities provides() const = 0;
    virtual std::unique_ptr<Model> make_model() const = 0;
};

//------------------------------------------------------------------------------
// stability_rule: whether a candidate start window, fitted by the stability
// fit, is stable. prepare() is called once per pixel with the change test's
// peek size and the detection band count, and keeps what it derives (pyccd's
// threshold) in the thread's scratch; stable() judges one candidate:
//
//   fit       the stability fit of the window's detection bands
//   dates     the window's dates
//------------------------------------------------------------------------------
struct StabilityRule
{
    static constexpr const char* stage = "stability_rule";

    virtual ~StabilityRule() = default;
    virtual Capabilities needs() const = 0;
    virtual std::unique_ptr<Scratch> make_scratch() const = 0;
    virtual void prepare(index_t peek, index_t detection_bands, Scratch& scratch) const = 0;
    virtual bool stable(const Model& fit, ArrayView<const std::int64_t, 1> dates,
                        const Variogram& variogram, const BandList& bands,
                        const Scratch& scratch) const = 0;
};

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
    virtual Capabilities provides() const = 0;
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
// peek, writing everything it provides into the PeekResult on every call,
// which the outlier rule reads after it. break_at() and magnitude() are the segment's break
// observation and each band's change magnitude, from the segment's last
// test.
//------------------------------------------------------------------------------

enum class TestOutcome : std::uint8_t
{
    Change,
    NoChange
};

// One Extend peek, as every stage that tests it reads it. Windows are
// positions in the masked series, whose dates are `dates`.
struct Peek
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

// What the change test found at a peek, for the outlier rule (by
// Capability: ComparisonRmse, PeekStatistic). Rewritten by every test.
struct PeekResult
{
    std::vector<scalar_t> comparison_rmse;   // by position in the detection bands
    std::vector<scalar_t> statistic;         // by peek observation
};

struct ChangeTest
{
    static constexpr const char* stage = "change_test";

    virtual ~ChangeTest() = default;
    virtual Capabilities needs() const = 0;
    virtual Capabilities provides() const = 0;
    virtual std::unique_ptr<Scratch> make_scratch() const = 0;
    virtual index_t prepare(ArrayView<const std::int64_t, 1> statistics_dates,
                            index_t detection_bands, Scratch& scratch) const = 0;
    virtual TestOutcome test(const Peek& peek, Scratch& scratch,
                             PeekResult& result) const = 0;
    virtual index_t break_at(const Peek& peek) const = 0;
    virtual scalar_t magnitude(const Peek& peek, index_t band) const = 0;
};

//------------------------------------------------------------------------------
// outlier: after a peek that is not a change, whether its first observation
// (at the model window's stop) is an outlier, which Extend drops.
//------------------------------------------------------------------------------

enum class OutlierOutcome : std::uint8_t
{
    Keep,
    Drop
};

struct OutlierRule
{
    static constexpr const char* stage = "outlier";

    virtual ~OutlierRule() = default;
    virtual Capabilities needs() const = 0;
    virtual OutlierOutcome judge(const Peek& peek, const PeekResult& result) const = 0;
};

// One call's strategies, one per stage, built from its configuration.
struct Strategies
{
    std::shared_ptr<const StartSearch> start_search;
    std::shared_ptr<const WindowScreen> window_screen;
    std::shared_ptr<const StabilityFit> stability_fit;
    std::shared_ptr<const StabilityRule> stability_rule;
    std::shared_ptr<const MonitoringModel> model;
    std::shared_ptr<const RefitSchedule> refit;
    std::shared_ptr<const ChangeTest> change_test;
    std::shared_ptr<const OutlierRule> outlier;
};

} // namespace ccd
