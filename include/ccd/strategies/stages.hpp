#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "ccd/core/array_view.hpp"
#include "ccd/core/band_list.hpp"
#include "ccd/input/pmask.hpp"
#include "ccd/core/types.hpp"

#include "ccd/input/pixel_series.hpp"
#include "ccd/input/quality.hpp"
#include "ccd/machine/fit.hpp"
#include "ccd/machine/pixel_procedure.hpp"
#include "ccd/strategies/capabilities.hpp"
#include "ccd/strategies/coefficients.hpp"
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
//   mode             the pixel's procedure: change detection, or one curve
//                    over the whole series
//   noise_floor      each band's noise, which every test scales by
//   start_search     where the search for a stable start window begins, and
//                    when it has no candidate left
//   window_screen    the outliers of a candidate start window, dropped before
//                    it is fitted
//   stability_fit    the fit Initialize judges a candidate start window on,
//                    which Lookback tests against
//   stability_rule   whether that candidate is stable
//   lookback         whether the observation before a stable start window
//                    joins it, is dropped, or is a change
//   model            the fit Extend tests each peek against
//   refit            when Extend refits it
//   change_test      the pixel's peek, whether a peek is a change, and the
//                    segment's break and magnitudes
//   outlier          whether the first peek observation is an outlier
//   segment_fit      the coefficients a segment Extend grew reports
//   edge_fit         the fits of the Start and End segments and of the
//                    whole-series curve
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
// mode: the pixel's procedure, from the QA statistics of its statistics
// window: Standard (the change-detection machine), or PermanentSnow or
// InsufficientClear (one curve over the whole series, the edge fit's); with
// what it asks of a Standard pixel's detection (PixelMode,
// machine/pixel_procedure.hpp).
//------------------------------------------------------------------------------
struct ModeRule
{
    static constexpr const char* stage = "mode";

    virtual ~ModeRule() = default;
    virtual PixelMode choose(const Quality::Statistics& statistics) const = 0;
};

//------------------------------------------------------------------------------
// noise_floor: each band's noise floor, from the pixel's statistics-window
// observations (dates, spectral (B, n)): one value per spectral row. A NaN in
// it ends the pixel without segments.
//------------------------------------------------------------------------------
struct NoiseFloor
{
    static constexpr const char* stage = "noise_floor";

    virtual ~NoiseFloor() = default;
    virtual std::vector<scalar_t> estimate(ArrayView<const std::int64_t, 1> dates,
                                           ArrayView<const scalar_t, 2> spectral) const = 0;
};

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
//   begin(d, at, e)   a search from observation `at` of the masked series
//                     (dates d; `at` is 0 at the pixel's first search), e
//                     the input's last day: its SearchPlan, the last day it
//                     sees (the largest: all of them), what it does when
//                     exhausted: End (the segment loop ends), Restore (it
//                     ends with the observations the search's window screen
//                     dropped back in the series) or Force (forced()), and
//                     whether a resume on a longer input may continue it
//                     (false when the plan reads the input's end)
//   exhausted(w, n)   at an Initialize step, whether no candidate is left
//                     among the n observations dated through the last day
//   forced(d, at, l)  the forced start of an exhausted search from `at`, l
//                     its last day
//   short_span(d)     whether a candidate with dates d spans too few days:
//                     it grows
//   usable(d)         whether a candidate shortened by the window screen's
//                     drops, with dates d, is still one; otherwise it grows
//                     with nothing dropped
//   min_window()      the fewest observations a candidate holds
//------------------------------------------------------------------------------
enum class Exhaustion : std::uint8_t
{
    End,
    Restore,
    Force
};

struct SearchPlan
{
    std::int64_t last_day = std::numeric_limits<std::int64_t>::max();
    Exhaustion exhausted = Exhaustion::End;
    bool resumable = true;
};

struct StartSearch
{
    static constexpr const char* stage = "start_search";

    virtual ~StartSearch() = default;
    virtual bool too_few(index_t observations) const = 0;
    virtual Window first(index_t at) const = 0;
    virtual bool room(const Window& window, index_t observations) const = 0;
    virtual SearchPlan begin(ArrayView<const std::int64_t, 1> dates, index_t at,
                             std::int64_t input_end) const = 0;
    virtual bool exhausted(const Window& window, index_t observations) const = 0;
    virtual Window forced(ArrayView<const std::int64_t, 1> dates, index_t at,
                          std::int64_t last_day) const = 0;
    virtual bool short_span(ArrayView<const std::int64_t, 1> dates) const = 0;
    virtual bool usable(ArrayView<const std::int64_t, 1> dates) const = 0;
    virtual index_t min_window() const = 0;
};

// The change test's plan for a pixel (ChangeTest::prepare(), below), which the
// stability rule and Lookback are prepared with too.
struct PeekPlan
{
    index_t peek;
    bool full = true;
    index_t min_window = 0;
    index_t reference = 0;   // the peek the test's threshold is for (0: peek)
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
    // Its fits' coefficients: a candidate is fitted with min, in a basis max
    // wide (strategies/coefficients.hpp).
    virtual CoefficientRule coefficients() const = 0;
};

//------------------------------------------------------------------------------
// stability_rule: whether a candidate start window, fitted by the stability
// fit, is stable. prepare() is called once per pixel with the change test's
// PeekPlan and the detection band count, and keeps what it derives (pyccd's
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
    virtual void prepare(const PeekPlan& peek, index_t detection_bands,
                         Scratch& scratch) const = 0;
    virtual bool stable(const Model& fit, ArrayView<const std::int64_t, 1> dates,
                        const Variogram& variogram, const BandList& bands,
                        const Scratch& scratch) const = 0;
};

//------------------------------------------------------------------------------
// lookback: one Lookback step, after a stable start: tests the observations
// before the window against the stability fit and judges the one just before
// it. prepare() is called once per pixel with the change test's PeekPlan
// and the detection band count, and keeps what it derives in the thread's
// scratch. judge() reads:
//
//   fit                the stability fit of the start window
//   dates, spectral    the masked series (spectral is (B, n))
//   window, prev_break the start window, which starts after prev_break
//
// and returns its verdict with the observation it is about (a position in
// the masked series): Change (the window stays), Drop (that observation
// leaves the series) or Include (the window starts there).
//------------------------------------------------------------------------------

enum class LookbackVerdict : std::uint8_t
{
    Change,
    Drop,
    Include
};

struct LookbackJudgement
{
    LookbackVerdict verdict;
    index_t observation;   // for Drop and Include
};

struct Lookback
{
    static constexpr const char* stage = "lookback";

    virtual ~Lookback() = default;
    virtual Capabilities needs() const = 0;
    virtual std::unique_ptr<Scratch> make_scratch() const = 0;
    virtual void prepare(const PeekPlan& peek, index_t detection_bands,
                         Scratch& scratch) const = 0;
    virtual LookbackJudgement judge(Model& fit, ArrayView<const std::int64_t, 1> dates,
                                    ArrayView<const scalar_t, 2> spectral, const Window& window,
                                    index_t prev_break, const Variogram& variogram,
                                    const BandList& bands, Scratch& scratch) const = 0;
};

//------------------------------------------------------------------------------
// refit: whether Extend refits the model before this peek test. The first
// test after a start always fits; this decides the tests after it.
//
//   window_size   observations in the model window now
//   window_span   days the model window spans now
//   fit_size      observations it held when the model was last fitted
//   fit_span      days it spanned then
//------------------------------------------------------------------------------
struct RefitSchedule
{
    static constexpr const char* stage = "refit";

    virtual ~RefitSchedule() = default;
    virtual bool due(index_t window_size, scalar_t window_span, index_t fit_size,
                     scalar_t fit_span) const = 0;
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
    // The coefficients it fits a window with, by the window's size, and the
    // widest (strategies/coefficients.hpp); read once per pixel.
    virtual CoefficientRule coefficients() const = 0;
};

//------------------------------------------------------------------------------
// change_test: the pixel's peek, whether a peek is a change, and what the
// segment a change ends reports.
//
// prepare() is called once per pixel, with its statistics-window dates and
// its detection band count: it returns the pixel's PeekPlan, and keeps what
// it derives for the pixel (chi2's threshold) in the thread's scratch. The
// plan's peek is the peek size, which the machine's guards, the stability
// rule and Lookback also read (with `reference`, the peek its threshold is
// set for, from which a longer peek's is adjusted); `full` whether a test reads only full peeks: the segment loop ends
// where no full peek follows the window (chi2's). Otherwise the peek is
// shortened to the observations left, and a peek with none ends the series.
// A model window of fewer than min_window observations grows untested.
//
// test() decides one Extend peek, writing everything it provides into the
// PeekResult on every call, which the outlier rule reads after it:
// Undecided is a shortened peek that decides nothing, which ends the
// series. break_at() is a change's first observation, from the last test
// (the peek's observations before it leave the series once its segment is
// written), and magnitude() each band's change magnitude, from the
// segment's last test; at a series end with no observation left after the
// window the peek it reads is empty, and no test was made on that window.
//------------------------------------------------------------------------------

enum class TestOutcome : std::uint8_t
{
    Change,
    NoChange,
    Undecided
};

// One Extend peek, as every stage that tests it reads it. Windows are
// positions in the masked series, whose dates are `dates`.
struct Peek
{
    const Model& model;                                    // fitted on fit_window
    ArrayView<const std::int64_t, 1> dates;
    ArrayView<const scalar_t, 2> spectral;                 // (B, n)
    Window window;                                         // the model window
    Window fit_window;                                     // the last refit's
    index_t fit;                                           // that refit's number: a new one per refit
    Window peek;
    const std::vector<std::vector<scalar_t>>& residuals;   // by band, over the peek
    const Variogram& variogram;
    const BandList& bands;                                 // the detection bands
};

// A per-observation statistic of a peek: the name the change test gives it
// (chi2's "change_magnitude") and an outlier rule selects it by, and its
// values by peek observation.
struct PeekStatistic
{
    std::string name;
    std::vector<scalar_t> values;
};

// What the change test found at a peek, for the outlier rule: the comparison
// RMSE (Capability::ComparisonRmse) and the statistics it provides
// (ChangeTest::statistics()). Rewritten by every test.
struct PeekResult
{
    std::vector<scalar_t> comparison_rmse;   // by position in the detection bands
    std::vector<PeekStatistic> statistics;   // the change test's, by name

    // The values of statistic `name`, for a change test to write: its slot is
    // made on first use and kept, with its storage, after.
    std::vector<scalar_t>& provide(const std::string& name)
    {
        for (PeekStatistic& s : statistics)
            if (s.name == name)
                return s.values;
        statistics.push_back({name, {}});
        return statistics.back().values;
    }

    // The values of statistic `name`, or null when the change test provides
    // none of that name.
    const std::vector<scalar_t>* find(const std::string& name) const
    {
        for (const PeekStatistic& s : statistics)
            if (s.name == name)
                return &s.values;
        return nullptr;
    }
};

struct ChangeTest
{
    static constexpr const char* stage = "change_test";

    virtual ~ChangeTest() = default;
    virtual Capabilities needs() const = 0;
    virtual Capabilities provides() const = 0;
    // The names of the per-observation statistics test() writes into the
    // PeekResult (PeekResult::provide()).
    virtual std::vector<std::string> statistics() const { return {}; }
    virtual std::unique_ptr<Scratch> make_scratch() const = 0;
    virtual PeekPlan prepare(ArrayView<const std::int64_t, 1> statistics_dates,
                             index_t detection_bands, Scratch& scratch) const = 0;
    virtual TestOutcome test(const Peek& peek, Scratch& scratch,
                             PeekResult& result) const = 0;
    virtual index_t break_at(const Peek& peek, const Scratch& scratch) const = 0;
    virtual scalar_t magnitude(const Peek& peek, const Scratch& scratch,
                               index_t band) const = 0;
};

//------------------------------------------------------------------------------
// outlier: after a peek that is not a change, whether its first observation
// (at the model window's stop) is an outlier, which Extend drops; or
// EndSeries, a peek too short to tell, which ends the series.
//------------------------------------------------------------------------------

enum class OutlierOutcome : std::uint8_t
{
    Keep,
    Drop,
    EndSeries
};

struct OutlierRule
{
    static constexpr const char* stage = "outlier";

    virtual ~OutlierRule() = default;
    virtual Capabilities needs() const = 0;
    // The names of the change test's statistics judge() reads; the registry
    // refuses a configuration whose change test does not provide them.
    virtual std::vector<std::string> statistics() const { return {}; }
    virtual OutlierOutcome judge(const Peek& peek, const PeekResult& result) const = 0;
};

//------------------------------------------------------------------------------
// segment_fit: the coefficients of every band of a segment Extend grew, into
// `bands` (one per band; the change test sets the magnitudes after), and the
// coefficient count the segment reports (its curve_qa). segment() reads the
// monitoring model as its last refit left it, fitted with num_coef on the
// segment's window `window` of the masked series (`dates`, `spectral`
// (B, n)); unfitted() is the segment of a forced start the series ended
// before Extend fitted, `model` not fitted on it and num_coef the count of
// its size. The pixel's mode caps the coefficients at max_coefficients (0:
// no cap). Either may refit `model`: the segment is its last use.
//------------------------------------------------------------------------------
struct SegmentFit
{
    static constexpr const char* stage = "segment_fit";

    virtual ~SegmentFit() = default;
    virtual std::unique_ptr<Scratch> make_scratch() const = 0;
    virtual index_t segment(Model& model, ArrayView<const std::int64_t, 1> dates,
                            ArrayView<const scalar_t, 2> spectral, const Window& window,
                            index_t num_coef, index_t max_coefficients, Scratch& scratch,
                            std::vector<BandFit>& bands) const = 0;
    virtual index_t unfitted(Model& model, ArrayView<const std::int64_t, 1> dates,
                             ArrayView<const scalar_t, 2> spectral, const Window& window,
                             index_t num_coef, index_t max_coefficients, Scratch& scratch,
                             std::vector<BandFit>& bands) const = 0;
};

//------------------------------------------------------------------------------
// edge_fit: the fits that are not a test's. catch_window() is the segment
// catching `window` of the masked series (`dates`, `spectral` (B, n)): the
// Start segment before the first stable start, the End segment after the
// last break; whole_series() is a PermanentSnow or InsufficientClear pixel's
// one curve over the observations `mask` keeps (no segment when too few).
// Both have change probability 0 and magnitude 0.
//
// start_due() is whether the Start segment is written, `before` the
// observations before the first stable start; end_due() whether the End
// segment is, `after` the observations after the last segment, and
// after_change whether that segment ended in a change. `peek` is the
// pixel's peek size.
//------------------------------------------------------------------------------
struct EdgeFit
{
    static constexpr const char* stage = "edge_fit";

    virtual ~EdgeFit() = default;
    virtual std::unique_ptr<Scratch> make_scratch() const = 0;
    virtual bool start_due(index_t before, index_t peek) const = 0;
    virtual bool end_due(index_t after, index_t peek, bool after_change) const = 0;
    virtual ChangeModel catch_window(ArrayView<const std::int64_t, 1> dates,
                                     ArrayView<const scalar_t, 2> spectral, const Window& window,
                                     CurveQA curve_qa, Scratch& scratch) const = 0;
    virtual FitResult whole_series(PixelProcedure procedure, const PixelSeries& pixel,
                                   ProcessingMask mask, Scratch& scratch) const = 0;
};

// One call's strategies, one per stage, built from its configuration.
struct Strategies
{
    std::shared_ptr<const ModeRule> mode;
    std::shared_ptr<const NoiseFloor> noise_floor;
    std::shared_ptr<const StartSearch> start_search;
    std::shared_ptr<const WindowScreen> window_screen;
    std::shared_ptr<const StabilityFit> stability_fit;
    std::shared_ptr<const StabilityRule> stability_rule;
    std::shared_ptr<const Lookback> lookback;
    std::shared_ptr<const MonitoringModel> model;
    std::shared_ptr<const RefitSchedule> refit;
    std::shared_ptr<const ChangeTest> change_test;
    std::shared_ptr<const OutlierRule> outlier;
    std::shared_ptr<const SegmentFit> segment_fit;
    std::shared_ptr<const EdgeFit> edge_fit;
};

} // namespace ccd
