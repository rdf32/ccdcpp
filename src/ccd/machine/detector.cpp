#include "detector.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>
#include <cassert>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

#include "ccd/core/maths.hpp"
#include "../core/logger.hpp"

namespace ccd
{

namespace machine
{

namespace
{

// The last day of the input, whatever its QA.
std::int64_t input_end(const Context& context)
{
    const auto dates = context.pixel.dates();
    return dates(dates.size() - 1);
}

} // namespace

namespace
{

#if CCD_OBSERVE
ObserverEvent day_event(ObserverEvent::Kind kind, std::int64_t day, std::string outcome)
{
    ObserverEvent e{kind};
    e.day = day;
    e.outcome = std::move(outcome);
    return e;
}
#endif

// Extend's last peek, as its tests read it.
Peek peek_of(const Context& context, const State& state)
{
    const ExtendState& extend = state.extend;
    return Peek{*extend.model, state.series.dates(), state.series.spectral(),
                    state.cursor.window,
                    extend.fit_window, extend.fits, extend.peek, extend.residuals,
                    state.noise, context.shared.detection_bands};
}

// Drops the observation at `position` for `why`, told to the observer, and
// moves `window` with the observations after it.
void drop(const Context& context, State& state, index_t position, Window& window, Reason why)
{
    CCD_OBSERVE_EVENT(context.observe, day_event(ObserverEvent::Kind::Drop,
                                                 state.series.dates()(position), name(why)));
    state.series.drop(position, window, why);
}

} // namespace

//==============================================================================
// The machine
//==============================================================================

const char* to_string(Phase phase)
{
    switch (phase)
    {
        case Phase::Prepare:     return "Prepare";
        case Phase::Initialize:  return "Initialize";
        case Phase::Lookback:    return "Lookback";
        case Phase::StartCatch:  return "StartCatch";
        case Phase::Extend:      return "Extend";
        case Phase::Restart:     return "Restart";
        case Phase::EndCatch:    return "EndCatch";
        case Phase::Done:        return "Done";
    }
    return "?";
}

const char* to_string(InitializeOutcome outcome)
{
    switch (outcome)
    {
        case InitializeOutcome::Exhausted:        return "exhausted";
        case InitializeOutcome::ShortSpan:        return "short span, grown";
        case InitializeOutcome::AllOutliers:      return "all Tmask outliers, grown";
        case InitializeOutcome::RejectedOutliers: return "Tmask drop rejected, grown";
        case InitializeOutcome::Unstable:         return "unstable, shifted";
        case InitializeOutcome::Stable:           return "stable";
        case InitializeOutcome::DroppedThenUnstable:
            return "Tmask outliers dropped, unstable, shifted";
        case InitializeOutcome::DroppedThenStable:
            return "Tmask outliers dropped, stable";
    }
    return "?";
}

const char* to_string(LookbackOutcome outcome)
{
    switch (outcome)
    {
        case LookbackOutcome::Change:   return "change";
        case LookbackOutcome::Dropped:  return "outlier dropped";
        case LookbackOutcome::Included: return "included";
    }
    return "?";
}

const char* to_string(ExtendOutcome outcome)
{
    switch (outcome)
    {
        case ExtendOutcome::Change:  return "change";
        case ExtendOutcome::Dropped: return "outlier dropped";
        case ExtendOutcome::Grown:   return "grown";
        case ExtendOutcome::SeriesEnd: return "series end";
        case ExtendOutcome::NoFit:   return "no fit";
    }
    return "?";
}

const char* to_string(Exit exit)
{
    switch (exit)
    {
        case Exit::None:                return "None";
        case Exit::TooFewObservations:  return "TooFewObservations";
        case Exit::VariogramNaN:        return "VariogramNaN";
        case Exit::InitializeExhausted: return "InitializeExhausted";
        case Exit::PeekBeyondEnd:       return "PeekBeyondEnd";
        case Exit::NoRoomToInitialize:  return "NoRoomToInitialize";
        case Exit::SeriesEnd:           return "SeriesEnd";
        case Exit::NoFit:               return "NoFit";
    }
    return "?";
}

void State::reset(const PixelSeries& pixel, ProcessingMask mask,
                  std::vector<std::uint8_t> reasons)
{
    phase = Phase::Prepare;
    exit = Exit::None;

    series.reset(pixel, std::move(mask), std::move(reasons));
    cursor = Cursor{};
    segments.clear();
    carried = 0;

    // Prepare's derived values, set again before anything reads them.
    peek_plan = PeekPlan{};
    extend.trained = false;
    forced.reset();
    search = SearchPlan{};
    resumable = true;
    carried_change = false;
    suspended = Suspended{};
}

void State::bind(const Strategies& strategies)
{
    stability_model.get(strategies.stability_fit,
                        [](const StabilityFit& f) { return f.make_model(); });
    stability.get(strategies.stability_rule,
                  [](const StabilityRule& r) { return r.make_scratch(); });
    lookback.get(strategies.lookback, [](const Lookback& l) { return l.make_scratch(); });
    edges.get(strategies.edge_segments, [](const EdgeSegments& e) { return e.make_scratch(); });
    segment_report.get(strategies.segment_report,
                       [](const SegmentReport& f) { return f.make_scratch(); });
    extend.model.get(strategies.model, [](const MonitoringModel& m) { return m.make_model(); });
    extend.test.get(strategies.change_test, [](const ChangeTest& c) { return c.make_scratch(); });
}

FitResult State::take_result()
{
    assert(phase == Phase::Done);

    // Every exit, the early ones too, returns the processing mask.
    FitResult result;
    result.models = std::move(segments);
    result.mask = series.take_mask();
    result.reasons = series.take_reasons();
    return result;
}

namespace
{

#if CCD_OBSERVE
// The observer's event of a segment written (Kind::Segment).
ObserverEvent segment_event(const ChangeModel& m)
{
    ObserverEvent e{ObserverEvent::Kind::Segment};
    e.day = m.start_day;
    e.day_to = m.end_day;
    e.observations = m.observation_count;
    e.outcome = "curve_qa " + std::to_string(static_cast<int>(m.curve_qa)) + " break " +
                std::to_string(m.break_day) + " change " + std::to_string(m.change_probability);
    return e;
}
#endif

// Records where the segment loop stopped, from the record as the failed
// guard found it.
void suspend(State& state, Exit why)
{
    Suspended& s = state.suspended;
    s.exit = why;
    s.window = state.cursor.window;
    s.prev_break = state.cursor.prev_break;
    s.fit_stop = 0;
    s.grow = false;
    s.num_coef = 0;
    s.magnitude.clear();
    s.final_segments = state.carried + state.segments.size();
}

// The route into EndCatch, recording why the segment loop ended.
Phase end_catch_because(State& state, Exit why)
{
    state.exit = why;
    suspend(state, why);
    return Phase::EndCatch;
}

// An early exit of Prepare: Done, recording why.
Phase done_because(State& state, Exit why)
{
    state.exit = why;
    suspend(state, why);
    return Phase::Done;
}

} // namespace

//------------------------------------------------------------------------------
// Routing
//------------------------------------------------------------------------------

bool room_to_initialize(const Context& context, const State& state)
{
    return context.strategies.start_search->room(state.cursor.window, state.series.size());
}

bool peek_fits(const State& state)
{
    return state.cursor.window.stop + shortest_peek(state.peek_plan) <= state.series.size();
}

bool start_catch_due(const Context& context, const State& state)
{
    const Cursor& cursor = state.cursor;
    assert(cursor.window.start >= cursor.prev_break);

    return state.segments.empty() && state.carried == 0 &&
        context.strategies.edge_segments->start_due(cursor.window.start - cursor.prev_break,
                                               state.peek_plan.peek);
}

Phase after_stable_start(const Context& context, State& state)
{
    if (state.cursor.window.start > state.cursor.prev_break)
        return Phase::Lookback;

    return after_lookback(context, state);
}

Phase after_lookback(const Context& context, State& state)
{
    if (start_catch_due(context, state))
        return Phase::StartCatch;

    return peek_guard(context, state);
}

Phase peek_guard(const Context&, State& state)
{
    if (peek_fits(state))
        return Phase::Extend;

    return end_catch_because(state, Exit::PeekBeyondEnd);
}

//------------------------------------------------------------------------------
// Transitions
//------------------------------------------------------------------------------

namespace
{

// Prepares the stages for the pixel from the statistics window's dates: the
// change test chooses the peek plan, which the stability rule and the
// lookback are prepared with, and the stability fit and the monitoring
// model give their coefficient rules.
void prepare_stages(const Context& context, State& state,
                    ArrayView<const std::int64_t, 1> stat_dates)
{
    const index_t detection_bands = context.shared.detection_bands.size();
    const ChangeTest& change_test = *context.strategies.change_test;
    const PeekPlan plan = change_test.prepare(
        stat_dates, detection_bands, *state.extend.test);
    // A peek holds an observation at least; chi2's schema refuses fewer than
    // two (a change is consecutive observations over the threshold).
    assert(plan.peek >= 1);
    state.peek_plan = plan;
    // A resume continues full peeks only (resume_series_end()).
    if (!plan.full)
        state.resumable = false;
    context.strategies.stability_rule->prepare(
        plan, detection_bands, *state.stability);
    context.strategies.lookback->prepare(
        plan, detection_bands, *state.lookback);
    state.stability_coefficients = context.strategies.stability_fit->coefficients();
    state.coefficients = context.strategies.model->coefficients();

    LOG_DEBUG("Peek size: " << state.peek_plan.peek);
}

} // namespace

Phase prepare(const Context& context, State& state)
{
    const config::Shared& shared = context.shared;
    MaskedSeries& series = state.series;

    LOG_DEBUG("num obs: " << context.pixel.dates().size());
    LOG_DEBUG("Processing mask initial count: " << series.size());
    //----------------------------------------------------------
    // Not enough observations
    //----------------------------------------------------------
    if (context.strategies.start_search->too_few(series.size()))
    {
        LOG_DEBUG("Processing Mask: " << series.mask().data());
        LOG_DEBUG("mask count < meow size" << series.size());
        return done_because(state, Exit::TooFewObservations);
    }

    //----------------------------------------------------------
    // Statistics window: the observations dated <= statistics_end
    // (views, no copy)
    //----------------------------------------------------------
    const index_t stat_count = series.count_until(shared.statistics_end);
    LOG_DEBUG("Stat mask count: " << stat_count);

    const auto stat_dates =
        series.dates().slice(range(index_t{0}, stat_count));

    const auto stat_spectral =
        series.spectral().slice(all(), range(index_t{0}, stat_count));

    LOG_DEBUG("stat dates: " << stat_dates);

    //----------------------------------------------------------
    // The pixel's peek plan and variogram, read by every step
    // below; the run's shared values are never written
    //----------------------------------------------------------
    prepare_stages(context, state, stat_dates);

    state.noise.bands = context.strategies.noise->estimate(stat_dates, stat_spectral);

    if (!state.noise.usable())
    {
        LOG_DEBUG("variogram has nans");
        return done_because(state, Exit::VariogramNaN);
    }
    LOG_DEBUG("Band noise: " << state.noise.bands);

    state.cursor = Cursor{context.strategies.start_search->first(0), 0};

    if (room_to_initialize(context, state))
        return begin_search(context, state);

    return end_catch_because(state, Exit::NoRoomToInitialize);
}

Phase begin_search(const Context& context, State& state)
{
    state.search = context.strategies.start_search->begin(
        state.series.dates(), state.cursor.prev_break, input_end(context));
    if (!state.search.resumable)
        state.resumable = false;
    if (state.search.exhausted == Exhaustion::Restore)
    {
        state.search_mask = state.series.mask();
        state.search_reasons = state.series.reasons();
    }
    return Phase::Initialize;
}

namespace
{

// What the window screen did to a candidate window.
enum class Screened : std::uint8_t
{
    Kept,          // nothing flagged
    Dropped,       // its outliers dropped, the window shortened by as many
    AllOutliers,   // every observation flagged: nothing dropped
    Rejected       // what its outliers leave is no candidate: nothing dropped
};

// The run's window screen (Tmask) on the window's observations. Its
// outliers leave the series only when what is left of the window is still a
// candidate (the start search's usable(); MaskedSeries::drop_outliers_if(),
// which shortens the window).
Screened screen_window(const Context& context, State& state,
                       ArrayView<const std::int64_t, 1> dates,
                       ArrayView<const scalar_t, 2> spectral)
{
    const StartSearch& search = *context.strategies.start_search;
    Window& window = state.cursor.window;

    auto outliers = context.strategies.window_screen->screen(dates, spectral, state.noise);
    LOG_DEBUG("Number of Tmask outliers found: " << outliers.count());
    //--------------------------------------------------------------------------
    // The screen removed everything
    //--------------------------------------------------------------------------
    if (outliers.count() == window.size())
        return Screened::AllOutliers;
    if (!outliers.any())
        return Screened::Kept;

#if CCD_OBSERVE
    // The outliers' days, for the observer, before a drop moves them.
    std::vector<std::int64_t> days;
    if (context.observe.observer)
        for (index_t j = 0; j < outliers.size(); ++j)
            if (outliers[j])
                days.push_back(dates(j));
#endif

    //--------------------------------------------------------------------------
    // Update persistent mask: drop them only if the shortened window is still
    // usable; otherwise grow, with nothing dropped
    //--------------------------------------------------------------------------
    const bool dropped = state.series.drop_outliers_if(
        outliers,
        window,
        Reason::WindowScreen,
        [&search](ArrayView<const std::int64_t, 1> shortened)
        {
            return search.usable(shortened);
        }
    );
#if CCD_OBSERVE
    if (dropped)
        for (const std::int64_t day : days)
            CCD_OBSERVE_EVENT(context.observe, day_event(ObserverEvent::Kind::Drop, day,
                                                         name(Reason::WindowScreen)));
#endif
    return dropped ? Screened::Dropped : Screened::Rejected;
}

// The stability fit of the detection bands over the window's observations
// (into stability_model, residuals kept), then the run's stability rule on it.
// Only the detection bands are read: the stability rule here and Lookback
// after it.
bool fit_stable(const Context& context, State& state,
                ArrayView<const std::int64_t, 1> dates,
                ArrayView<const scalar_t, 2> spectral)
{
    const BandList& detection_bands = context.shared.detection_bands;
    Model& fit = *state.stability_model;
    fit.fit(
        dates,
        spectral,
        state.stability_coefficients.min,
        state.stability_coefficients.max,
        detection_bands,
        KeepResiduals::Yes
    );

    return context.strategies.stability_rule->stable(
        fit,
        dates,
        state.noise,
        detection_bands,
        *state.stability);
}

} // namespace

InitializeOutcome initialize_step(const Context& context, State& state)
{
    MaskedSeries& series = state.series;
    Window& window = state.cursor.window;
    const StartSearch& search = *context.strategies.start_search;
    assert(window.size() >= search.min_window());

    //--------------------------------------------------------------------------
    // No candidate left before the end of the series
    //--------------------------------------------------------------------------
    const std::int64_t last_day = state.search.last_day;
    const index_t seen = last_day == std::numeric_limits<std::int64_t>::max()
        ? series.size()
        : count_through(series.dates(), last_day);
    if (search.exhausted(window, seen))
    {
        LOG_DEBUG("Final model window: (" << window.start << ", " << window.stop << ")");
        return InitializeOutcome::Exhausted;
    }

    // The window's observations; a Tmask drop below replaces them.
    auto dates =
        series.dates().slice(range(window.start, window.stop));
    auto spectral =
        series.spectral().slice(all(), range(window.start, window.stop));

    //--------------------------------------------------------------------------
    // Check minimum time span
    //--------------------------------------------------------------------------
    if (search.short_span(dates))
    {
        window.grow();
        return InitializeOutcome::ShortSpan;
    }

    LOG_DEBUG(
        "Checking window ("
        << window.start
        << ", "
        << window.stop
        << ")"
    );
    //--------------------------------------------------------------------------
    // The window screen (Tmask)
    //--------------------------------------------------------------------------
    const Screened screened = screen_window(context, state, dates, spectral);
    if (screened == Screened::AllOutliers)
    {
        window.grow();
        return InitializeOutcome::AllOutliers;
    }
    if (screened == Screened::Rejected)
    {
        window.grow();
        return InitializeOutcome::RejectedOutliers;
    }
    const bool dropped = screened == Screened::Dropped;
    if (dropped)
    {
        dates = series.dates().slice(range(window.start, window.stop));
        spectral = series.spectral().slice(all(), range(window.start, window.stop));
    }

    LOG_DEBUG("Generating models to check for stability");
    LOG_DEBUG(
        "window for stability: ("
        << window.start
        << ", "
        << window.stop
        << ")"
    );
    //--------------------------------------------------------------------------
    // Fit the detection bands, and the stability check
    //--------------------------------------------------------------------------
    if (fit_stable(context, state, dates, spectral))
    {
        LOG_DEBUG("Stable start found: (" << window.start << ", " << window.stop << ")");
        return dropped ? InitializeOutcome::DroppedThenStable : InitializeOutcome::Stable;
    }

    window.shift();
    LOG_DEBUG("Unstable model, shift window to: (" << window.start << ", " << window.stop << ")");
    return dropped ? InitializeOutcome::DroppedThenUnstable : InitializeOutcome::Unstable;
}

Phase initialize(const Context& context, State& state)
{
    LOG_DEBUG("Initialize for change model #: " << state.segments.size());
    const InitializeOutcome outcome = initialize_step(context, state);
    LOG_DEBUG("Initialize: " << to_string(outcome));

    switch (outcome)
    {
        case InitializeOutcome::Exhausted:
        {
            switch (state.search.exhausted)
            {
                case Exhaustion::Force:
                    state.cursor.window = context.strategies.start_search->forced(
                        state.series.dates(), state.cursor.prev_break, state.search.last_day);
                    state.extend.trained = false;
                    state.forced = state.cursor.window;
                    state.resumable = false;
                    return peek_guard(context, state);
                case Exhaustion::Restore:
                    // The window screen's drops of this search come back.
                    state.series.reset(context.pixel, state.search_mask, state.search_reasons);
                    state.resumable = false;
                    break;
                case Exhaustion::End:
                    break;
            }
            return end_catch_because(state, Exit::InitializeExhausted);
        }

        case InitializeOutcome::Stable:
        case InitializeOutcome::DroppedThenStable:
            return after_stable_start(context, state);

        case InitializeOutcome::ShortSpan:
        case InitializeOutcome::AllOutliers:
        case InitializeOutcome::RejectedOutliers:
        case InitializeOutcome::Unstable:
        case InitializeOutcome::DroppedThenUnstable:
            break;
    }
    return Phase::Initialize;
}

LookbackOutcome lookback_step(const Context& context, State& state)
{
    MaskedSeries& series = state.series;
    Window& window = state.cursor.window;

    const LookbackJudgement judged = context.strategies.lookback->judge(
        *state.stability_model, series.dates(), series.spectral(), window,
        state.cursor.prev_break, state.noise, context.shared.detection_bands, *state.lookback);

    switch (judged.verdict)
    {
        case LookbackVerdict::Change:
            return LookbackOutcome::Change;
        case LookbackVerdict::Drop:
            // The window moves down one position with the observations after
            // the dropped one.
            drop(context, state, judged.observation, window, Reason::LookbackOutlier);
            return LookbackOutcome::Dropped;
        case LookbackVerdict::Include:
            window.start = judged.observation;
            return LookbackOutcome::Included;
    }
    return LookbackOutcome::Change;
}

Phase lookback(const Context& context, State& state)
{
    const LookbackOutcome outcome = lookback_step(context, state);
    LOG_DEBUG("Lookback: " << to_string(outcome));

    if (outcome != LookbackOutcome::Change &&
        state.cursor.window.start > state.cursor.prev_break)
        return Phase::Lookback;

    return after_lookback(context, state);
}

void write_start_segment(const Context& context, State& state)
{
    const Cursor& cursor = state.cursor;
    assert(state.segments.empty() && cursor.prev_break == 0);

    auto window_0 = Window(cursor.prev_break, cursor.window.start);
    ChangeModel result = context.strategies.edge_segments->catch_window(
        state.series.dates(),
        state.series.spectral(),
        window_0,
        CurveQA::Start,
        *state.edges
    );
    CCD_OBSERVE_EVENT(context.observe, segment_event(result));
    state.segments.push_back(std::move(result));
}

Phase start_catch(const Context& context, State& state)
{
    write_start_segment(context, state);
    return peek_guard(context, state);
}

namespace
{

// The bands no test reads (those not in the detection bands), as the last test
// would have had them: fitted on the last refit's window with its
// coefficient count, their residuals over the last test's peek.
void complete_models(const Context& context, State& state)
{
    ExtendState& extend = state.extend;
    const auto masked_dates = state.series.dates();
    const auto masked_spectral = state.series.spectral();

    BandList others;
    for (index_t band = 0; band < masked_spectral.extent(0); ++band)
        if (!context.shared.detection_bands.contains(band))
            others.push_back(band);

    const Window& fit = extend.fit_window;
    const Window& peek = extend.peek;
    extend.model->fit(
        masked_dates.slice(range(fit.start, fit.stop)),
        masked_spectral.slice(all(), range(fit.start, fit.stop)),
        state.coefficients.count(fit.size()),
        state.coefficients.max,
        others,
        KeepResiduals::No
    );
    extend.model->peek_residuals(
        masked_dates.slice(range(peek.start, peek.stop)),
        masked_spectral.slice(all(), range(peek.start, peek.stop)),
        others,
        extend.residuals
    );
    extend.complete = true;
}

// The first observation after the window, or the last when none is left:
// where a segment that reaches the series end breaks.
index_t after_window(const State& state)
{
    return std::min(state.cursor.window.stop, state.series.size() - 1);
}

// Writes the segment Extend has grown, from its last peek test, breaking at
// observation `at`, and ends the extend: on a change (change probability 1)
// the change test's break_at(), at the series end (0) after_window(). Its
// coefficients are segment_report's; each band's magnitude is the change
// test's, or `magnitude` when given (a resumed series end, whose last test
// the stored run made).
void write_extended_segment(
    const Context& context,
    State& state,
    bool change,
    index_t at,
    const std::vector<scalar_t>* magnitude = nullptr
)
{
    const MaskedSeries& series = state.series;
    const Window& window = state.cursor.window;
    ExtendState& extend = state.extend;

    assert(extend.trained);

    const auto masked_dates = series.dates();
    const index_t num_bands = series.spectral().extent(0);

    if (!extend.complete)
        complete_models(context, state);

    ChangeModel result;
    result.start_day = masked_dates(window.start);
    result.end_day = masked_dates(window.stop - 1);
    const ChangeTest& change_test = *context.strategies.change_test;
    const Scratch& scratch = *extend.test;
    const Peek last = peek_of(context, state);
    result.break_day = masked_dates(at);
    result.observation_count = window.stop - window.start;
    result.change_probability = change ? 1.0 : 0.0;

    const index_t num_coef = context.strategies.segment_report->segment(
        *extend.model, masked_dates, series.spectral(), window, extend.num_coef,
        context.max_coefficients, *state.segment_report, result.bands);
    result.curve_qa = static_cast<CurveQA>(num_coef);
    for (index_t band = 0; band < num_bands; ++band)
        result.bands[band].magnitude =
            magnitude ? (*magnitude)[band] : change_test.magnitude(last, scratch, band);

    CCD_OBSERVE_EVENT(context.observe, segment_event(result));
    state.segments.push_back(std::move(result));
    extend.trained = false;
}

// The segment of a forced start the series ended before Extend fitted:
// segment_report's of its window (SegmentReport::unfitted()), through the last
// observation, change probability 0.
void write_forced_segment(const Context& context, State& state)
{
    const MaskedSeries& series = state.series;
    const auto masked_dates = series.dates();
    const Window window = *state.forced;
    state.forced.reset();
    ++state.extend.fits;   // the forced window's fit is a refit

    ChangeModel result;
    result.start_day = masked_dates(window.start);
    result.end_day = masked_dates(series.size() - 1);
    result.break_day = result.end_day;
    result.observation_count = window.size();
    result.change_probability = 0.0;
    const index_t num_coef = context.strategies.segment_report->unfitted(
        *state.extend.model, masked_dates, series.spectral(), window,
        state.coefficients.count(window.size()), context.max_coefficients,
        *state.segment_report, result.bands);
    result.curve_qa = static_cast<CurveQA>(num_coef);

    CCD_OBSERVE_EVENT(context.observe, segment_event(result));
    state.segments.push_back(std::move(result));
}

#if CCD_OBSERVE
const char* to_string(TestOutcome outcome)
{
    switch (outcome)
    {
        case TestOutcome::Change:    return "change";
        case TestOutcome::NoChange:  return "no change";
        case TestOutcome::Undecided: return "undecided";
    }
    return "?";
}
#endif

} // namespace

namespace
{

// A window smaller than the change test's min_window() grows untested: NoFit
// when no observation is left to grow by.
ExtendOutcome grow_untested(State& state)
{
    Window& window = state.cursor.window;
    if (window.stop >= state.series.size())
        return ExtendOutcome::NoFit;
    window = Window(window.start, window.stop + 1);
    return ExtendOutcome::Grown;
}

// The refit schedule: refits the detection bands over the window when Extend
// is not trained or the run's refit schedule is due, with the coefficient
// count of the next test. False when the model cannot be fitted.
bool refit_if_due(const Context& context, State& state)
{
    const Window& window = state.cursor.window;
    ExtendState& extend = state.extend;
    const auto masked_dates = state.series.dates();
    const auto masked_spectral = state.series.spectral();

    // # Used for comparison against fit_span
    auto model_span = span(masked_dates, window);

    if (extend.trained &&
        !context.strategies.refit->due(window.size(), model_span, extend.fit_window.size(),
                                       extend.fit_span))
        return true;

    // A refit on the window the last one grew from extends it.
    const bool extends = extend.trained;
    extend.trained = true;
    extend.fit_window = window;
    extend.fit_span = span(masked_dates, extend.fit_window);
    LOG_DEBUG("Retrain models");

    // The detection bands: seasonal_rmse() reads their residuals; the
    // magnitude is set from the peek residuals when the segment is
    // written, which fits the other bands (complete_models()).
    ++extend.fits;
    Model& model = *extend.model;
    const auto fit_dates =
        masked_dates.slice(range(extend.fit_window.start, extend.fit_window.stop));
    const auto fit_spectral =
        masked_spectral.slice(all(), range(extend.fit_window.start, extend.fit_window.stop));
    const BandList& detection_bands = context.shared.detection_bands;
    const bool fitted = extends
        ? model.extend(fit_dates, fit_spectral, extend.num_coef, state.coefficients.max,
                       detection_bands, KeepResiduals::Yes)
        : model.fit(fit_dates, fit_spectral, extend.num_coef, state.coefficients.max,
                    detection_bands, KeepResiduals::Yes);
    if (!fitted)
    {
        extend.trained = false;
        return false;
    }
    state.forced.reset();
    return true;
}

// The peek's residuals of the detection bands, then the run's change test
// on `peek` (which reads them), reported to the observer.
TestOutcome test_peek(const Context& context, State& state, const Peek& peek)
{
    ExtendState& extend = state.extend;
    const Window& peek_window = extend.peek;
    const auto masked_dates = state.series.dates();

    extend.model->peek_residuals(
        masked_dates.slice(range(peek_window.start, peek_window.stop)),
        state.series.spectral().slice(all(), range(peek_window.start, peek_window.stop)),
        context.shared.detection_bands,
        extend.residuals
    );

    const TestOutcome outcome =
        context.strategies.change_test->test(peek, *extend.test, extend.result);

    CCD_OBSERVE_EVENT(context.observe, [&] {
        ObserverEvent e = day_event(ObserverEvent::Kind::Test, masked_dates(peek_window.start),
                                    to_string(outcome));
        if (!extend.result.statistics.empty())
            e.values = extend.result.statistics.front().values;
        return e;
    }());
    return outcome;
}

// After a test that found no change: the run's outlier rule on the peek,
// then the series-end check, then the grow.
ExtendOutcome after_no_change(const Context& context, State& state, const Peek& peek)
{
    MaskedSeries& series = state.series;
    Window& window = state.cursor.window;
    const Window& peek_window = state.extend.peek;
    const index_t shortest = shortest_peek(state.peek_plan);

    const OutlierOutcome judged = context.strategies.outlier->judge(peek, state.extend.result);
    //--------------------------------------------------------------------------
    // Outlier detected: drop the first peek observation, at window.stop; the
    // window keeps its coordinates
    //--------------------------------------------------------------------------
    if (judged == OutlierOutcome::Drop)
    {
        LOG_DEBUG("Outlier detected at: " << peek_window.start);
        LOG_DEBUG("Updating processing mask extend");
        // A drop that leaves no full peek ends the curve at this test: its
        // other bands are this test's, before the drop moves the peek.
        if (window.stop + shortest > series.size() - 1)
            complete_models(context, state);
        drop(context, state, peek_window.start, window, Reason::Outlier);
        return ExtendOutcome::Dropped;
    }
    // A peek too short for the outlier rule to tell: the curve ends here.
    if (judged == OutlierOutcome::EndSeries)
    {
        state.resumable = false;
        return ExtendOutcome::SeriesEnd;
    }

    //--------------------------------------------------------------------------
    // The peek reaches the last observation and holds no change: the curve
    // ends at the window, its last test a full peek (no peek the test reads
    // is left after the next observation)
    //--------------------------------------------------------------------------
    assert(peek_fits(state));
    if (window.stop + shortest >= series.size())
        return ExtendOutcome::SeriesEnd;

    //--------------------------------------------------------------------------
    // Otherwise grow
    //--------------------------------------------------------------------------
    window = Window(window.start, window.stop + 1);
    return ExtendOutcome::Grown;
}

} // namespace

ExtendOutcome extend_step(const Context& context, State& state)
{
    MaskedSeries& series = state.series;
    const Window& window = state.cursor.window;
    ExtendState& extend = state.extend;

    // while model_window.stop + peek_size <= period.shape[0]:
    assert(peek_fits(state));

    //--------------------------------------------------------------------------
    // A window too small to test grows untested
    //--------------------------------------------------------------------------
    if (window.size() < state.peek_plan.min_window)
        return grow_untested(state);

    if (!extend.trained)
    {
        LOG_DEBUG("initial model window extend: " << window.start << ", " << window.stop);
    }

    // Indexed by band.
    extend.residuals.resize(series.spectral().extent(0));

    extend.num_coef = state.coefficients.count(window.size());

    // The next `peek` observations, fewer at the series end.
    const Window& peek_window = extend.peek =
        Window(window.stop, std::min(window.stop + state.peek_plan.peek, series.size()));

    LOG_DEBUG("detecting change for: " << peek_window.start << ", " << peek_window.stop);

    //--------------------------------------------------------------------------
    // The refit schedule
    //--------------------------------------------------------------------------
    if (!refit_if_due(context, state))
        return ExtendOutcome::NoFit;
    extend.complete = false;

    // No observation left to test: the curve ends at the window.
    if (peek_window.size() == 0)
        return ExtendOutcome::SeriesEnd;

    //--------------------------------------------------------------------------
    // The peek's residuals, comparison RMSE and magnitudes
    //--------------------------------------------------------------------------
    // A Peek holds the residuals and the model by reference, so it is built
    // before test_peek() writes the residuals it reads.
    const Peek peek = peek_of(context, state);
    const TestOutcome outcome = test_peek(context, state, peek);

    //--------------------------------------------------------------------------
    // Change detected: the segment ends at the window
    //--------------------------------------------------------------------------
    if (outcome == TestOutcome::Change)
    {
        LOG_DEBUG("Change detected at: " << peek_window.start);
        return ExtendOutcome::Change;
    }
    // A shortened peek that decides nothing: the curve ends at the window
    // (where a longer series would test again: not resumable).
    if (outcome == TestOutcome::Undecided)
    {
        state.resumable = false;
        return ExtendOutcome::SeriesEnd;
    }

    return after_no_change(context, state, peek);
}

namespace
{

// After a change: the segment with change probability 1, breaking at the
// change test's break_at(); the peek's observations before the break leave
// the series after it. Next: Restart.
Phase on_change(const Context& context, State& state)
{
    const index_t first = state.extend.peek.start;
    const index_t at =
        context.strategies.change_test->break_at(peek_of(context, state), *state.extend.test);
    write_extended_segment(context, state, true, at);
    if (at > first)
        state.resumable = false;   // the record keeps no drops before a change
    for (index_t t = first; t < at; ++t)
        drop(context, state, first, state.cursor.window, Reason::BeforeChange);
    return Phase::Restart;
}

// No fit: the series ended before the window was fitted, or it could not
// be. A forced start the series ended before Extend fitted reports its own
// segment; otherwise the curve is not written. Not resumable. Next: Done.
Phase on_no_fit(const Context& context, State& state)
{
    if (state.forced)
        write_forced_segment(context, state);
    state.exit = Exit::NoFit;
    state.resumable = false;
    return Phase::Done;
}

// Series end: the curve reached the end of the series, after a test whose
// peek reached the last observation (`grow`: the grow it would make is
// pending), or a drop that left no full peek. It is the last curve: the
// segment with change probability 0, the suspension at it, and no Restart;
// the End segment's guard reads the curve's stop. Next: EndCatch.
Phase on_series_end(const Context& context, State& state, bool grow)
{
    suspend(state, Exit::SeriesEnd);
    state.suspended.fit_stop = state.extend.fit_window.stop;
    state.suspended.grow = grow;

    write_extended_segment(context, state, false, after_window(state));
    state.suspended.num_coef = state.extend.num_coef;
    for (const BandFit& band : state.segments.back().bands)
        state.suspended.magnitude.push_back(band.magnitude);

    state.cursor.prev_break = state.cursor.window.stop;
    state.exit = Exit::SeriesEnd;
    return Phase::EndCatch;
}

} // namespace

Phase extend(const Context& context, State& state)
{
    const ExtendOutcome outcome = extend_step(context, state);
    LOG_DEBUG("Extend: " << to_string(outcome));

    if (outcome == ExtendOutcome::Change)
        return on_change(context, state);
    if (outcome == ExtendOutcome::NoFit)
        return on_no_fit(context, state);
    if (outcome != ExtendOutcome::SeriesEnd && peek_fits(state))
        return Phase::Extend;
    return on_series_end(context, state, outcome == ExtendOutcome::SeriesEnd);
}

void move_past_segment(const Context& context, State& state)
{
    Cursor& cursor = state.cursor;

    cursor.prev_break = cursor.window.stop;
    cursor.window = context.strategies.start_search->first(cursor.window.stop);
}

Phase restart(const Context& context, State& state)
{
    move_past_segment(context, state);

    if (room_to_initialize(context, state))
        return begin_search(context, state);

    return end_catch_because(state, Exit::NoRoomToInitialize);
}

void write_end_segment(const Context& context, State& state)
{
    const MaskedSeries& series = state.series;
    auto window_1 = Window(state.cursor.prev_break, series.size());
    ChangeModel result = context.strategies.edge_segments->catch_window(
        series.dates(),
        series.spectral(),
        window_1,
        CurveQA::End,
        *state.edges
    );
    CCD_OBSERVE_EVENT(context.observe, segment_event(result));
    state.segments.push_back(std::move(result));
}

Phase end_catch(const Context& context, State& state)
{
    const index_t prev_break = state.cursor.prev_break;
    assert(prev_break <= state.series.size());
    const bool after_change = state.segments.empty()
        ? state.carried > 0 && state.carried_change
        : state.segments.back().change_probability == 1.0;
    if (context.strategies.edge_segments->end_due(state.series.size() - prev_break,
                                             state.peek_plan.peek, after_change))
        write_end_segment(context, state);
    return Phase::Done;
}

//------------------------------------------------------------------------------
// The machine
//------------------------------------------------------------------------------

Phase step(const Context& context, State& state)
{
    switch (state.phase)
    {
        case Phase::Prepare:     return prepare(context, state);
        case Phase::Initialize:  return initialize(context, state);
        case Phase::Lookback:    return lookback(context, state);
        case Phase::StartCatch:  return start_catch(context, state);
        case Phase::Extend:      return extend(context, state);
        case Phase::Restart:     return restart(context, state);
        case Phase::EndCatch:    return end_catch(context, state);
        case Phase::Done:        break;
    }
    assert(false && "step() on a Done machine");
    return Phase::Done;
}

namespace
{

// Reports the transition from `from` to `to`, the record as the step left
// it: to the context's observer with the cursor and the series length
// (CCD_OBSERVE), and as one DEBUG line through the logger, so the log level
// is the trace's only switch, and below DEBUG it compiles to nothing:
//
//     [DEBUG] standard: <phase> -> <next phase> | <cursor, observations,
//                                                  segments, exit>
void report_transition(const Context& context, const State& state, Phase from, Phase to)
{
#if CCD_OBSERVE
    if (context.observe.observer)
    {
        ObserverEvent e{ObserverEvent::Kind::Transition};
        e.machine = "standard";
        e.phase = to_string(from);
        e.next = to_string(to);
        e.window_start = state.cursor.window.start;
        e.window_stop = state.cursor.window.stop;
        e.prev_break = state.cursor.prev_break;
        e.observations = state.series.size();
        context.observe.observer->event(context.observe.pixel, e);
    }
#else
    (void)context;
    (void)state;
    (void)from;
    (void)to;
#endif

    LOG_DEBUG(
        "standard: " << to_string(from) << " -> " << to_string(to)
        << " | window (" << state.cursor.window.start << ", " << state.cursor.window.stop
        << "), prev_break " << state.cursor.prev_break << ", " << state.series.size()
        << " observations, " << state.segments.size() << " segments, exit "
        << to_string(state.exit)
    );
}

// Reports the pixel's exit (state.exit) to the context's observer, if any.
void report_exit(const Context& context, const State& state)
{
#if CCD_OBSERVE
    if (context.observe.observer)
    {
        ObserverEvent e{ObserverEvent::Kind::Exit};
        e.outcome = to_string(state.exit);
        e.observations = state.segments.size();
        context.observe.observer->event(context.observe.pixel, e);
    }
#else
    (void)context;
    (void)state;
#endif
}

// Steps `state` until its phase is Done, reporting each transition: the
// machine's only loop.
void drive(const Context& context, State& state)
{
    while (state.phase != Phase::Done)
    {
        const Phase next = step(context, state);
        report_transition(context, state, state.phase, next);
        state.phase = next;
    }
}

} // namespace

//------------------------------------------------------------------------------
// Detector
//------------------------------------------------------------------------------

FitResult Detector::fit(
    const PixelMode& mode,
    const PixelSeries& series,
    ProcessingMask mask,
    const config::Shared& shared,
    const Strategies& strategies,
    Observe observe,
    std::vector<std::uint8_t> reasons
)
{
    state_.bind(strategies);
    if (mode.procedure != PixelProcedure::Standard)
    {
        FitResult result = strategies.edge_segments->whole_series(
            mode.procedure, series, std::move(mask), *state_.edges);
        result.reasons = std::move(reasons);
        return result;
    }

    LOG_DEBUG("Standard");
    const Context context{shared, series, strategies, observe, mode.max_coefficients};
    State& state = state_;

    state.reset(series, std::move(mask), std::move(reasons));
    if (context.max_coefficients > 0)
        state.resumable = false;   // a resume does not cap
    drive(context, state);
    report_exit(context, state);
    return state.take_result();
}

//------------------------------------------------------------------------------
// Resume
//------------------------------------------------------------------------------

namespace
{

// That a stored suspension can be where a run on this series stopped: its
// cursor inside the series, the window as long as the exit's guard leaves
// it, and at a series end a fit window inside the window and the last
// test's values. A record that does not is refused, before anything reads
// at its positions.
void check_fits(const Context& context, const State& state, const Suspended& from)
{
    const index_t size = state.series.size();
    const index_t meow = context.strategies.start_search->min_window();
    const Window& w = from.window;
    const auto refuse = [&](const std::string& what)
    {
        throw std::invalid_argument(
            "the resume record does not fit the series (" + std::to_string(size) +
            " kept observations): " + what);
    };

    if (!(from.prev_break <= w.start && w.start <= w.stop))
        refuse("its cursor is not prev_break <= window start <= window stop");
    if (w.start > size)
        refuse("its window starts after the series");

    switch (from.exit)
    {
        case Exit::NoRoomToInitialize:
            if (w.size() != meow)
                refuse("NoRoomToInitialize with a window other than the start search's");
            break;
        case Exit::InitializeExhausted:
        case Exit::PeekBeyondEnd:
            if (w.size() < meow || w.stop > size)
                refuse(std::string(to_string(from.exit)) +
                       " with a window shorter than the start search's or past the series");
            break;
        case Exit::SeriesEnd:
            if (w.size() < meow || w.stop >= size)
                refuse("SeriesEnd with a window shorter than the start search's or reaching the series end");
            if (!(w.start < from.fit_stop && from.fit_stop <= w.stop))
                refuse("SeriesEnd with a fit window outside the window");
            if (from.num_coef < 1 || from.num_coef > state.coefficients.max)
                refuse("SeriesEnd with a coefficient count outside 1..the model's widest");
            if (from.magnitude.size() != state.series.spectral().extent(0))
                refuse("SeriesEnd with " + std::to_string(from.magnitude.size()) +
                       " magnitudes for " + std::to_string(state.series.spectral().extent(0)) +
                       " bands");
            break;
        default:
            refuse("an exit with nothing to resume");
    }
}

// The stored run's drops (`excluded_days`) leave the series, the latest
// first, so each earlier position stays where the stored run found it. A
// day that is not an observation the filters keep is refused.
void replay_drops(State& state, const std::vector<std::int64_t>& excluded_days)
{
    Window unused;
    for (auto day = excluded_days.rbegin(); day != excluded_days.rend(); ++day)
    {
        const auto dates = state.series.dates();
        const std::int64_t* begin = &dates(0);
        const std::int64_t* end = begin + dates.size();
        const std::int64_t* at = std::lower_bound(begin, end, *day);
        if (at == end || *at != *day)
            throw std::invalid_argument(
                "the resume record's dropped day " + std::to_string(*day) +
                " is not an observation the filters keep");
        state.series.drop(static_cast<index_t>(at - begin), unused, Reason::Stored);
    }
}

// resume()'s SeriesEnd: the open segment as the stored run left it, its
// models refitted on its fit window (as the run fitted them, bit for bit),
// then the guard that failed tested again on the longer series.
Phase resume_series_end(const Context& context, State& state, const Suspended& from)
{
    ExtendState& extend = state.extend;
    Window& window = state.cursor.window;
    const auto dates = state.series.dates();
    const auto spectral = state.series.spectral();

    extend.trained = true;
    extend.fit_window = Window(window.start, from.fit_stop);
    extend.fit_span = span(dates, extend.fit_window);
    ++extend.fits;
    extend.model->fit(
        dates.slice(range(extend.fit_window.start, extend.fit_window.stop)),
        spectral.slice(all(), range(extend.fit_window.start, extend.fit_window.stop)),
        state.coefficients.count(extend.fit_window.size()),
        state.coefficients.max,
        every_band(spectral.extent(0)),
        KeepResiduals::Yes
    );
    extend.complete = true;

    // After a grow the last peek reached the last observation: the series
    // continues once it holds one more, and the grow is applied (testing
    // that peek again would give the same outcome). After a drop no full
    // peek was left.
    const index_t reach = window.stop + state.peek_plan.peek;
    if (from.grow ? reach < state.series.size() : reach <= state.series.size())
    {
        if (from.grow)
            window = Window(window.start, window.stop + 1);
        return Phase::Extend;
    }

    // The series still ends here: the stored run's last test, its segment.
    extend.num_coef = from.num_coef;
    extend.peek = Window(window.stop, reach);
    write_extended_segment(context, state, false, after_window(state), &from.magnitude);
    state.cursor.prev_break = window.stop;
    state.exit = Exit::SeriesEnd;
    state.suspended = from;
    return Phase::EndCatch;
}

} // namespace

FitResult Detector::resume(
    const PixelSeries& series,
    ProcessingMask mask,
    const config::Shared& shared,
    const Strategies& strategies,
    const Suspended& from,
    const std::vector<std::int64_t>& excluded_days,
    Observe observe,
    bool carried_change,
    std::vector<std::uint8_t> reasons
)
{
    const Context context{shared, series, strategies, observe};
    State& state = state_;

    state.bind(strategies);
    state.reset(series, std::move(mask), std::move(reasons));
    // The statistics are the stored run's, and the series is no shorter, so
    // Prepare neither exits early nor computes other values: it reads the
    // series before any drop, as the stored run's did. Should it exit early
    // all the same, the pixel ends there, as a fresh run's would.
    const Phase prepared = prepare(context, state);
    if (prepared == Phase::Done)
    {
        report_transition(context, state, Phase::Prepare, prepared);
        state.phase = prepared;
        report_exit(context, state);
        return state.take_result();
    }
    state.exit = Exit::None;
    state.suspended = Suspended{};

    replay_drops(state, excluded_days);
    check_fits(context, state, from);
    state.carried = from.final_segments;
    state.carried_change = carried_change;
    state.cursor = Cursor{from.window, from.prev_break};

    switch (from.exit)
    {
        case Exit::NoRoomToInitialize:
            state.phase = room_to_initialize(context, state)
                ? begin_search(context, state)
                : end_catch_because(state, Exit::NoRoomToInitialize);
            break;
        case Exit::InitializeExhausted:
            state.phase = begin_search(context, state);
            break;
        case Exit::PeekBeyondEnd:
            state.phase = peek_guard(context, state);
            break;
        case Exit::SeriesEnd:
            state.phase = resume_series_end(context, state, from);
            break;
        default:
            assert(false && "resume() from an exit with nothing to resume");
            state.phase = Phase::Done;
    }
    // The observer sees Prepare go where the stored run continues, with the
    // cursor it continues from.
    report_transition(context, state, Phase::Prepare, state.phase);

    drive(context, state);
    report_exit(context, state);
    return state.take_result();
}

const Suspended& Detector::suspended() const
{
    static const Suspended none;
    return state_.resumable ? state_.suspended : none;
}

} // namespace machine

} // namespace ccd
