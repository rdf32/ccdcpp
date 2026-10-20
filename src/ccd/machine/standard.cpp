#include "ccd/machine/standard.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>
#include <cassert>
#include <cmath>
#include <string>
#include <utility>

#include "ccd/core/maths.hpp"
#include "ccd/core/logger.hpp"

namespace ccd
{

namespace standard
{

InitializeCriteria initialize_criteria(const HarmonicOptions& options)
{
    return InitializeCriteria{
        options.DETECTION_BANDS,
        options.COEFFICIENT_MIN,
        options.COEFFICIENT_MAX
    };
}

Model& stability_fit(const Context& context, State& state)
{
    return state.init_model.get(context.strategies.stability_fit,
                                [](const StabilityFit& f) { return f.make_model(); });
}

Scratch& stability_scratch(const Context& context, State& state)
{
    return state.stability.get(context.strategies.stability_rule,
                               [](const StabilityRule& r) { return r.make_scratch(); });
}

Scratch& edge_scratch(const Context& context, State& state)
{
    return state.edges.get(context.strategies.edge_fit,
                           [](const EdgeFit& e) { return e.make_scratch(); });
}

Scratch& lookback_scratch(const Context& context, State& state)
{
    return state.lookback.get(context.strategies.lookback,
                              [](const Lookback& l) { return l.make_scratch(); });
}

namespace
{

// The state's segment-fit scratch, of the run's segment-fit strategy.
Scratch& segment_scratch(const Context& context, State& state)
{
    return state.segment_fit.get(context.strategies.segment_fit,
                                 [](const SegmentFit& f) { return f.make_scratch(); });
}

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

// The extension's change-test buffers, of the run's change-test strategy.
Scratch& test_scratch(const Context& context, Extension& extension)
{
    return extension.test.get(context.strategies.change_test,
                              [](const ChangeTest& c) { return c.make_scratch(); });
}

// Extend's last peek, as its tests read it.
Peek peek_of(const Context& context, const State& state)
{
    const Extension& extension = state.extension;
    return Peek{*extension.model, state.series.dates(), state.series.spectral(),
                    state.cursor.window,
                    extension.fit_window, extension.fits, extension.peek, extension.residuals,
                    state.variogram, context.options.DETECTION_BANDS};
}

} // namespace

index_t coefficient_count(
    index_t observations,
    const HarmonicOptions& options
)
{
    const index_t span =
        observations / options.NUM_OBS_FACTOR;

    if (span < options.COEFFICIENT_MID)
    {
        return options.COEFFICIENT_MIN;
    }
    else if (span < options.COEFFICIENT_MAX)
    {
        return options.COEFFICIENT_MID;
    }
    else
    {
        return options.COEFFICIENT_MAX;
    }
}

//==============================================================================
// The Standard machine
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

void State::reset(const HarmonicWorkspace& workspace, ProcessingMask mask)
{
    phase = Phase::Prepare;
    exit = Exit::None;

    series.reset(workspace, std::move(mask));
    cursor = Cursor{};
    segments.clear();
    carried = 0;

    // Prepare's derived values, set again before anything reads them.
    cadence = Cadence{};
    initialize_criteria = InitializeCriteria{};
    extension.trained = false;
    forced.reset();
    search = SearchPlan{};
    resumable = true;
    carried_change = false;
    suspended = Suspended{};
}

FitResult State::take_result()
{
    assert(phase == Phase::Done);

    // Every exit, the early ones too, returns the processing mask.
    FitResult result;
    result.models = std::move(segments);
    result.mask = series.take_mask();
    return result;
}

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

ObserverEvent observed(const State& state)
{
    ObserverEvent e{ObserverEvent::Kind::Transition};
    e.window_start = state.cursor.window.start;
    e.window_stop = state.cursor.window.stop;
    e.prev_break = state.cursor.prev_break;
    e.observations = state.series.size();
    return e;
}

std::string summary(const State& state)
{
    return
        "window (" + std::to_string(state.cursor.window.start) +
        ", " + std::to_string(state.cursor.window.stop) +
        "), prev_break " + std::to_string(state.cursor.prev_break) +
        ", " + std::to_string(state.series.size()) + " observations, " +
        std::to_string(state.segments.size()) + " segments, exit " +
        to_string(state.exit);
}

namespace
{

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
    return state.cursor.window.stop + state.cadence.shortest <= state.series.size();
}

bool start_catch_due(const Context& context, const State& state)
{
    const Cursor& cursor = state.cursor;
    assert(cursor.window.start >= cursor.prev_break);

    return state.segments.empty() && state.carried == 0 &&
        context.strategies.edge_fit->start_due(cursor.window.start - cursor.prev_break,
                                               state.cadence.peek);
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

Phase prepare(const Context& context, State& state)
{
    const auto& options = context.options;
    MaskedSeries& series = state.series;

    // validate() refuses less: the change test needs consecutive
    // observations, and both peeks read their first magnitude.
    assert(options.PEEK_SIZE >= 2);

    LOG_DEBUG("num obs: " << context.pixel.dates().size());
    LOG_DEBUG("Processing mask initial count: " << series.size());
    //----------------------------------------------------------
    // Not enough observations
    //----------------------------------------------------------
    if (context.strategies.start_search->too_few(series.size()))
    {
        LOG_DEBUG("Processing Mask: " << series.mask().data());
        LOG_DEBUG("mask count < meow size" << series.size());
        state.exit = Exit::TooFewObservations;
        suspend(state, Exit::TooFewObservations);
        return Phase::Done;
    }

    //----------------------------------------------------------
    // Statistics window: the observations dated <= STAT_ORD
    // (views, no copy)
    //----------------------------------------------------------
    const index_t stat_count = series.count_until(options.STAT_ORD);
    LOG_DEBUG("Stat mask count: " << stat_count);

    const auto stat_dates =
        series.dates().slice(range(index_t{0}, stat_count));

    const auto stat_spectral =
        series.spectral().slice(all(), range(index_t{0}, stat_count));

    LOG_DEBUG("stat dates: " << stat_dates);

    //----------------------------------------------------------
    // The pixel's cadence and variogram, read by every step
    // below; the caller's options are never written
    //----------------------------------------------------------
    const index_t detection_bands = options.DETECTION_BANDS.size();
    const ChangeTest& change_test = *context.strategies.change_test;
    const PeekPlan plan = change_test.prepare(
        stat_dates, detection_bands, test_scratch(context, state.extension));
    state.cadence = Cadence{plan.peek, plan.full ? plan.peek : 0, plan.min_window};
    // A resume continues full peeks only (resume_series_end()).
    if (!plan.full)
        state.resumable = false;
    context.strategies.stability_rule->prepare(
        state.cadence.peek, detection_bands, stability_scratch(context, state));
    context.strategies.lookback->prepare(
        state.cadence.peek, detection_bands, lookback_scratch(context, state));
    state.initialize_criteria = initialize_criteria(options);

    LOG_DEBUG("Peek size: " << state.cadence.peek);

    state.variogram.bands = context.strategies.noise_floor->estimate(stat_dates, stat_spectral);

    if (!state.variogram.usable())
    {
        LOG_DEBUG("variogram has nans");
        state.exit = Exit::VariogramNaN;
        suspend(state, Exit::VariogramNaN);
        return Phase::Done;
    }
    LOG_DEBUG("Variogram values: " << state.variogram.bands);

    state.cursor = Cursor{context.strategies.start_search->first(0), 0};

    if (room_to_initialize(context, state))
        return begin_search(context, state);

    return end_catch_because(state, Exit::NoRoomToInitialize);
}

Phase begin_search(const Context& context, State& state)
{
    state.search = context.strategies.start_search->begin(
        state.series.dates(), state.cursor.prev_break, input_end(context));
    state.initialize_criteria.last_day = state.search.last_day;
    if (!state.search.resumable)
        state.resumable = false;
    if (state.search.exhausted == Exhaustion::Restore)
        state.search_mask = state.series.mask();
    return Phase::Initialize;
}

InitializeOutcome initialize_outcome(const Context& context, State& state)
{
    LOG_DEBUG("Initialize for change model #: " << state.segments.size());

    const InitializeOutcome outcome = initialize_step(
        state.series,
        state.cursor.window,
        state.initialize_criteria,
        state.variogram,
        InitializeStages{*context.strategies.start_search, *context.strategies.window_screen,
                         stability_fit(context, state),
                         *context.strategies.stability_rule, stability_scratch(context, state)}
    );
    LOG_DEBUG("Initialize: " << to_string(outcome));
    return outcome;
}

Phase initialize(const Context& context, State& state)
{
    const InitializeOutcome outcome = initialize_outcome(context, state);

    switch (outcome)
    {
        case InitializeOutcome::Exhausted:
        {
            switch (state.search.exhausted)
            {
                case Exhaustion::Force:
                    state.cursor.window = context.strategies.start_search->forced(
                        state.series.dates(), state.cursor.prev_break, state.search.last_day);
                    state.extension.trained = false;
                    state.forced = state.cursor.window;
                    state.resumable = false;
                    return peek_guard(context, state);
                case Exhaustion::Restore:
                    // The window screen's drops of this search come back.
                    state.series.reset(context.pixel, state.search_mask);
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
    assert(state.init_model);   // a stable Initialize made it

    const LookbackJudgement judged = context.strategies.lookback->judge(
        *state.init_model, series.dates(), series.spectral(), window, state.cursor.prev_break,
        state.variogram, context.options.DETECTION_BANDS, lookback_scratch(context, state));

    switch (judged.verdict)
    {
        case LookbackVerdict::Change:
            return LookbackOutcome::Change;
        case LookbackVerdict::Drop:
            // The window moves down one position with the observations after
            // the dropped one.
            CCD_OBSERVE_EVENT(context.observe,
                              day_event(ObserverEvent::Kind::Drop,
                                        series.dates()(judged.observation), "lookback outlier"));
            series.drop(judged.observation, window);
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
    ChangeModel result = context.strategies.edge_fit->catch_window(
        state.series.dates(),
        state.series.spectral(),
        window_0,
        CurveQA::Start,
        edge_scratch(context, state)
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

// The extension's model, of the run's monitoring-model strategy, for a
// refit: the refit's number is the next.
Model& refit_model(const Context& context, Extension& extension)
{
    ++extension.fits;
    return extension.model.get(context.strategies.model,
                               [](const MonitoringModel& m) { return m.make_model(); });
}

// The bands no test reads (those not in DETECTION_BANDS), as the last test
// would have had them: fitted on the last refit's window with its
// coefficient count, their residuals over the last test's peek.
void complete_models(const Context& context, State& state)
{
    Extension& extension = state.extension;
    const auto masked_dates = state.series.dates();
    const auto masked_spectral = state.series.spectral();

    BandList others;
    for (index_t band = 0; band < masked_spectral.extent(0); ++band)
        if (!context.options.DETECTION_BANDS.contains(band))
            others.push_back(band);

    const Window& fit = extension.fit_window;
    const Window& peek = extension.peek;
    extension.model->fit(
        masked_dates.slice(range(fit.start, fit.stop)),
        masked_spectral.slice(all(), range(fit.start, fit.stop)),
        coefficient_count(fit.size(), context.options),
        context.options.COEFFICIENT_MAX,
        others,
        KeepResiduals::No
    );
    extension.model->peek_residuals(
        masked_dates.slice(range(peek.start, peek.stop)),
        masked_spectral.slice(all(), range(peek.start, peek.stop)),
        others,
        extension.residuals
    );
    extension.complete = true;
}

// Writes the segment Extend has grown, from its last peek test, and ends the
// extension. On a change (change probability 1) it breaks at the change
// test's break_at(); at the series end (0) at the first observation after
// the window, or its last when none is left. Its coefficients are the
// segment fit's; each band's magnitude is the change test's, or `magnitude`
// when given (a resumed series end, whose last test the stored run made).
void write_extended_segment(
    const Context& context,
    State& state,
    bool change,
    const std::vector<scalar_t>* magnitude = nullptr
)
{
    const MaskedSeries& series = state.series;
    const Window& window = state.cursor.window;
    Extension& extension = state.extension;

    assert(extension.trained);

    const auto masked_dates = series.dates();
    const index_t num_bands = series.spectral().extent(0);

    if (!extension.complete)
        complete_models(context, state);

    ChangeModel result;
    result.start_day = masked_dates(window.start);
    result.end_day = masked_dates(window.stop - 1);
    const ChangeTest& change_test = *context.strategies.change_test;
    const Scratch& scratch = test_scratch(context, extension);
    const Peek last = peek_of(context, state);
    result.break_day = masked_dates(change ? change_test.break_at(last, scratch)
                                           : std::min(window.stop, series.size() - 1));
    result.observation_count = window.stop - window.start;
    result.change_probability = change ? 1.0 : 0.0;

    const index_t num_coef = context.strategies.segment_fit->segment(
        *extension.model, masked_dates, series.spectral(), window, extension.num_coef,
        context.max_coefficients, segment_scratch(context, state), result.bands);
    result.curve_qa = static_cast<CurveQA>(num_coef);
    for (index_t band = 0; band < num_bands; ++band)
        result.bands[band].magnitude =
            magnitude ? (*magnitude)[band] : change_test.magnitude(last, scratch, band);

    CCD_OBSERVE_EVENT(context.observe, segment_event(result));
    state.segments.push_back(std::move(result));
    extension.trained = false;
}

// The segment of a forced start the series ended before Extend fitted: the
// segment fit's of its window (SegmentFit::unfitted()), through the last
// observation, change probability 0.
void write_forced_segment(const Context& context, State& state)
{
    const MaskedSeries& series = state.series;
    const auto masked_dates = series.dates();
    const Window window = *state.forced;
    state.forced.reset();

    ChangeModel result;
    result.start_day = masked_dates(window.start);
    result.end_day = masked_dates(series.size() - 1);
    result.break_day = result.end_day;
    result.observation_count = window.size();
    result.change_probability = 0.0;
    const index_t num_coef = context.strategies.segment_fit->unfitted(
        refit_model(context, state.extension), masked_dates, series.spectral(), window,
        coefficient_count(window.size(), context.options), context.max_coefficients,
        segment_scratch(context, state), result.bands);
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

ExtendOutcome extend_step(const Context& context, State& state)
{
    const auto& options = context.options;
    MaskedSeries& series = state.series;
    Window& window = state.cursor.window;
    const Cadence& cadence = state.cadence;
    Extension& extension = state.extension;

    const BandList& detection_bands = options.DETECTION_BANDS;

    // while model_window.stop + peek_size <= period.shape[0]:
    assert(peek_fits(state));

    //--------------------------------------------------------------------------
    // A window too small to test grows untested
    //--------------------------------------------------------------------------
    if (window.size() < cadence.min_window)
    {
        if (window.stop >= series.size())
            return ExtendOutcome::NoFit;
        window = Window(window.start, window.stop + 1);
        return ExtendOutcome::Grown;
    }

    if (!extension.trained)
    {
        LOG_DEBUG("initial model window extend: " << window.start << ", " << window.stop);
    }

    const auto masked_dates = series.dates();
    const auto masked_spectral = series.spectral();

    // Indexed by band.
    extension.residuals.resize(masked_spectral.extent(0));

    extension.num_coef = coefficient_count(window.size(), options);

    // The next `peek` observations, fewer at the series end.
    const Window& peek_window = extension.peek =
        Window(window.stop, std::min(window.stop + cadence.peek, series.size()));

    // # Used for comparison against fit_span
    auto model_span = span(masked_dates, window);

    LOG_DEBUG("detecting change for: " << peek_window.start << ", " << peek_window.stop);

    //--------------------------------------------------------------------------
    // The refit schedule
    //--------------------------------------------------------------------------
    if (!extension.trained ||
        context.strategies.refit->due(window.size(), model_span, extension.fit_window.size(),
                                      extension.fit_span))
    {
        // A refit on the window the last one grew from extends it.
        const bool extends = extension.trained;
        extension.trained = true;
        extension.fit_window = window;
        extension.fit_span = span(masked_dates, extension.fit_window);
        LOG_DEBUG("Retrain models");

        // The detection bands: seasonal_rmse() reads their residuals; the
        // magnitude is set from the peek residuals when the segment is
        // written, which fits the other bands (complete_models()).
        Model& model = refit_model(context, extension);
        const auto fit_dates =
            masked_dates.slice(range(extension.fit_window.start, extension.fit_window.stop));
        const auto fit_spectral =
            masked_spectral.slice(all(), range(extension.fit_window.start, extension.fit_window.stop));
        const bool fitted = extends
            ? model.extend(fit_dates, fit_spectral, extension.num_coef, options.COEFFICIENT_MAX,
                           detection_bands, KeepResiduals::Yes)
            : model.fit(fit_dates, fit_spectral, extension.num_coef, options.COEFFICIENT_MAX,
                        detection_bands, KeepResiduals::Yes);
        if (!fitted)
        {
            extension.trained = false;
            return ExtendOutcome::NoFit;
        }
        state.forced.reset();
    }
    extension.complete = false;

    // No observation left to test: the curve ends at the window.
    if (peek_window.size() == 0)
        return ExtendOutcome::SeriesEnd;

    //--------------------------------------------------------------------------
    // The peek's residuals, comparison RMSE and magnitudes
    //--------------------------------------------------------------------------

    extension.model->peek_residuals(
        masked_dates.slice(range(peek_window.start, peek_window.stop)),
        masked_spectral.slice(all(), range(peek_window.start, peek_window.stop)),
        detection_bands,
        extension.residuals
    );

    const Peek peek = peek_of(context, state);
    const TestOutcome outcome = context.strategies.change_test->test(
        peek, test_scratch(context, extension), extension.result);

    CCD_OBSERVE_EVENT(context.observe, [&] {
        ObserverEvent e = day_event(ObserverEvent::Kind::Test, masked_dates(peek_window.start),
                                    to_string(outcome));
        e.values = extension.result.statistic;
        return e;
    }());

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

    const OutlierOutcome judged = context.strategies.outlier->judge(peek, extension.result);
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
        if (window.stop + cadence.shortest > series.size() - 1)
            complete_models(context, state);
        CCD_OBSERVE_EVENT(context.observe, day_event(ObserverEvent::Kind::Drop,
                                                     masked_dates(peek_window.start), "outlier"));
        series.drop(peek_window.start, window);
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
    if (window.stop + cadence.shortest >= series.size())
        return ExtendOutcome::SeriesEnd;

    //--------------------------------------------------------------------------
    // Otherwise grow
    //--------------------------------------------------------------------------
    window = Window(window.start, window.stop + 1);
    return ExtendOutcome::Grown;
}

Phase extend(const Context& context, State& state)
{
    const ExtendOutcome outcome = extend_step(context, state);
    LOG_DEBUG("Extend: " << to_string(outcome));

    if (outcome == ExtendOutcome::Change)
    {
        // The change test's break may be past the peek's first observation:
        // the ones before it leave the series, after the segment is written.
        const index_t first = state.extension.peek.start;
        const index_t at = context.strategies.change_test->break_at(
            peek_of(context, state), test_scratch(context, state.extension));
        write_extended_segment(context, state, true);
        if (at > first)
            state.resumable = false;   // the record keeps no drops before a change
        for (index_t t = first; t < at; ++t)
        {
            CCD_OBSERVE_EVENT(context.observe,
                              day_event(ObserverEvent::Kind::Drop, state.series.dates()(first),
                                        "before change"));
            state.series.drop(first, state.cursor.window);
        }
        return Phase::Restart;
    }

    //--------------------------------------------------------------------------
    // No fit: the series ended before the window was fitted, or it could
    // not be. A forced start the series ended before Extend fitted reports
    // its own segment; otherwise the curve is not written. Not resumable.
    //--------------------------------------------------------------------------
    if (outcome == ExtendOutcome::NoFit)
    {
        if (state.forced)
            write_forced_segment(context, state);
        state.exit = Exit::NoFit;
        state.resumable = false;
        return Phase::Done;
    }

    if (outcome != ExtendOutcome::SeriesEnd && peek_fits(state))
        return Phase::Extend;

    //--------------------------------------------------------------------------
    // Series end: the curve reached the end of the series, after a test
    // whose peek reached the last observation, or a drop that left no full
    // peek. It is the last curve: no Restart, the End segment's guard reads
    // the curve's stop.
    //--------------------------------------------------------------------------
    suspend(state, Exit::SeriesEnd);
    state.suspended.fit_stop = state.extension.fit_window.stop;
    state.suspended.grow = outcome == ExtendOutcome::SeriesEnd;

    write_extended_segment(context, state, false);
    state.suspended.num_coef = state.extension.num_coef;
    for (const BandFit& band : state.segments.back().bands)
        state.suspended.magnitude.push_back(band.magnitude);

    state.cursor.prev_break = state.cursor.window.stop;
    state.exit = Exit::SeriesEnd;
    return Phase::EndCatch;
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
    ChangeModel result = context.strategies.edge_fit->catch_window(
        series.dates(),
        series.spectral(),
        window_1,
        CurveQA::End,
        edge_scratch(context, state)
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
    if (context.strategies.edge_fit->end_due(state.series.size() - prev_break,
                                             state.cadence.peek, after_change))
        write_end_segment(context, state);
    return Phase::Done;
}

//------------------------------------------------------------------------------
// The machine
//------------------------------------------------------------------------------

Phase Machine::step(const Context& context, State& state)
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

FitResult run(const Context& context, State& state, ProcessingMask mask)
{
    state.reset(context.pixel, std::move(mask));
    if (context.max_coefficients > 0)
        state.resumable = false;   // a resume does not cap
    drive<Machine>(context, state);
    report_exit(context, state);
    return state.take_result();
}

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
                refuse("NoRoomToInitialize with a window other than MEOW_SIZE long");
            break;
        case Exit::InitializeExhausted:
            if (w.size() < meow || w.stop > size)
                refuse("InitializeExhausted with a window shorter than MEOW_SIZE or past the series");
            break;
        case Exit::PeekBeyondEnd:
            if (w.size() < meow || w.stop > size)
                refuse("PeekBeyondEnd with a window shorter than MEOW_SIZE or past the series");
            break;
        case Exit::SeriesEnd:
            if (w.size() < meow || w.stop >= size)
                refuse("SeriesEnd with a window shorter than MEOW_SIZE or reaching the series end");
            if (!(w.start < from.fit_stop && from.fit_stop <= w.stop))
                refuse("SeriesEnd with a fit window outside the window");
            if (from.num_coef < 1 || from.num_coef > context.options.COEFFICIENT_MAX)
                refuse("SeriesEnd with a coefficient count outside 1..COEFFICIENT_MAX");
            if (from.magnitude.size() != state.series.spectral().extent(0))
                refuse("SeriesEnd with " + std::to_string(from.magnitude.size()) +
                       " magnitudes for " + std::to_string(state.series.spectral().extent(0)) +
                       " bands");
            break;
        default:
            refuse("an exit with nothing to resume");
    }
}

// resume()'s SeriesEnd: the open segment as the stored run left it, its
// models refitted on its fit window (as the run fitted them, bit for bit),
// then the guard that failed tested again on the longer series.
Phase resume_series_end(const Context& context, State& state, const Suspended& from)
{
    Extension& extension = state.extension;
    Window& window = state.cursor.window;
    const auto dates = state.series.dates();
    const auto spectral = state.series.spectral();

    extension.trained = true;
    extension.fit_window = Window(window.start, from.fit_stop);
    extension.fit_span = span(dates, extension.fit_window);
    BandList bands;
    for (index_t band = 0; band < spectral.extent(0); ++band)
        bands.push_back(band);
    refit_model(context, extension).fit(
        dates.slice(range(extension.fit_window.start, extension.fit_window.stop)),
        spectral.slice(all(), range(extension.fit_window.start, extension.fit_window.stop)),
        coefficient_count(extension.fit_window.size(), context.options),
        context.options.COEFFICIENT_MAX,
        bands,
        KeepResiduals::Yes
    );
    extension.complete = true;

    // After a grow the last peek reached the last observation: the series
    // continues once it holds one more, and the grow is applied (testing
    // that peek again would give the same outcome). After a drop no full
    // peek was left.
    const index_t reach = window.stop + state.cadence.peek;
    if (from.grow ? reach < state.series.size() : reach <= state.series.size())
    {
        if (from.grow)
            window = Window(window.start, window.stop + 1);
        return Phase::Extend;
    }

    // The series still ends here: the stored run's last test, its segment.
    extension.num_coef = from.num_coef;
    extension.peek = Window(window.stop, reach);
    write_extended_segment(context, state, false, &from.magnitude);
    state.cursor.prev_break = window.stop;
    state.exit = Exit::SeriesEnd;
    state.suspended = from;
    return Phase::EndCatch;
}

} // namespace

FitResult resume(
    const Context& context,
    State& state,
    ProcessingMask mask,
    const Suspended& from,
    const std::vector<std::int64_t>& excluded_days,
    bool carried_change
)
{
    state.reset(context.pixel, std::move(mask));

    // The statistics are the stored run's, and the series is no shorter, so
    // Prepare neither exits early nor computes other values: it reads the
    // series before any drop, as the stored run's did.
    const Phase prepared = prepare(context, state);
#if CCD_OBSERVE
    if (context.observe.observer)
    {
        ObserverEvent e = observed(state);
        e.kind = ObserverEvent::Kind::Transition;
        e.machine = Machine::name;
        e.phase = to_string(Phase::Prepare);
        e.next = to_string(prepared);
        context.observe.observer->event(context.observe.pixel, e);
    }
#endif
    if (prepared == Phase::Done)
    {
        report_exit(context, state);
        return state.take_result();
    }
    state.exit = Exit::None;
    state.suspended = Suspended{};

    // The stored run's drops, the latest first, so each earlier position
    // stays where the stored run found it.
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
        state.series.drop(static_cast<index_t>(at - begin), unused);
    }

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

    drive<Machine>(context, state);
    report_exit(context, state);
    return state.take_result();
}

FitResult Procedure::resume(
    const HarmonicWorkspace& hworkspace,
    ProcessingMask mask,
    LassoWorkspace& lworkspace,
    LassoSolver& solver,
    const Strategies& strategies,
    const Suspended& from,
    const std::vector<std::int64_t>& excluded_days,
    Observe observe,
    bool carried_change
) {
    const Context context{hworkspace.options(), hworkspace, lworkspace, solver, strategies,
                          observe};
    return standard::resume(context, state_, std::move(mask), from, excluded_days,
                            carried_change);
}

FitResult Detector::resume(
    const HarmonicWorkspace& hworkspace,
    ProcessingMask mask,
    LassoWorkspace& lworkspace,
    LassoSolver& solver,
    const Strategies& strategies,
    const Suspended& from,
    const std::vector<std::int64_t>& excluded_days,
    Observe observe,
    bool carried_change
)
{
    return procedure_.resume(hworkspace, std::move(mask), lworkspace, solver, strategies, from,
                             excluded_days, observe, carried_change);
}

FitResult Procedure::run(
    const HarmonicWorkspace& hworkspace,
    ProcessingMask mask,
    LassoWorkspace& lworkspace,
    LassoSolver& solver,
    const Strategies& strategies,
    Observe observe,
    index_t max_coefficients
) {
    const Context context{
        hworkspace.options(),
        hworkspace,
        lworkspace,
        solver,
        strategies,
        observe,
        max_coefficients
    };

    return standard::run(context, state_, std::move(mask));
}

const Suspended& Procedure::suspended() const
{
    static const Suspended none;
    return state_.resumable ? state_.suspended : none;
}

FitResult Detector::fit(
    const PixelMode& mode,
    const HarmonicWorkspace& hworkspace,
    ProcessingMask mask,
    LassoWorkspace& lworkspace,
    LassoSolver& solver,
    const Strategies& strategies,
    Observe observe
)
{
    if (mode.procedure == PixelProcedure::Standard)
    {
        LOG_DEBUG("Standard");
        return procedure_.run(hworkspace, std::move(mask), lworkspace, solver, strategies, observe,
                              mode.max_coefficients);
    }

    return strategies.edge_fit->whole_series(
        mode.procedure, hworkspace, std::move(mask),
        edges_.get(strategies.edge_fit, [](const EdgeFit& e) { return e.make_scratch(); }));
}

} // namespace standard

} // namespace ccd
