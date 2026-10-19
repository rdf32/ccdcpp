#include "ccd/procedure/standard.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>
#include <cassert>
#include <cmath>
#include <string>
#include <utility>

#include "ccd/maths.hpp"
#include "ccd/logger.hpp"

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

namespace
{

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
    return Peek{*extension.model, state.series.dates(), state.cursor.window,
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
    return state.cursor.window.stop + state.cadence.peek <= state.series.size();
}

bool start_catch_due(const State& state)
{
    const Cursor& cursor = state.cursor;
    assert(cursor.window.start >= cursor.prev_break);

    return state.segments.empty() && state.carried == 0 &&
        cursor.window.start - cursor.prev_break > state.cadence.peek;
}

Phase after_stable_start(const Context& context, State& state)
{
    if (state.cursor.window.start > state.cursor.prev_break)
        return Phase::Lookback;

    return after_lookback(context, state);
}

Phase after_lookback(const Context& context, State& state)
{
    if (start_catch_due(state))
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
    state.cadence.peek = context.strategies.change_test->prepare(
        stat_dates, detection_bands, test_scratch(context, state.extension));
    state.cadence.change_threshold = state.thresholds(
        state.cadence.peek, options.PEEK_SIZE, options.CHANGE_THRESHOLD, detection_bands,
        options.CHANGE_PROBABILITY);
    context.strategies.stability_rule->prepare(
        state.cadence.peek, detection_bands, stability_scratch(context, state));
    state.initialize_criteria = initialize_criteria(options);

    LOG_DEBUG("Peek size: " << state.cadence.peek);
    LOG_DEBUG("Chng thresh: " << state.cadence.change_threshold);

    state.variogram.bands =
        adjusted_variogram(
            stat_dates,
            stat_spectral,
            options.VARIOGRAM_MIN_LAG_DAYS
        );

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
        return Phase::Initialize;

    return end_catch_because(state, Exit::NoRoomToInitialize);
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
            return end_catch_because(state, Exit::InitializeExhausted);

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

ReversePeek lookback_peek(index_t window_start, index_t prev_break, index_t peek)
{
    assert(window_start > prev_break);

    const auto start = static_cast<std::ptrdiff_t>(window_start) - 1;

    // `peek` observations, start - 1 down to start - peek, all above
    // prev_break.
    if ((window_start - prev_break) > peek)
        return ReversePeek{start, static_cast<std::ptrdiff_t>(window_start) -
                                  static_cast<std::ptrdiff_t>(peek) - 1};

    // Near the break: start - 1 down to prev_break, never below it.
    return ReversePeek{start, static_cast<std::ptrdiff_t>(prev_break) - 1};
}

LookbackOutcome lookback_step(const Context& context, State& state)
{
    const auto& options = context.options;
    MaskedSeries& series = state.series;
    Window& window = state.cursor.window;
    const index_t prev_break = state.cursor.prev_break;
    assert(state.init_model);   // a stable Initialize made it
    Model& model = *state.init_model;
    const Cadence& cadence = state.cadence;

    const BandList& detection_bands = options.DETECTION_BANDS;

    std::vector<std::int64_t>& peek_dates = state.lookback.peek_dates;
    std::vector<scalar_t>& peek_spectral = state.lookback.peek_spectral;

    // Indexed by band; only the detection bands' are written.
    std::vector<std::vector<scalar_t>>& residuals = state.lookback.residuals;
    residuals.resize(series.spectral().extent(0));
    // Indexed by position in DETECTION_BANDS.
    std::vector<scalar_t>& comp_rmses = state.lookback.comp_rmse;
    comp_rmses.resize(detection_bands.size());
    std::vector<scalar_t>& magnitude = state.lookback.magnitude;

    assert(window.start > prev_break);

    const auto masked_dates = series.dates();
    const auto masked_spectral = series.spectral();

    const ReversePeek peek_window =
        lookback_peek(window.start, prev_break, cadence.peek);

    peek_dates.clear();
    peek_dates.reserve(cadence.peek);
    for (std::ptrdiff_t i = peek_window.start; i > peek_window.stop; --i)
    {
        peek_dates.push_back(masked_dates[i]);
    }

    auto masked_dates_window =
        ArrayView<const std::int64_t, 1>::contiguous(
            peek_dates.data(),
            {peek_dates.size()}
        );

    const index_t bands   = masked_spectral.extent(0);
    const index_t samples = peek_dates.size();

    peek_spectral.resize(bands * samples);
    for (index_t band = 0; band < bands; ++band)
    {
        for (index_t j = 0; j < samples; ++j)
        {
            std::ptrdiff_t source =
                peek_window.start - static_cast<std::ptrdiff_t>(j);

            assert(source >= 0);
            assert(source < static_cast<std::ptrdiff_t>(masked_spectral.extent(1)));

            peek_spectral[band * samples + j] =
                masked_spectral(band, static_cast<index_t>(source));
        }
    }
    auto masked_spectral_window =
        ArrayView<const scalar_t, 2>::contiguous(
            peek_spectral.data(),
            {bands, samples}
        );

    model.peek_residuals(
        masked_dates_window,
        masked_spectral_window,
        detection_bands,
        residuals
    );

    // The comparison RMSE is the stable start window's: nothing is refitted
    // while the window grows backward (Zhu's TrendSeasonalFit, pyccd).
    for (index_t i = 0; i < detection_bands.size(); ++i)
    {
        comp_rmses[i] = model.rmse(detection_bands[i]);
    }

    change_magnitude(
        residuals,
        detection_bands,
        state.variogram,
        comp_rmses,
        magnitude
    );

    //--------------------------------------------------------------------------
    // Change detected: the window stays where it is
    //--------------------------------------------------------------------------
    if (detect_change(magnitude, cadence.change_threshold))
    {
        LOG_DEBUG("Change before " << window.start);
        return LookbackOutcome::Change;
    }
    //--------------------------------------------------------------------------
    // Outlier detected: drop the observation immediately before the window;
    // the window moves down one position with the observations after it
    //--------------------------------------------------------------------------
    else if (detect_outlier(magnitude[0], options.OUTLIER_THRESHOLD))
    {
        LOG_DEBUG("Outlier dropped at " << peek_window.start);
        series.drop(static_cast<index_t>(peek_window.start), window);
        return LookbackOutcome::Dropped;
    }
    //--------------------------------------------------------------------------
    // Otherwise include it
    //--------------------------------------------------------------------------
    LOG_DEBUG("Included " << peek_window.start);
    window.start = static_cast<index_t>(peek_window.start);
    return LookbackOutcome::Included;
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
    ChangeModel result = catch_segment(
        state.series.dates(),
        state.series.spectral(),
        window_0,
        context.options.COEFFICIENT_MIN,
        CurveQA::Start,
        context.lasso,
        context.solver
    );
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
// extension. change_probability is 1 on a change, 0 at the series end. Each
// band's magnitude is the median of its last peek residuals, or `magnitude`
// when given (a resumed series end, whose last test the stored run made).
void write_extended_segment(
    const Context& context,
    State& state,
    scalar_t change_probability,
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
    const Peek last = peek_of(context, state);
    result.break_day = masked_dates(change_test.break_at(last));
    result.observation_count = window.stop - window.start;
    result.change_probability = change_probability;
    result.curve_qa = static_cast<CurveQA>(extension.num_coef);

    result.bands.reserve(num_bands);
    for (index_t band = 0; band < num_bands; ++band) {
        BandFit fit = extension.model->band_fit(band);
        fit.magnitude = magnitude ? (*magnitude)[band] : change_test.magnitude(last, band);
        result.bands.push_back(fit);
    }

    state.segments.push_back(std::move(result));
    extension.trained = false;
}

} // namespace

ExtendOutcome extend_step(const Context& context, State& state)
{
    const auto& options = context.options;
    MaskedSeries& series = state.series;
    Window& window = state.cursor.window;
    const Cadence& cadence = state.cadence;
    const Variogram& variogram = state.variogram;
    Extension& extension = state.extension;

    const BandList& detection_bands = options.DETECTION_BANDS;

    // while model_window.stop + peek_size <= period.shape[0]:
    assert(peek_fits(state));

    if (!extension.trained)
    {
        LOG_DEBUG("initial model window extend: " << window.start << ", " << window.stop);
    }

    const auto masked_dates = series.dates();
    const auto masked_spectral = series.spectral();

    // Indexed by band.
    extension.residuals.resize(masked_spectral.extent(0));

    extension.num_coef = coefficient_count(window.size(), options);

    const Window& peek_window = extension.peek =
        Window(window.stop, window.stop + cadence.peek);

    // # Used for comparison against fit_span
    auto model_span = span(masked_dates, window);

    LOG_DEBUG("detecting change for: " << peek_window.start << ", " << peek_window.stop);

    //--------------------------------------------------------------------------
    // The refit schedule
    //--------------------------------------------------------------------------
    if (!extension.trained ||
        context.strategies.refit->due(window.stop - window.start, model_span, extension.fit_span))
    {
        extension.trained = true;
        extension.fit_window = window;
        extension.fit_span = span(masked_dates, extension.fit_window);
        LOG_DEBUG("Retrain models");

        // The detection bands: seasonal_rmse() reads their residuals; the
        // magnitude is set from the peek residuals when the segment is
        // written, which fits the other bands (complete_models()).
        refit_model(context, extension).fit(
            masked_dates.slice(range(extension.fit_window.start, extension.fit_window.stop)),
            masked_spectral.slice(all(), range(extension.fit_window.start, extension.fit_window.stop)),
            extension.num_coef,
            options.COEFFICIENT_MAX,
            detection_bands,
            KeepResiduals::Yes
        );
    }
    extension.complete = false;

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

    //--------------------------------------------------------------------------
    // Change detected: the segment ends at the window
    //--------------------------------------------------------------------------
    if (outcome == TestOutcome::Change)
    {
        LOG_DEBUG("Change detected at: " << peek_window.start);
        return ExtendOutcome::Change;
    }
    //--------------------------------------------------------------------------
    // Outlier detected: drop the first peek observation, at window.stop; the
    // window keeps its coordinates
    //--------------------------------------------------------------------------
    else if (context.strategies.outlier->judge(peek, extension.result) == OutlierOutcome::Drop)
    {
        LOG_DEBUG("Outlier detected at: " << peek_window.start);
        LOG_DEBUG("Updating processing mask extend");
        // A drop that leaves no full peek ends the curve at this test: its
        // other bands are this test's, before the drop moves the peek.
        if (window.stop + cadence.peek > series.size() - 1)
            complete_models(context, state);
        series.drop(peek_window.start, window);
        return ExtendOutcome::Dropped;
    }
    //--------------------------------------------------------------------------
    // The peek reaches the last observation and holds no change: the curve
    // ends at the window, its last test a full peek
    //--------------------------------------------------------------------------
    assert(peek_fits(state));
    if (window.stop + cadence.peek >= series.size())
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
        write_extended_segment(context, state, 1.0);
        return Phase::Restart;
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

    write_extended_segment(context, state, 0.0);
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
        return Phase::Initialize;

    return end_catch_because(state, Exit::NoRoomToInitialize);
}

void write_end_segment(const Context& context, State& state)
{
    const MaskedSeries& series = state.series;
    auto window_1 = Window(state.cursor.prev_break, series.size());
    ChangeModel result = catch_segment(
        series.dates(),
        series.spectral(),
        window_1,
        context.options.COEFFICIENT_MIN,
        CurveQA::End,
        context.lasso,
        context.solver
    );
    state.segments.push_back(std::move(result));
}

Phase end_catch(const Context& context, State& state)
{
    if (state.cursor.prev_break + state.cadence.peek < state.series.size())
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
    drive<Machine>(context, state);
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
    write_extended_segment(context, state, 0.0, &from.magnitude);
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
    const std::vector<std::int64_t>& excluded_days
)
{
    state.reset(context.pixel, std::move(mask));

    // The statistics are the stored run's, and the series is no shorter, so
    // Prepare neither exits early nor computes other values: it reads the
    // series before any drop, as the stored run's did.
    if (prepare(context, state) == Phase::Done)
        return state.take_result();
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
    state.cursor = Cursor{from.window, from.prev_break};

    switch (from.exit)
    {
        case Exit::NoRoomToInitialize:
            state.phase = room_to_initialize(context, state)
                ? Phase::Initialize
                : end_catch_because(state, Exit::NoRoomToInitialize);
            break;
        case Exit::InitializeExhausted:
            state.phase = Phase::Initialize;
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
    return state.take_result();
}

FitResult Procedure::resume(
    const HarmonicWorkspace& hworkspace,
    ProcessingMask mask,
    LassoWorkspace& lworkspace,
    LassoSolver& solver,
    const Strategies& strategies,
    const Suspended& from,
    const std::vector<std::int64_t>& excluded_days
) {
    const Context context{hworkspace.options(), hworkspace, lworkspace, solver, strategies};
    return standard::resume(context, state_, std::move(mask), from, excluded_days);
}

FitResult Detector::resume(
    const HarmonicWorkspace& hworkspace,
    ProcessingMask mask,
    LassoWorkspace& lworkspace,
    LassoSolver& solver,
    const Strategies& strategies,
    const Suspended& from,
    const std::vector<std::int64_t>& excluded_days
)
{
    return procedure_.resume(hworkspace, std::move(mask), lworkspace, solver, strategies, from,
                             excluded_days);
}

FitResult Procedure::run(
    const HarmonicWorkspace& hworkspace,
    ProcessingMask mask,
    LassoWorkspace& lworkspace,
    LassoSolver& solver,
    const Strategies& strategies
) {
    const Context context{
        hworkspace.options(),
        hworkspace,
        lworkspace,
        solver,
        strategies
    };

    return standard::run(context, state_, std::move(mask));
}

FitResult Detector::fit(
    PixelProcedure procedure,
    const HarmonicWorkspace& hworkspace,
    ProcessingMask mask,
    LassoWorkspace& lworkspace,
    LassoSolver& solver,
    const Strategies& strategies
)
{
    if (procedure == PixelProcedure::Standard)
    {
        LOG_DEBUG("Standard");
        return procedure_.run(hworkspace, std::move(mask), lworkspace, solver, strategies);
    }

    return fit_whole_series(procedure, hworkspace, std::move(mask), lworkspace, solver);
}

} // namespace standard

} // namespace ccd
