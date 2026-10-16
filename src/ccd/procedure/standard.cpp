#include "ccd/procedure/standard.hpp"

#include <algorithm>
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

Cadence measure_cadence(
    ArrayView<const std::int64_t, 1> statistics_dates,
    const HarmonicOptions& options
)
{
    Cadence cadence;

    cadence.peek =
        adjust_peek(
            statistics_dates,
            options.PEEK_SIZE,
            options.PEEK_REFERENCE_DAYS
        );

    // Compares the adjusted peek with the caller's PEEK_SIZE and starts from
    // the caller's CHANGE_THRESHOLD.
    cadence.change_threshold =
        adjust_change_threshold(
            cadence.peek,
            options.PEEK_SIZE,
            options.CHANGE_THRESHOLD,
            options.DETECTION_BANDS.size(),
            options.CHANGE_PROBABILITY
        );

    return cadence;
}

InitializeCriteria initialize_criteria(
    const HarmonicOptions& options,
    const Cadence& cadence
)
{
    return InitializeCriteria{
        options.MEOW_SIZE,
        options.DAY_DELTA,
        options.TMASK_BANDS,
        options.T_CONST,
        options.DETECTION_BANDS,
        4,   // pyccd's literal, not COEFFICIENT_MIN (quirk 4)
        cadence.change_threshold
    };
}

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

namespace
{

// One band's peek residuals, in place: the prediction, then |y - prediction|.
void peek_band_residuals(
    const HarmonicBasis& peek_basis,
    ArrayView<const scalar_t, 1> y,
    const LassoModel& model,
    std::vector<scalar_t>& residuals
)
{
    residuals.resize(y.size());

    model.predict(
        peek_basis.X(),
        residuals
    );

    for (index_t i = 0; i < y.size(); ++i)
    {
        residuals[i] =
            std::abs(y(i) - residuals[i]);
    }
}

} // namespace

void peek_residuals(
    const HarmonicBasis& peek_basis,
    ArrayView<const scalar_t, 2> peek_spectral,
    const std::vector<LassoResult>& models,
    const BandList& bands,
    std::vector<std::vector<scalar_t>>& residuals
)
{
    assert(peek_basis.samples() == peek_spectral.extent(1));

    for (const index_t band : bands)
    {
        peek_band_residuals(
            peek_basis,
            peek_spectral.slice(fixed(band), all()),
            models[band].model,
            residuals[band]
        );
    }
}

void peek_residuals(
    const HarmonicBasis& peek_basis,
    ArrayView<const scalar_t, 2> peek_spectral,
    const std::vector<LassoResult>& models,
    std::vector<std::vector<scalar_t>>& residuals
)
{
    assert(peek_basis.samples() == peek_spectral.extent(1));

    for (index_t band = 0; band < peek_spectral.extent(0); ++band)
    {
        peek_band_residuals(
            peek_basis,
            peek_spectral.slice(fixed(band), all()),
            models[band].model,
            residuals[band]
        );
    }
}

void change_magnitude(
    const std::vector<std::vector<scalar_t>>& residuals,
    const BandList& bands,
    const Variogram& variogram,
    const std::vector<scalar_t>& rmse,
    std::vector<scalar_t>& magnitude
)
{
    assert(rmse.size() >= bands.size());

    magnitude.clear();

    if (bands.empty())
        return;

    const index_t count =
        residuals[bands.front()].size();

    magnitude.assign(count, 0.0);
    for (index_t i = 0; i < bands.size(); ++i)
    {
        const scalar_t norm =
            std::max(
                variogram[bands[i]],
                rmse[i]
            );

        if (norm <= 0.0)
            continue;

        const auto& band_residuals = residuals[bands[i]];

        for (index_t j = 0; j < count; ++j)
        {
            const scalar_t value =
                band_residuals[j] / norm;

            magnitude[j] += value * value;
        }
    }
    LOG_DEBUG("Magnitudes of change: " << magnitude);
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

    cadence = Cadence{};
    extension.trained = false;
}

FitResult State::take_result()
{
    assert(phase == Phase::Done);

    // The early exits drop the processing mask (quirk 5).
    if (exit == Exit::TooFewObservations || exit == Exit::VariogramNaN)
        return {};

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

// The route into EndCatch, recording why the segment loop ended.
Phase end_catch_because(State& state, Exit why)
{
    state.exit = why;
    return Phase::EndCatch;
}

} // namespace

//------------------------------------------------------------------------------
// Routing
//------------------------------------------------------------------------------

bool room_to_initialize(const Context& context, const State& state)
{
    // window.stop <= observations - MEOW_SIZE, written without the
    // subtraction: index_t is unsigned, and observations < MEOW_SIZE would
    // wrap it to a huge bound.
    return state.cursor.window.stop + context.options.MEOW_SIZE <= state.series.size();
}

bool peek_fits(const State& state)
{
    return state.cursor.window.stop + state.cadence.peek <= state.series.size();
}

bool start_catch_due(const State& state)
{
    const Cursor& cursor = state.cursor;
    assert(cursor.window.start >= cursor.prev_break);

    return state.segments.empty() &&
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

    // validate() refuses less: Lookback's far peek reads the adjusted peek
    // minus one observations, at least PEEK_SIZE - 1, and both peeks read
    // their first magnitude.
    assert(options.PEEK_SIZE >= 2);

    LOG_DEBUG("num obs: " << context.pixel.dates().size());
    LOG_DEBUG("Processing mask initial count: " << series.size());
    //----------------------------------------------------------
    // Not enough observations
    //----------------------------------------------------------
    if (series.size() <= options.MEOW_SIZE)
    {
        LOG_DEBUG("Processing Mask: " << series.mask().data());
        LOG_DEBUG("mask count < meow size" << series.size());
        state.exit = Exit::TooFewObservations;
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
    state.cadence = measure_cadence(stat_dates, options);
    state.initialize_criteria = initialize_criteria(options, state.cadence);

    assert(state.cadence.peek >= options.PEEK_SIZE);
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
        return Phase::Done;
    }
    LOG_DEBUG("Variogram values: " << state.variogram.bands);

    state.cursor = Cursor{Window(0, options.MEOW_SIZE), 0};

    if (room_to_initialize(context, state))
        return Phase::Initialize;

    return end_catch_because(state, Exit::NoRoomToInitialize);
}

Phase initialize(const Context& context, State& state)
{
    LOG_DEBUG("Initialize for change model #: " << state.segments.size());

    state.init_models.resize(state.series.spectral().extent(0));

    const InitializeOutcome outcome = initialize_step(
        state.series,
        state.cursor.window,
        state.initialize_criteria,
        state.variogram,
        context.lasso,
        context.solver,
        state.init_models
    );
    LOG_DEBUG("Initialize: " << to_string(outcome));

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

    // Python:
    // slice(window.start - 1,
    //       window.start - peek_size,
    //       -1)
    const auto start = static_cast<std::ptrdiff_t>(window_start) - 1;

    if ((window_start - prev_break) > peek)
        return ReversePeek{start, static_cast<std::ptrdiff_t>(window_start) -
                                  static_cast<std::ptrdiff_t>(peek)};

    if (window_start <= peek)
        return ReversePeek{start, -1};

    return ReversePeek{start, static_cast<std::ptrdiff_t>(prev_break) - 1};
}

Phase lookback(const Context& context, State& state)
{
    const auto& options = context.options;
    MaskedSeries& series = state.series;
    Window& window = state.cursor.window;
    const index_t prev_break = state.cursor.prev_break;
    const std::vector<LassoResult>& models = state.init_models;
    const Cadence& cadence = state.cadence;
    HarmonicBasis& peek_basis = state.peek_basis;

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

    // The 8-coefficient basis, whatever the models' count (quirk 4).
    peek_basis.build(masked_dates_window, 8);

    peek_residuals(
        peek_basis,
        masked_spectral_window,
        models,
        detection_bands,
        residuals
    );

    // use original model rmse like python code
    // not sure if this is intended or python bug
    for (index_t i = 0; i < detection_bands.size(); ++i)
    {
        comp_rmses[i] = models[detection_bands[i]].score.rmse;
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
        LOG_DEBUG("Lookback: change before " << window.start);
        return after_lookback(context, state);
    }
    //--------------------------------------------------------------------------
    // Outlier detected: drop the observation immediately before the window;
    // the window moves down one position with the observations after it
    //--------------------------------------------------------------------------
    else if (detect_outlier(magnitude[0], options.OUTLIER_THRESHOLD))
    {
        LOG_DEBUG("Lookback: outlier dropped at " << peek_window.start);
        series.drop(static_cast<index_t>(peek_window.start), window);
    }
    //--------------------------------------------------------------------------
    // Otherwise include it
    //--------------------------------------------------------------------------
    else
    {
        LOG_DEBUG("Lookback: included " << peek_window.start);
        window.start = static_cast<index_t>(peek_window.start);
    }

    if (window.start > prev_break)
        return Phase::Lookback;

    return after_lookback(context, state);
}

Phase start_catch(const Context& context, State& state)
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

    return peek_guard(context, state);
}

namespace
{

// Writes the segment Extend has grown, from its last peek test, and ends the
// extension. change_probability is 1 on a change, 0 at the series end.
Phase write_extended_segment(State& state, scalar_t change_probability)
{
    const MaskedSeries& series = state.series;
    const Window& window = state.cursor.window;
    Extension& extension = state.extension;

    assert(extension.trained);

    const auto masked_dates = series.dates();
    const index_t num_bands = series.spectral().extent(0);

    ChangeModel result;
    result.start_day = masked_dates(window.start);
    result.end_day = masked_dates(window.stop - 1);
    result.break_day = masked_dates(extension.peek.start);
    result.observation_count = window.stop - window.start;
    result.change_probability = change_probability;
    result.curve_qa = static_cast<CurveQA>(extension.num_coef);

    result.bands = std::move(extension.models);
    for (index_t band = 0; band < num_bands; ++band) {
        result.bands[band].score.magn = median(extension.residuals[band]);

        // The fit-window residuals were for seasonal_rmse(); nothing reads
        // a returned segment's residuals.
        result.bands[band].score.residuals = std::vector<scalar_t>();
    }

    state.segments.push_back(std::move(result));
    extension.trained = false;
    return Phase::Restart;
}

} // namespace

Phase extend(const Context& context, State& state)
{
    const auto& options = context.options;
    MaskedSeries& series = state.series;
    Window& window = state.cursor.window;
    const Cadence& cadence = state.cadence;
    const Variogram& variogram = state.variogram;
    HarmonicBasis& peek_basis = state.peek_basis;
    LassoWorkspace& lworkspace = context.lasso;
    LassoSolver& solver = context.solver;
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
    // Indexed by position in DETECTION_BANDS.
    extension.comp_rmse.resize(detection_bands.size());

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
        window.stop - window.start < options.RETRAIN_MIN_OBS ||
        model_span >= options.RETRAIN_SPAN_FACTOR * extension.fit_span)
    {
        extension.trained = true;
        extension.fit_window = window;
        extension.fit_span = span(masked_dates, extension.fit_window);
        LOG_DEBUG("Retrain models");

        // seasonal_rmse() reads the residuals; the magnitude is set from the
        // peek residuals when the segment is written.
        fit_bands(
            masked_dates.slice(range(extension.fit_window.start, extension.fit_window.stop)),
            masked_spectral.slice(all(), range(extension.fit_window.start, extension.fit_window.stop)),
            extension.num_coef,
            KeepResiduals::Yes,
            lworkspace,
            solver,
            extension.models
        );
    }

    //--------------------------------------------------------------------------
    // The peek's residuals, comparison RMSE and magnitudes
    //--------------------------------------------------------------------------

    // The 8-coefficient basis, whatever the models' count (quirk 4).
    peek_basis.build(masked_dates.slice(range(peek_window.start, peek_window.stop)), 8);

    peek_residuals(
        peek_basis,
        masked_spectral.slice(all(), range(peek_window.start, peek_window.stop)),
        extension.models,
        extension.residuals
    );

    std::vector<scalar_t>& comp_rmses = extension.comp_rmse;
    if (window.stop - window.start <= options.RETRAIN_MIN_OBS) {
        LOG_DEBUG("normal residuals and rmse");
        for (index_t i = 0; i < detection_bands.size(); ++i)
        {
            comp_rmses[i] = extension.models[detection_bands[i]].score.rmse;
        }
    } else {

        LOG_DEBUG("closest day residuals and rmse");
        auto closest_indexes =
            find_closest_doy(masked_dates, peek_window.stop - 1, extension.fit_window, options.SEASONAL_RMSE_COUNT);

        LOG_DEBUG("period: " << masked_dates);
        LOG_DEBUG("peek window: (" << peek_window.start << ", " << peek_window.stop << ")");
        LOG_DEBUG("fit window: (" << extension.fit_window.start << ", " << extension.fit_window.stop << ")");
        LOG_DEBUG("closest inds: " << closest_indexes);

        for (index_t i = 0; i < detection_bands.size(); ++i)
        {
            comp_rmses[i] = seasonal_rmse(extension.models[detection_bands[i]], closest_indexes, options.SEASONAL_RMSE_DIVISOR);
        }
    }

    change_magnitude(
        extension.residuals,
        detection_bands,
        variogram,
        comp_rmses,
        extension.magnitude
    );

    //--------------------------------------------------------------------------
    // Change detected: the segment ends at the window
    //--------------------------------------------------------------------------
    if (detect_change(extension.magnitude, cadence.change_threshold))
    {
        LOG_DEBUG("Change detected at: " << peek_window.start);
        return write_extended_segment(state, 1.0);
    }
    //--------------------------------------------------------------------------
    // Outlier detected: drop the first peek observation, at window.stop; the
    // window keeps its coordinates
    //--------------------------------------------------------------------------
    else if (detect_outlier(extension.magnitude[0], options.OUTLIER_THRESHOLD))
    {
        LOG_DEBUG("Outlier detected at: " << peek_window.start);
        LOG_DEBUG("mag: " << extension.magnitude[0] << " thresh: " << options.OUTLIER_THRESHOLD);
        LOG_DEBUG("Updating processing mask extend");
        series.drop(peek_window.start, window);
    }
    //--------------------------------------------------------------------------
    // Otherwise grow. pyccd checks for the series end first (procedures.py
    // 691-692), which cannot hold: the peek fitted at the top of this step
    // and nothing since changed the window or the series (quirk 6a).
    //--------------------------------------------------------------------------
    else
    {
        assert(peek_fits(state));
        window = Window(window.start, window.stop + 1);
    }

    if (peek_fits(state))
        return Phase::Extend;

    //--------------------------------------------------------------------------
    // Series end: no full peek after the window
    //--------------------------------------------------------------------------
    return write_extended_segment(state, 0.0);
}

Phase restart(const Context& context, State& state)
{
    Cursor& cursor = state.cursor;

    cursor.prev_break = cursor.window.stop;
    cursor.window = Window(cursor.window.stop, cursor.window.stop + context.options.MEOW_SIZE);

    if (room_to_initialize(context, state))
        return Phase::Initialize;

    return end_catch_because(state, Exit::NoRoomToInitialize);
}

Phase end_catch(const Context& context, State& state)
{
    const MaskedSeries& series = state.series;
    const index_t prev_break = state.cursor.prev_break;

    if (prev_break + state.cadence.peek < series.size()) {
        auto window_1 = Window(prev_break, series.size());
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

} // namespace standard

FitResult StandardProcedure::run(
    const HarmonicWorkspace& hworkspace,
    ProcessingMask mask,
    LassoWorkspace& lworkspace,
    LassoSolver& solver
) {
    const standard::Context context{
        hworkspace.options(),
        hworkspace,
        lworkspace,
        solver
    };

    return standard::run(context, state_, std::move(mask));
}

namespace standard
{

FitResult Detector::fit(
    PixelProcedure procedure,
    const HarmonicWorkspace& hworkspace,
    ProcessingMask mask,
    LassoWorkspace& lworkspace,
    LassoSolver& solver
)
{
    if (procedure == PixelProcedure::Standard)
    {
        LOG_DEBUG("Standard");
        return standard_.run(hworkspace, std::move(mask), lworkspace, solver);
    }

    return fit_whole_series(procedure, hworkspace, std::move(mask), lworkspace, solver);
}

} // namespace standard

} // namespace ccd
