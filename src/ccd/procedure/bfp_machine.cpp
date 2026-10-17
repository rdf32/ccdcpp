#include "ccd/procedure/bfp_machine.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <utility>

#include "ccd/constants.hpp"
#include "ccd/logger.hpp"
#include "ccd/maths.hpp"
#include "ccd/procedure/machine.hpp"

namespace ccd
{
namespace bfp
{

const char* to_string(Phase phase)
{
    switch (phase)
    {
        case Phase::Prepare:     return "Prepare";
        case Phase::Initialize:  return "Initialize";
        case Phase::Lookback:    return "Lookback";
        case Phase::StartCatch:  return "StartCatch";
        case Phase::Lookforward: return "Lookforward";
        case Phase::Restart:     return "Restart";
        case Phase::EndCatch:    return "EndCatch";
        case Phase::Done:        return "Done";
    }
    return "?";
}

const char* to_string(LookforwardOutcome outcome)
{
    switch (outcome)
    {
        case LookforwardOutcome::Change:    return "Change";
        case LookforwardOutcome::SeriesEnd: return "SeriesEnd";
        case LookforwardOutcome::Dropped:   return "Dropped";
        case LookforwardOutcome::Appended:  return "Appended";
    }
    return "?";
}

//------------------------------------------------------------------------------
// LeastSquares
//------------------------------------------------------------------------------

namespace
{

// The trend column's unit: years from the segment's first day.
constexpr scalar_t YEAR = 365.25;

} // namespace

void LeastSquares::reset(std::int64_t origin, index_t bands)
{
    origin_ = origin;
    bands_ = bands;
    n_ = 0;
    q_ = 0;
    xtx_.fill(0.0);
    xty_.assign(bands * COLUMNS, 0.0);
    yty_.assign(bands, 0.0);
    beta_.assign(bands * COLUMNS, 0.0);
    ssr_.assign(bands, 0.0);
}

std::array<scalar_t, LeastSquares::COLUMNS> LeastSquares::row(std::int64_t day) const
{
    // The harmonic columns as HarmonicBasis computes them.
    const scalar_t omega = 2.0 * PI / AVG_DAYS_YR;
    const scalar_t wt = omega * static_cast<scalar_t>(day);
    const scalar_t w2t = 2.0 * wt;
    const scalar_t w3t = 3.0 * wt;
    return {1.0, static_cast<scalar_t>(day - origin_) / YEAR,
            std::cos(wt), std::sin(wt), std::cos(w2t), std::sin(w2t), std::cos(w3t), std::sin(w3t)};
}

void LeastSquares::add(std::int64_t day, ArrayView<const scalar_t, 2> spectral, index_t t)
{
    const auto x = row(day);
    for (index_t i = 0; i < COLUMNS; ++i)
        for (index_t j = 0; j <= i; ++j)
            xtx_[i * COLUMNS + j] += x[i] * x[j];
    for (index_t b = 0; b < bands_; ++b)
    {
        const scalar_t y = spectral(b, t);
        for (index_t i = 0; i < COLUMNS; ++i)
            xty_[b * COLUMNS + i] += x[i] * y;
        yty_[b] += y * y;
    }
    ++n_;
}

bool LeastSquares::fit(index_t q)
{
    assert(q >= 1 && q <= COLUMNS);
    q_ = q;

    // X'X[q,q] = L L' (the lower triangle of the sums).
    for (index_t i = 0; i < q; ++i)
    {
        for (index_t j = 0; j <= i; ++j)
        {
            scalar_t s = xtx_[i * COLUMNS + j];
            for (index_t p = 0; p < j; ++p)
                s -= chol_[i * COLUMNS + p] * chol_[j * COLUMNS + p];
            if (i == j)
            {
                if (!(s > 0))
                    return false;
                chol_[i * COLUMNS + i] = std::sqrt(s);
            }
            else
            {
                chol_[i * COLUMNS + j] = s / chol_[j * COLUMNS + j];
            }
        }
    }

    // Per band L z = X'y, L' beta = z; SSR = y'y - |z|^2.
    for (index_t b = 0; b < bands_; ++b)
    {
        std::array<scalar_t, COLUMNS> z{};
        scalar_t zz = 0.0;
        for (index_t i = 0; i < q; ++i)
        {
            scalar_t v = xty_[b * COLUMNS + i];
            for (index_t p = 0; p < i; ++p)
                v -= chol_[i * COLUMNS + p] * z[p];
            z[i] = v / chol_[i * COLUMNS + i];
            zz += z[i] * z[i];
        }
        for (index_t i = q; i-- > 0;)
        {
            scalar_t v = z[i];
            for (index_t p = i + 1; p < q; ++p)
                v -= chol_[p * COLUMNS + i] * beta_[b * COLUMNS + p];
            beta_[b * COLUMNS + i] = v / chol_[i * COLUMNS + i];
        }
        for (index_t i = q; i < COLUMNS; ++i)
            beta_[b * COLUMNS + i] = 0.0;
        ssr_[b] = std::max(scalar_t{0}, yty_[b] - zz);
    }
    return true;
}

scalar_t LeastSquares::residual(index_t band, std::int64_t day, scalar_t y) const
{
    const auto x = row(day);
    scalar_t fitted = 0.0;
    for (index_t i = 0; i < q_; ++i)
        fitted += x[i] * beta_[band * COLUMNS + i];
    return y - fitted;
}

void LeastSquares::whitened(std::int64_t day, scalar_t* out) const
{
    const auto x = row(day);
    for (index_t i = 0; i < q_; ++i)
    {
        scalar_t v = x[i];
        for (index_t p = 0; p < i; ++p)
            v -= chol_[i * COLUMNS + p] * out[p];
        out[i] = v / chol_[i * COLUMNS + i];
    }
}

BandFit LeastSquares::band_fit(index_t band, scalar_t rmse, scalar_t magnitude) const
{
    const scalar_t* beta = &beta_[band * COLUMNS];
    BandFit f;
    const scalar_t slope = q_ >= 2 ? beta[1] / YEAR : 0.0;
    f.intercept = beta[0] - slope * static_cast<scalar_t>(origin_);
    f.coefficients[0] = slope;
    for (index_t i = 2; i < q_; ++i)
        f.coefficients[i - 1] = beta[i];
    f.rmse = rmse;
    f.magnitude = magnitude;
    f.iterations = 0;
    return f;
}

//------------------------------------------------------------------------------
// State
//------------------------------------------------------------------------------

void State::reset(const HarmonicWorkspace& workspace, ProcessingMask mask)
{
    phase = Phase::Prepare;
    core.reset(workspace, std::move(mask));
    extension.trained = false;
}

std::string summary(const State& state)
{
    return standard::summary(state.core);
}

//------------------------------------------------------------------------------
// Lookforward
//------------------------------------------------------------------------------

namespace
{

// r / sqrt(MSE), with a zero MSE read as no spread: 0 for a zero residual,
// infinite otherwise (the test's p is then 0).
scalar_t scaled(scalar_t r, scalar_t mse)
{
    if (mse > 0)
        return r / std::sqrt(mse);
    if (r == 0)
        return 0.0;
    return std::copysign(std::numeric_limits<scalar_t>::infinity(), r);
}

// r^2 / MSE, with a zero MSE read as scaled() does.
scalar_t squared_ratio(scalar_t r, scalar_t mse)
{
    if (mse > 0)
        return r * r / mse;
    return r == 0 ? 0.0 : std::numeric_limits<scalar_t>::infinity();
}

// JuliaCCD's selectBFP over the first m peek observations: the start of the
// run of observations with P({j}) below the threshold that ends the m, when
// it holds at least `minimum`; otherwise 0.
index_t select_start(Mixture& mixture, index_t m, scalar_t threshold, index_t minimum)
{
    std::vector<bool> plausible(m);
    for (index_t j = 0; j < m; ++j)
        plausible[j] = mixture.single(j) < threshold;
    for (index_t i = 0; i + minimum <= m; ++i)
    {
        index_t count = 0;
        bool all = true;
        for (index_t j = i; j < m; ++j)
        {
            count += plausible[j];
            all = all && plausible[j];
        }
        if (count < minimum)
            return 0;
        if (all)
            return i;
    }
    return 0;
}

} // namespace

LookforwardOutcome lookforward_step(const Context& context, State& state)
{
    const HarmonicOptions& options = context.options;
    const BfpOptions& bo = options.BFP;
    standard::State& core = state.core;
    MaskedSeries& series = core.series;
    Window& window = core.cursor.window;
    Extension& ext = state.extension;

    const auto dates = series.dates();
    const auto spectral = series.spectral();
    const index_t B = spectral.extent(0);
    const BandList& detection = options.DETECTION_BANDS;
    const index_t D = detection.size();

    //--------------------------------------------------------------------------
    // The window's fit, refitted whenever it holds new observations.
    //--------------------------------------------------------------------------
    if (!ext.trained)
    {
        ext.sums.reset(dates(window.start), B);
        for (index_t t = window.start; t < window.stop; ++t)
            ext.sums.add(dates(t), spectral, t);
        ext.trained = true;
        ext.terms = 0;
    }
    const index_t n = ext.sums.count();
    if (ext.terms == 0 || n != ext.fitted)
    {
        const index_t q = standard::coefficient_count(n, options);
        if (!ext.sums.fit(q))
            return LookforwardOutcome::SeriesEnd;
        ext.terms = q;
        ext.fitted = n;
        ext.mse.resize(D);
        for (index_t d = 0; d < D; ++d)
            ext.mse[d] = ext.sums.ssr(detection[d]) / static_cast<scalar_t>(n - q);
    }
    const index_t q = ext.terms;

    //--------------------------------------------------------------------------
    // The peek: the next MAX_PEEK observations, fewer at the series end.
    //--------------------------------------------------------------------------
    const index_t t0 = window.stop;
    const index_t N = std::min(bo.MAX_PEEK, series.size() - t0);
    if (N == 0)
        return LookforwardOutcome::SeriesEnd;

    ext.peek_dates.resize(N);
    ext.residuals.resize(D * N);
    ext.whitened.resize(N * q);
    Mixture::Inputs in;
    in.h.assign(N * N, 0.0);
    in.u.resize(D * N);
    in.bands = D;
    for (index_t j = 0; j < N; ++j)
    {
        const std::int64_t day = dates(t0 + j);
        ext.peek_dates[j] = day;
        ext.sums.whitened(day, &ext.whitened[j * q]);
        for (index_t d = 0; d < D; ++d)
        {
            const scalar_t r = ext.sums.residual(detection[d], day, spectral(detection[d], t0 + j));
            ext.residuals[d * N + j] = r;
            in.u[d * N + j] = scaled(r, ext.mse[d]);
        }
    }
    // H = I + G'G, G = L^-1 X_p': the prediction errors' covariance.
    for (index_t i = 0; i < N; ++i)
    {
        for (index_t j = 0; j <= i; ++j)
        {
            scalar_t v = i == j ? 1.0 : 0.0;
            for (index_t p = 0; p < q; ++p)
                v += ext.whitened[i * q + p] * ext.whitened[j * q + p];
            in.h[i * N + j] = v;
            in.h[j * N + i] = v;
        }
    }
    in.dates = ext.peek_dates;
    in.degrees_of_freedom = static_cast<scalar_t>(n - q);
    in.good = bo.GOOD;
    in.min_days = bo.MIN_DAYS_BETWEEN;
    ext.mixture.reset(std::move(in));

    //--------------------------------------------------------------------------
    // The test, one peek size at a time.
    //--------------------------------------------------------------------------
    Rule rule;
    rule.threshold = bo.THRESHOLD;
    rule.scale_threshold = bo.SCALE_THRESHOLD;
    rule.no_change_cutoff = bo.NO_CHANGE_CUTOFF;
    rule.min_observations = bo.MIN_OBSERVATIONS_FOR_BREAK;
    rule.min_days_for_break = bo.MIN_DAYS_FOR_BREAK;
    rule.max_days_for_break = bo.MAX_DAYS_FOR_BREAK;

    Decision decision = Decision::Continue;
    index_t m = 0;
    while (m < N && decision == Decision::Continue)
    {
        const Mixture::Stats s = ext.mixture.add();
        ++m;
        decision = decide(ext.mixture, s, ext.peek_dates[m - 1] - ext.peek_dates[0], rule);
    }

    if (decision == Decision::Break)
    {
        const index_t start = bo.BREAK_START == BreakStart::Select
            ? select_start(ext.mixture, m, bo.SELECT_THRESHOLD, bo.SELECT_MIN_OBSERVATIONS)
            : 0;
        ext.decided_at = m;
        ext.break_start = start;

        // Each band's median residual from the change's first observation to
        // the deciding one, before the observations it skips leave.
        ext.magnitude.resize(B);
        std::vector<scalar_t> r(m - start);
        for (index_t b = 0; b < B; ++b)
        {
            for (index_t j = start; j < m; ++j)
                r[j - start] = ext.sums.residual(b, dates(t0 + j), spectral(b, t0 + j));
            ext.magnitude[b] = median(r);
        }
        for (index_t j = 0; j < start; ++j)
            series.drop(t0, window);
        return LookforwardOutcome::Change;
    }

    if (decision == Decision::Continue && N < bo.MAX_PEEK)
        return LookforwardOutcome::SeriesEnd;

    //--------------------------------------------------------------------------
    // No change: the outlier test on the first peek observation.
    //--------------------------------------------------------------------------
    scalar_t magnitude = 0.0;
    for (index_t d = 0; d < D; ++d)
        magnitude += squared_ratio(ext.residuals[d * N], ext.mse[d]);
    if (magnitude > options.OUTLIER_THRESHOLD)
    {
        series.drop(t0, window);
        return LookforwardOutcome::Dropped;
    }
    ext.sums.add(dates(t0), spectral, t0);
    window = Window(window.start, window.stop + 1);
    return LookforwardOutcome::Appended;
}


namespace
{

// The segment Lookforward has grown: the window's fit in every band, its
// coefficient count and size, its break day and magnitudes.
void write_segment(State& state, scalar_t change_probability, std::int64_t break_day)
{
    standard::State& core = state.core;
    const Window& window = core.cursor.window;
    const Extension& ext = state.extension;
    const auto dates = core.series.dates();
    const index_t B = core.series.spectral().extent(0);
    const index_t n = ext.sums.count();
    const index_t q = ext.terms;

    ChangeModel result;
    result.start_day = dates(window.start);
    result.end_day = dates(window.stop - 1);
    result.break_day = break_day;
    result.observation_count = window.size();
    result.change_probability = change_probability;
    result.curve_qa = static_cast<CurveQA>(q);
    result.bands.reserve(B);
    for (index_t b = 0; b < B; ++b)
    {
        const scalar_t rmse = std::sqrt(ext.sums.ssr(b) / static_cast<scalar_t>(n - q));
        const scalar_t magnitude = change_probability == 1.0 ? ext.magnitude[b] : 0.0;
        result.bands.push_back(ext.sums.band_fit(b, rmse, magnitude));
    }
    core.segments.push_back(std::move(result));
}

// After a stable start and after Lookback: the Start segment when none is
// written yet and enough observations precede the window, else Lookforward.
Phase start_guard(const Context& context, State& state)
{
    const standard::State& core = state.core;
    const Cursor& cursor = core.cursor;
    if (core.segments.empty() && core.carried == 0 &&
        cursor.window.start - cursor.prev_break >= context.options.BFP.START_CATCH_MIN)
        return Phase::StartCatch;
    return Phase::Lookforward;
}

Phase end_catch_because(State& state, standard::Exit why)
{
    state.core.exit = why;
    return Phase::EndCatch;
}

} // namespace

Phase lookforward(const Context& context, State& state)
{
    const LookforwardOutcome outcome = lookforward_step(context, state);
    LOG_DEBUG("Lookforward: " << to_string(outcome));

    standard::State& core = state.core;
    const Window& window = core.cursor.window;
    const auto dates = core.series.dates();
    switch (outcome)
    {
        case LookforwardOutcome::Change:
            // The change's first observation is at window.stop now that the
            // ones it skipped have left.
            write_segment(state, 1.0, dates(window.stop));
            state.extension.trained = false;
            return Phase::Restart;

        case LookforwardOutcome::SeriesEnd:
            // A fit failed before any test (dependent columns): nothing to
            // write, the series ends with the segments written.
            if (state.extension.terms == 0 || state.extension.fitted != state.extension.sums.count())
                return Phase::Done;
            write_segment(state, 0.0,
                          window.stop < core.series.size() ? dates(window.stop)
                                                           : dates(window.stop - 1));
            state.extension.trained = false;
            core.exit = standard::Exit::SeriesEnd;
            return Phase::Done;

        case LookforwardOutcome::Dropped:
        case LookforwardOutcome::Appended:
            break;
    }
    return Phase::Lookforward;
}

Phase Machine::step(const Context& context, State& state)
{
    standard::State& core = state.core;
    switch (state.phase)
    {
        case Phase::Prepare:
            switch (standard::prepare(context, core))
            {
                case standard::Phase::Initialize: return Phase::Initialize;
                case standard::Phase::EndCatch:   return Phase::EndCatch;
                default:                          return Phase::Done;
            }

        case Phase::Initialize:
            switch (standard::initialize_outcome(context, core))
            {
                case InitializeOutcome::Exhausted:
                    return end_catch_because(state, standard::Exit::InitializeExhausted);
                case InitializeOutcome::Stable:
                case InitializeOutcome::DroppedThenStable:
                    if (core.cursor.window.start > core.cursor.prev_break)
                        return Phase::Lookback;
                    return start_guard(context, state);
                default:
                    return Phase::Initialize;
            }

        case Phase::Lookback:
        {
            const standard::LookbackOutcome outcome = standard::lookback_step(context, core);
            if (outcome != standard::LookbackOutcome::Change &&
                core.cursor.window.start > core.cursor.prev_break)
                return Phase::Lookback;
            return start_guard(context, state);
        }

        case Phase::StartCatch:
            standard::write_start_segment(context, core);
            return Phase::Lookforward;

        case Phase::Lookforward:
            return lookforward(context, state);

        case Phase::Restart:
            standard::move_past_segment(context, core);
            if (standard::room_to_initialize(context, core))
                return Phase::Initialize;
            return end_catch_because(state, standard::Exit::NoRoomToInitialize);

        case Phase::EndCatch:
            // JuliaCCD's end fit: after a segment that ended in a change,
            // when enough observations follow it.
            if (!core.segments.empty() && core.segments.back().change_probability == 1.0 &&
                core.series.size() - core.cursor.prev_break >= context.options.BFP.END_CATCH_MIN)
                standard::write_end_segment(context, core);
            return Phase::Done;

        case Phase::Done:
            break;
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
        const Context context{hworkspace.options(), hworkspace, lworkspace, solver};
        return run(context, state_, std::move(mask));
    }
    return fit_whole_series(procedure, hworkspace, std::move(mask), lworkspace, solver);
}

} // namespace bfp
} // namespace ccd
