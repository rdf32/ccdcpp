#include "ccd/procedure/standard.hpp"

#include <algorithm>
#include <vector>
#include <cassert>
#include <cmath>

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

bool lookback(
    Series& series,
    Window& window,
    index_t prev_break,
    const std::vector<LassoResult>& models,
    const Cadence& cadence,
    const Variogram& variogram,
    const HarmonicOptions& options,
    HarmonicBasis& peek_basis
) {
    struct ReverseWindow
    {
        std::ptrdiff_t start;
        std::ptrdiff_t stop;
    };

    const BandList& detection_bands = options.DETECTION_BANDS;

    std::vector<std::int64_t> peek_dates;
    std::vector<scalar_t> peek_spectral;

    // Indexed by band; only the detection bands' are written.
    std::vector<std::vector<scalar_t>> residuals(series.spectral().extent(0));
    // Indexed by position in DETECTION_BANDS.
    std::vector<scalar_t> comp_rmses(detection_bands.size());
    std::vector<scalar_t> magnitude;

    while(window.start > prev_break) {

        const auto masked_dates = series.dates();
        const auto masked_spectral = series.spectral();

        // Python:
        // slice(window.start - 1,
        //       window.start - peek_size,
        //       -1)
        ReverseWindow peek_window;
        if ((window.start - prev_break) > cadence.peek)
        {
            peek_window.start = static_cast<std::ptrdiff_t>(window.start) - 1;
            peek_window.stop  = static_cast<std::ptrdiff_t>(window.start) - cadence.peek;
        }
        else if (window.start <= cadence.peek)
        {
            peek_window.start = static_cast<std::ptrdiff_t>(window.start) - 1;
            peek_window.stop  = -1;
        }
        else
        {
            peek_window.start = static_cast<std::ptrdiff_t>(window.start) - 1;
            peek_window.stop  = static_cast<std::ptrdiff_t>(prev_break) - 1;
        }

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
            variogram,
            comp_rmses,
            magnitude
        );

        //----------------------------------------------------------------------
        // Change detected
        //----------------------------------------------------------------------
        if (detect_change(magnitude, cadence.change_threshold))
        {   
            return true;
        }
        //----------------------------------------------------------------------
        // Outlier detected
        //----------------------------------------------------------------------
        else if (detect_outlier(magnitude[0], options.OUTLIER_THRESHOLD))
        {
            //----------------------------------------------------------
            // Drop the observation immediately before the window; the
            // window moves down one position with the observations
            // after it
            //----------------------------------------------------------
            series.drop(static_cast<index_t>(peek_window.start), window);
            continue;
        }
        window.start = static_cast<index_t>(peek_window.start);
    }
    return false;
}

ChangeModel lookforward(
    Series& series,
    Window& window,
    const Cadence& cadence,
    const Variogram& variogram,
    const HarmonicOptions& options,
    HarmonicBasis& peek_basis,
    LassoWorkspace& lworkspace,
    LassoSolver& solver
) {
    LOG_DEBUG("initial model window lookforward: " << window.start << ", " << window.stop);
    ChangeModel result;
    const BandList& detection_bands = options.DETECTION_BANDS;

    index_t change = 0;
    auto fit_window = window;
    auto fit_span = span(series.dates(), window);
    Window peek_window;
    index_t num_coef = options.COEFFICIENT_MIN;
    auto num_bands = series.spectral().extent(0);
    std::vector<std::vector<scalar_t>> full_resids(num_bands);
    std::vector<LassoResult> models(num_bands);
    bool trained = false;

    // Indexed by position in DETECTION_BANDS.
    std::vector<scalar_t> comp_rmses(detection_bands.size());
    std::vector<scalar_t> magnitude;

    // while model_window.stop + peek_size <= period.shape[0]:
    while (window.stop + cadence.peek <= series.size()) {

        const auto masked_dates = series.dates();
        const auto masked_spectral = series.spectral();

        num_coef = coefficient_count(window.size(), options);

        peek_window = 
            Window(window.stop, window.stop + cadence.peek);

        // # Used for comparison against fit_span
        auto model_span = span(masked_dates, window);
        
        LOG_DEBUG("detecting change for: " << peek_window.start << ", " << peek_window.stop);

        if (!trained || window.stop - window.start < options.RETRAIN_MIN_OBS || model_span >= options.RETRAIN_SPAN_FACTOR * fit_span) 
        {   
            trained = true;
            fit_window = window;
            fit_span = span(masked_dates, fit_window);
            LOG_DEBUG("Retrain models");

            // seasonal_rmse() reads the residuals; the magnitude is set
            // from the peek residuals when the segment is returned.
            fit_bands(
                masked_dates.slice(range(fit_window.start, fit_window.stop)),
                masked_spectral.slice(all(), range(fit_window.start, fit_window.stop)),
                num_coef,
                KeepResiduals::Yes,
                lworkspace,
                solver,
                models
            );
        }

        // The 8-coefficient basis, whatever the models' count (quirk 4).
        peek_basis.build(masked_dates.slice(range(peek_window.start, peek_window.stop)), 8);

        peek_residuals(
            peek_basis,
            masked_spectral.slice(all(), range(peek_window.start, peek_window.stop)),
            models,
            full_resids
        );

        if (window.stop - window.start <= options.RETRAIN_MIN_OBS) {
            LOG_DEBUG("normal residuals and rmse");
            for (index_t i = 0; i < detection_bands.size(); ++i)
            {   
                comp_rmses[i] = models[detection_bands[i]].score.rmse;
            }
        } else {

            LOG_DEBUG("closest day residuals and rmse");
            auto closest_indexes = 
                find_closest_doy(masked_dates, peek_window.stop - 1, fit_window, options.SEASONAL_RMSE_COUNT);
            
            LOG_DEBUG("period: " << masked_dates);
            LOG_DEBUG("peek window: (" << peek_window.start << ", " << peek_window.stop << ")");
            LOG_DEBUG("fit window: (" << fit_window.start << ", " << fit_window.stop << ")");
            LOG_DEBUG("closest inds: " << closest_indexes);
   
            for (index_t i = 0; i < detection_bands.size(); ++i)
            {   
                comp_rmses[i] = seasonal_rmse(models[detection_bands[i]], closest_indexes, options.SEASONAL_RMSE_DIVISOR);
            }
        }

        change_magnitude(
            full_resids,
            detection_bands,
            variogram,
            comp_rmses,
            magnitude
        );

        if (detect_change(magnitude, cadence.change_threshold))
        {   
            LOG_DEBUG("Change detected at: " << peek_window.start);
            change = 1;
            break;
        } 
        else if (detect_outlier(magnitude[0], options.OUTLIER_THRESHOLD))
        {   
            LOG_DEBUG("Outlier detected at: " << peek_window.start);
            LOG_DEBUG("mag: " << magnitude[0] << " thresh: " << options.OUTLIER_THRESHOLD);
            LOG_DEBUG("Updating processing mask lookforward");
            //----------------------------------------------------------
            // Drop the first peek observation, at window.stop; the
            // window keeps its coordinates
            //----------------------------------------------------------
            series.drop(peek_window.start, window);
            continue;
        }

        if (window.stop + cadence.peek > series.size()) {
            break;
        }

        window = Window(window.start, window.stop + 1);
    }

    const auto masked_dates = series.dates();

    result.start_day = masked_dates(window.start);
    result.end_day = masked_dates(window.stop - 1);
    result.break_day = masked_dates(peek_window.start);
    result.observation_count = window.stop - window.start;
    result.change_probability = static_cast<scalar_t>(change);
    result.curve_qa = static_cast<CurveQA>(num_coef);

    result.bands = std::move(models);
    for (index_t band = 0; band < num_bands; ++band) {
        result.bands[band].score.magn = median(full_resids[band]);

        // The fit-window residuals were for seasonal_rmse(); nothing reads
        // a returned segment's residuals.
        result.bands[band].score.residuals = std::vector<scalar_t>();
    }

    return result;
}

} // namespace standard

FitResult StandardProcedure::run(
    const HarmonicWorkspace& hworkspace,
    ProcessingMask mask,
    LassoWorkspace& lworkspace,
    LassoSolver& solver
) {
    FitResult results;

    const auto& options = hworkspace.options();

    // validate() refuses less: Lookback's far peek reads PEEK_SIZE - 1
    // observations and both peeks read their first magnitude.
    assert(options.PEEK_SIZE >= 2);

    LOG_DEBUG("num obs: " << hworkspace.dates().size());
    LOG_DEBUG("Processing mask initial count: " << mask.count());
    //----------------------------------------------------------
    // Not enough observations
    //----------------------------------------------------------
    if (mask.count() <= options.MEOW_SIZE)
    {     
        LOG_DEBUG("Processing Mask: " << mask.data());        
        LOG_DEBUG("mask count < meow size" << mask.count());
        return {};
    }

    series_.reset(hworkspace, std::move(mask));

    //----------------------------------------------------------
    // Statistics window: the observations dated <= STAT_ORD
    // (views, no copy)
    //----------------------------------------------------------
    const index_t stat_count = series_.count_until(options.STAT_ORD);
    LOG_DEBUG("Stat mask count: " << stat_count);

    const auto stat_dates =
        series_.dates().slice(range(index_t{0}, stat_count));

    const auto stat_spectral =
        series_.spectral().slice(all(), range(index_t{0}, stat_count));

    LOG_DEBUG("stat dates: " << stat_dates);

    //----------------------------------------------------------
    // The pixel's cadence and variogram, read by every step
    // below; the caller's options are never written
    //----------------------------------------------------------
    const standard::Cadence cadence =
        standard::measure_cadence(stat_dates, options);

    assert(cadence.peek >= options.PEEK_SIZE);
    LOG_DEBUG("Peek size: " << cadence.peek);
    LOG_DEBUG("Chng thresh: " << cadence.change_threshold);

    const Variogram variogram{
        adjusted_variogram(
            stat_dates,
            stat_spectral,
            options.VARIOGRAM_MIN_LAG_DAYS
        )
    };

    if (!variogram.usable())
    {   
        LOG_DEBUG("variogram has nans");
        return {};
    }
    LOG_DEBUG("Variogram values: " << variogram.bands);

    // Initialise's criteria: the options' and this pixel's change threshold,
    // with the 4-coefficient stability fits pyccd hard-codes (quirk 4).
    const InitializeCriteria init_criteria{
        options.MEOW_SIZE,
        options.DAY_DELTA,
        options.TMASK_BANDS,
        options.T_CONST,
        options.DETECTION_BANDS,
        4,
        cadence.change_threshold
    };

    // initialize processing context
    bool start = true;
    index_t prev_break = 0;
    index_t num_bands = series_.spectral().extent(0);
    auto window = Window(0, options.MEOW_SIZE);

    // window.stop <= observations - MEOW_SIZE, written without the
    // subtraction: index_t is unsigned, and observations < MEOW_SIZE would
    // wrap it to a huge bound.
    while(window.stop + options.MEOW_SIZE <= series_.size())
    {   
        LOG_DEBUG("Initialize for change model #: " << results.models.size());
        if (!results.models.empty()) {
            start = false;
        }

        std::vector<LassoResult> init_models(num_bands);
        auto initialized = initialize(
            series_,
            window,
            init_criteria,
            variogram,
            lworkspace,
            solver,
            init_models
        );

        if (!initialized) {
            LOG_DEBUG("Model initialization failed");
            break;
        }

        if (window.start > prev_break)
        {
            standard::lookback(
                series_,
                window,
                prev_break,
                init_models,
                cadence,
                variogram,
                options,
                peek_basis_
            );
        }

        if ((window.start - prev_break > cadence.peek)  && start)
        {   
            auto window_0 = Window(prev_break, window.start);
            ChangeModel result = catch_segment(
                series_.dates(),
                series_.spectral(),
                window_0,
                options.COEFFICIENT_MIN,
                CurveQA::Start,
                lworkspace,
                solver
            );
            // append result
            start = false;
            results.models.push_back(std::move(result));
        }

        if (window.stop + cadence.peek > series_.size()) {
            break;
        }

        ChangeModel result = standard::lookforward(
            series_,
            window,
            cadence,
            variogram,
            options,
            peek_basis_,
            lworkspace,
            solver
        );
        results.models.push_back(std::move(result));
        //--------------------------------------------------
        // iterate
        //--------------------------------------------------
        prev_break = window.stop;
        window = Window(window.stop, window.stop + options.MEOW_SIZE);
    }
    // --------------------------------
    // End catch
    // --------------------------------
    if (prev_break + cadence.peek < series_.size()) {
        auto window_1 = Window(prev_break, series_.size());
        ChangeModel result = catch_segment(
            series_.dates(),
            series_.spectral(),
            window_1,
            options.COEFFICIENT_MIN,
            CurveQA::End,
            lworkspace,
            solver
        );
        results.models.push_back(std::move(result));
    }
    results.mask = series_.take_mask();
    return results;
}

} // namespace ccd