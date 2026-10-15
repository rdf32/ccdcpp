#include "ccd/procedure/standard.hpp"

#include <algorithm>
#include <vector>
#include <iostream>

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

} // namespace standard

bool StandardProcedure::initialize(
    const HarmonicWorkspace& workspace,
    LassoWorkspace& lworkspace,
    LassoSolver& solver,
    const Variogram& variogram,
    scalar_t change_threshold,
    Window& window,
    std::vector<LassoResult>& models
)
{   
    LOG_DEBUG(
        "Initial model window (" 
        << window.start 
        << ", " 
        << window.stop 
        << ")"
    );

    const auto& options = workspace.options();

    while (window.stop + options.MEOW_SIZE < series_.size())
    {   
        // The window's observations; a Tmask drop below replaces them.
        auto dates =
            series_.dates().slice(range(window.start, window.stop));
        auto spectral =
            series_.spectral().slice(all(), range(window.start, window.stop));

        //----------------------------------------------------------------------
        // Check minimum time span
        //----------------------------------------------------------------------
        if (!enough_time(dates, options.DAY_DELTA))
        {
            window.grow();
            continue;
        }

        LOG_DEBUG(
            "Checking window (" 
            << window.start 
            << ", " 
            << window.stop 
            << ")"
        );
        //----------------------------------------------------------------------
        // Run TMask
        //----------------------------------------------------------------------
        auto outliers =
            tmask(
                dates,
                spectral,
                variogram,
                options.TMASK_BANDS,
                options.T_CONST
            );
        LOG_DEBUG("Number of Tmask outliers found: " << outliers.count());
        //----------------------------------------------------------------------
        // TMask removed everything
        //----------------------------------------------------------------------
        if (outliers.count() == window.size())
        {   
            window.grow();
            continue;
        }
       
        //---------------------------------------------------------------------
        // Update persistent mask
        //---------------------------------------------------------------------
        if (outliers.any())
        {
            //--------------------------------------------------------------
            // Drop them only if the shortened window is still usable;
            // otherwise grow, with nothing dropped
            //--------------------------------------------------------------
            const bool dropped = series_.drop_outliers_if(
                outliers,
                window,
                [&options](ArrayView<const std::int64_t, 1> shortened)
                {
                    return enough_time(shortened, options.DAY_DELTA) &&
                        enough_samples(shortened, options.MEOW_SIZE);
                }
            );

            if (!dropped)
            {
                window.grow();
                continue;
            }

            dates = series_.dates().slice(range(window.start, window.stop));
            spectral = series_.spectral().slice(all(), range(window.start, window.stop));
        }

        LOG_DEBUG("Generating models to check for stability");
        LOG_DEBUG(
            "window for stability: (" 
            << window.start 
            << ", " 
            << window.stop 
            << ")"
        );
        //----------------------------------------------------------------------
        // Fit models
        //----------------------------------------------------------------------
        const index_t num_coef = 4;

        //----------------------------------------------------------
        // Build harmonic basis
        //----------------------------------------------------------
        lworkspace.resize(dates.size());
        lworkspace.build_basis(dates, num_coef);

        //----------------------------------------------------------
        // Fit the detection bands
        //
        // Only these are read: stable() here and lookback() after, and
        // run() discards the initialization models once lookforward()
        // retrains. Each fit is independent of the others (fit() resets
        // the workspace), so skipping the rest changes nothing.
        //----------------------------------------------------------
        for (const index_t band : options.DETECTION_BANDS)
        {
            auto y = spectral.slice(
                fixed(band), 
                all()
            );

            LassoModel model = solver.fit(
                lworkspace,
                y
            );

            model.predict(
                lworkspace.X(),
                lworkspace.predictions
            );

            // stable() and lookback() read the RMSE and the residuals; the
            // magnitude of these models is never read.
            models[band].model = std::move(model);
            models[band].score.rmse = rmse(
                y,
                lworkspace.predictions,
                lworkspace.residuals,
                num_coef,
                true
            );
            models[band].score.magn = 0.0;
            models[band].score.residuals = lworkspace.residuals;
        }

        //----------------------------------------------------------------------
        // Stability check
        //----------------------------------------------------------------------
        if (stable(
                dates,
                variogram,
                models,
                change_threshold,
                options.DETECTION_BANDS))
        {
            LOG_DEBUG("Stable start found: (" << window.start << ", " << window.stop << ")");
            LOG_DEBUG("Final model window: (" << window.start << ", " << window.stop << ")");
            return true;
        }
        window.shift();
        LOG_DEBUG("Unstable model, shift window to: (" << window.start << ", " << window.stop << ")");
    }
    LOG_DEBUG("Final model window: (" << window.start << ", " << window.stop << ")");
    return false;
}

void StandardProcedure::calc_residuals(
    ArrayView<const scalar_t, 1> y,
    const std::vector<scalar_t>& preds,
    std::vector<scalar_t>& residuals
)
{
    assert(y.size() == preds.size());

    residuals.resize(y.size());

    for (index_t i = 0; i < static_cast<index_t>(y.size()); ++i)
    {
        residuals[i] =
            std::abs(y(i) - preds[i]);
    }
}

//------------------------------------------------------------------------------
// change_magnitude() over residuals[bands[i]], into `magnitude`.
//
// The same sum in the same band order as change_magnitude() given the
// per-detection-band copies, without making the copies.
//------------------------------------------------------------------------------
static void detection_magnitude(
    const std::vector<std::vector<scalar_t>>& residuals,
    const BandList& bands,
    const std::vector<scalar_t>& variogram,
    const std::vector<scalar_t>& rmse,
    std::vector<scalar_t>& magnitude
)
{
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
                variogram[i],
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
}

bool StandardProcedure::lookback(
    const HarmonicWorkspace& workspace,
    LassoWorkspace& lworkspace,
    const standard::Cadence& cadence,
    const Variogram& variogram,
    Window& window,
    const std::vector<LassoResult>& models,
    index_t prev_break
) {
    struct ReverseWindow
    {
        std::ptrdiff_t start;
        std::ptrdiff_t stop;
    };

    const auto& options = workspace.options();

    std::vector<std::int64_t> peek_dates;
    std::vector<scalar_t> peek_spectral;

    std::vector<scalar_t> comp_rmses(options.DETECTION_BANDS.size());
    std::vector<std::vector<scalar_t>> comp_resids(options.DETECTION_BANDS.size());
    std::vector<scalar_t> comp_vario(options.DETECTION_BANDS.size());

    while(window.start > prev_break) {

        const auto masked_dates = series_.dates();
        const auto masked_spectral = series_.spectral();

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
        
        lworkspace.resize(masked_dates_window.size());
        lworkspace.build_basis(masked_dates_window, 8);

        for (index_t i = 0; i < options.DETECTION_BANDS.size(); ++i)
        {
            auto band = options.DETECTION_BANDS[i];
            auto y = masked_spectral_window.slice(
                fixed(band), 
                all()
            );

            models[band].model.predict(
                lworkspace.X(),
                lworkspace.predictions
            );

            calc_residuals(y, lworkspace.predictions, comp_resids[i]);
            // use original model rmse like python code 
            // not sure if this is intended or python bug
            comp_rmses[i] = models[band].score.rmse;
            comp_vario[i] = variogram[band];
        }

        auto magnitude = 
            change_magnitude(comp_resids, comp_vario, comp_rmses);

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
            series_.drop(static_cast<index_t>(peek_window.start), window);
            continue;
        }
        window.start = static_cast<index_t>(peek_window.start);
    }
    return false;
}


ChangeModel StandardProcedure::catch_model(
    const HarmonicWorkspace& workspace,
    LassoWorkspace& lworkspace,
    LassoSolver& solver,
    const Window& window,
    CurveQA curve_qa
) {
    LOG_DEBUG(
        "Catching observations: (" 
        << window.start 
        << ", " 
        << window.stop 
        << ")"
    );
    ChangeModel result;
    const auto& options = workspace.options();

    const auto masked_dates = series_.dates();
    const auto masked_spectral = series_.spectral();

    LOG_DEBUG("masked spectral catch: " << masked_spectral);
    auto masked_dates_window = 
        masked_dates.slice(range(window.start, window.stop));

    auto masked_spectral_window = 
        masked_spectral.slice(all(), range(window.start, window.stop));

    LOG_DEBUG("masked spectral window catch: " << masked_spectral_window);

    //----------------------------------------------------------
    // Determine harmonic complexity
    //----------------------------------------------------------
    const index_t num_coef = options.COEFFICIENT_MIN;

    //----------------------------------------------------------
    // Build harmonic basis
    //----------------------------------------------------------
    lworkspace.resize(masked_dates_window.size());
    lworkspace.build_basis(masked_dates_window, num_coef);

    //----------------------------------------------------------
    // Fit every spectral band
    //----------------------------------------------------------
    LOG_DEBUG("catch model training: ");
    result.bands.resize(masked_spectral_window.extent(0));
    for (index_t band = 0; band < masked_spectral_window.extent(0); ++band)
    {
        auto y = masked_spectral_window.slice(
            fixed(band), 
            all()
        );

        LOG_DEBUG("band: " << band);
        LOG_DEBUG("y: " << y);
        LOG_DEBUG("X: " << lworkspace.X());

        LassoModel model = solver.fit(
            lworkspace,
            y
        );

        model.predict(
            lworkspace.X(),
            lworkspace.predictions
        );

        // Magnitude is 0 for a caught segment and nothing reads a returned
        // segment's residuals, so only the RMSE is computed.
        LassoScore metrics{
            rmse(
                y,
                lworkspace.predictions,
                lworkspace.residuals,
                num_coef,
                true
            ),
            0.0,
            {}
        };

        result.bands[band] = LassoResult{
            std::move(model),
            std::move(metrics)
        };
    }

    if (window.stop >= masked_dates.size()) {
        result.break_day = masked_dates(masked_dates.size() - 1);
    } else {
        result.break_day = masked_dates(window.stop);
    }
    result.start_day = masked_dates(window.start);
    result.end_day   = masked_dates(window.stop - 1);
    
    result.observation_count = window.stop - window.start;
    result.change_probability = 0.0;
    result.curve_qa = curve_qa;

    return result;
}

ChangeModel StandardProcedure::lookforward(
    const HarmonicWorkspace& workspace,
    LassoWorkspace& lworkspace,
    LassoSolver& solver,
    const standard::Cadence& cadence,
    const Variogram& variogram,
    Window& window
) {
    LOG_DEBUG("initial model window lookforward: " << window.start << ", " << window.stop);
    ChangeModel result;
    const auto& options = workspace.options();

    index_t change = 0;
    auto fit_window = window;
    auto fit_span = span(series_.dates(), window);
    Window peek_window;
    index_t num_coef = options.COEFFICIENT_MIN;
    auto num_bands = series_.spectral().extent(0);
    std::vector<std::vector<scalar_t>> full_resids(num_bands);
    std::vector<LassoResult> models(num_bands);
    bool trained = false;

    std::vector<scalar_t> comp_rmses(options.DETECTION_BANDS.size());
    std::vector<scalar_t> comp_vario(options.DETECTION_BANDS.size());
    std::vector<scalar_t> magnitude;

    // while model_window.stop + peek_size <= period.shape[0]:
    while (window.stop + cadence.peek <= series_.size()) {

        const auto masked_dates = series_.dates();
        const auto masked_spectral = series_.spectral();

        num_coef = coefficient_count(
            masked_dates.slice(range(window.start, window.stop)), options);

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

            auto masked_dates_window = 
                masked_dates.slice(range(fit_window.start, fit_window.stop));

            auto masked_spectral_window = 
                masked_spectral.slice(all(), range(fit_window.start, fit_window.stop));
            //----------------------------------------------------------
            // Build harmonic basis
            //----------------------------------------------------------
            lworkspace.resize(masked_dates_window.size());
            lworkspace.build_basis(masked_dates_window, num_coef);
           
            //----------------------------------------------------------
            // Fit every spectral band
            //----------------------------------------------------------
            for (index_t band = 0; band < masked_spectral_window.extent(0); ++band)
            {
                auto y = masked_spectral_window.slice(
                    fixed(band), 
                    all()
                );
                LassoModel model = solver.fit(
                    lworkspace,
                    y
                );
                model.predict(
                    lworkspace.X(),
                    lworkspace.predictions
                );
                // seasonal_rmse() reads the residuals; the magnitude is
                // set from the peek residuals when the segment is returned.
                auto& fitted = models[band];
                fitted.model = std::move(model);
                fitted.score.rmse = rmse(
                    y,
                    lworkspace.predictions,
                    lworkspace.residuals,
                    num_coef,
                    true
                );
                fitted.score.magn = 0.0;
                fitted.score.residuals.assign(
                    lworkspace.residuals.begin(),
                    lworkspace.residuals.end()
                );
            }
        }

        lworkspace.resize(masked_dates.slice(range(peek_window.start, peek_window.stop)).size());
        lworkspace.build_basis(masked_dates.slice(range(peek_window.start, peek_window.stop)), 8);

        for (index_t band = 0; band < num_bands; ++band)
        {
            auto y = masked_spectral.slice(all(), range(peek_window.start, peek_window.stop)).slice(
                fixed(band), 
                all()
            );

            models[band].model.predict(
                lworkspace.X(),
                lworkspace.predictions
            );

            calc_residuals(y, lworkspace.predictions, full_resids[band]);
        }

        if (window.stop - window.start <= options.RETRAIN_MIN_OBS) {
            LOG_DEBUG("normal residuals and rmse");
            for (index_t i = 0; i < options.DETECTION_BANDS.size(); ++i)
            {   
                auto band = options.DETECTION_BANDS[i];
                comp_rmses[i] = models[band].score.rmse;
                comp_vario[i] = variogram[band];
            }
        } else {

            LOG_DEBUG("closest day residuals and rmse");
            auto closest_indexes = 
                find_closest_doy(masked_dates, peek_window.stop - 1, fit_window, options.SEASONAL_RMSE_COUNT);
            
            LOG_DEBUG("period: " << masked_dates);
            LOG_DEBUG("peek window: (" << peek_window.start << ", " << peek_window.stop << ")");
            LOG_DEBUG("fit window: (" << fit_window.start << ", " << fit_window.stop << ")");
            LOG_DEBUG("closest inds: " << closest_indexes);
   
            for (index_t i = 0; i < options.DETECTION_BANDS.size(); ++i)
            {   
                auto band = options.DETECTION_BANDS[i];
                comp_rmses[i] = seasonal_rmse(models[band], closest_indexes, options.SEASONAL_RMSE_DIVISOR);
                comp_vario[i] = variogram[band];
            }
        }

        detection_magnitude(
            full_resids,
            options.DETECTION_BANDS,
            comp_vario,
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
            series_.drop(peek_window.start, window);
            continue;
        }

        if (window.stop + cadence.peek > series_.size()) {
            break;
        }

        window = Window(window.start, window.stop + 1);
    }

    const auto masked_dates = series_.dates();

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

FitResult StandardProcedure::run(
    const HarmonicWorkspace& hworkspace,
    ProcessingMask mask,
    LassoWorkspace& lworkspace,
    LassoSolver& solver
) {
    FitResult results;

    const auto& options = hworkspace.options();

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
            hworkspace,
            lworkspace,
            solver,
            variogram,
            cadence.change_threshold,
            window,
            init_models
        );

        if (!initialized) {
            LOG_DEBUG("Model initialization failed");
            break;
        }

        if (window.start > prev_break)
        {
            lookback(hworkspace, lworkspace, cadence, variogram, window, init_models, prev_break);
        }

        if ((window.start - prev_break > cadence.peek)  && start)
        {   
            auto window_0 = Window(prev_break, window.start);
            ChangeModel result = 
                catch_model(hworkspace, lworkspace, solver, window_0, CurveQA::Start);
            // append result
            start = false;
            results.models.push_back(std::move(result));
        }

        if (window.stop + cadence.peek > series_.size()) {
            break;
        }

        ChangeModel result = lookforward(
            hworkspace,
            lworkspace,
            solver,
            cadence,
            variogram,
            window
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
        ChangeModel result = 
            catch_model(hworkspace, lworkspace, solver, window_1, CurveQA::End);
        results.models.push_back(std::move(result));
    }
    results.mask = series_.take_mask();
    return results;
}

} // namespace ccd