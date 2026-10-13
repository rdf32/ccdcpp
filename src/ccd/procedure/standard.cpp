#include "ccd/procedure/standard.hpp"

#include <algorithm>
#include <vector>
#include <iostream>

#include "ccd/maths.hpp"
#include "ccd/logger.hpp"

namespace ccd
{

//------------------------------------------------------------------------------
// `src` without the observations at masked positions offset + i for which
// outliers[i] is set, into `out`.
//
// This is apply_mask() of the mask remove_outliers() would produce, taken
// from the already-masked data instead of the full time series.
//------------------------------------------------------------------------------
static void drop_outliers(
    const MaskedData& src,
    const ProcessingMask& outliers,
    index_t offset,
    MaskedData& out
)
{
    const index_t n = src.observations;

    auto dropped = [&](index_t i)
    {
        return i >= offset &&
            i - offset < outliers.size() &&
            outliers[i - offset];
    };

    out.bands = src.bands;
    out.dates.clear();
    out.spectral.clear();

    for (index_t i = 0; i < n; ++i)
    {
        if (!dropped(i))
            out.dates.push_back(src.dates[i]);
    }

    out.observations = out.dates.size();

    for (index_t b = 0; b < src.bands; ++b)
    {
        for (index_t i = 0; i < n; ++i)
        {
            if (!dropped(i))
                out.spectral.push_back(src.spectral[b * n + i]);
        }
    }
}

bool StandardProcedure::initialize(
    const HarmonicWorkspace& workspace,
    LassoWorkspace& lworkspace,
    LassoSolver& solver,
    std::vector<scalar_t>& variogram,
    ProcessingMask& mask,
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

    auto& options = workspace.options();

    auto masked_dates = 
        masked_.dates_view();

    auto masked_spectral = 
        masked_.spectral_view();
    

    while (window.stop + options.MEOW_SIZE < masked_dates.size())
    {   

        //----------------------------------------------------------------------
        // Check minimum time span
        //----------------------------------------------------------------------
        if (!enough_time(
            masked_dates.slice(range(window.start, window.stop)), options.DAY_DELTA)
        )
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
                masked_dates.slice(range(window.start, window.stop)),
                masked_spectral.slice(all(), range(window.start, window.stop)),
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
            const index_t removed = outliers.count();

            //--------------------------------------------------------------
            // Build the candidate data first (don't commit yet)
            //--------------------------------------------------------------
            drop_outliers(
                masked_,
                outliers,
                window.start,
                candidate_
            );

            auto candidate_dates =
                candidate_.dates_view();

            //--------------------------------------------------------------
            // Window is now in the candidate masked coordinate space
            //--------------------------------------------------------------
            Window candidate_window = window;
            candidate_window.stop -= removed;

            //--------------------------------------------------------------
            // Verify the shortened window is still usable
            //--------------------------------------------------------------
            auto candidate_dates_window =
                candidate_dates.slice(
                    range(
                        candidate_window.start,
                        candidate_window.stop
                    )
                );

            if (!enough_time(candidate_dates_window, options.DAY_DELTA) ||
                !enough_samples(candidate_dates_window, options.MEOW_SIZE))
            {
                window.grow();
                continue;
            }

            //--------------------------------------------------------------
            // Candidate accepted -> commit everything
            //--------------------------------------------------------------
            remove_outliers(mask, outliers, window);
            std::swap(masked_, candidate_);

            masked_dates = masked_.dates_view();
            masked_spectral = masked_.spectral_view();

            window = candidate_window;
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
        lworkspace.resize(masked_dates.slice(range(window.start, window.stop)).size());
        lworkspace.build_basis(masked_dates.slice(range(window.start, window.stop)), num_coef);

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
            auto y = masked_spectral.slice(all(), range(window.start, window.stop)).slice(
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
                masked_dates.slice(range(window.start, window.stop)),
                variogram,
                models,
                options.CHANGE_THRESHOLD,
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

std::vector<scalar_t> StandardProcedure::calc_residuals(
    ArrayView<const scalar_t, 1> y,
    std::vector<scalar_t>& preds
)
{
    assert(y.size() == preds.size());

    std::vector<scalar_t> residuals(y.size());

    for (index_t i = 0; i < static_cast<index_t>(y.size()); ++i)
    {
        residuals[i] =
            std::abs(y(i) - preds[i]);
    }

    return residuals;
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
    const std::vector<index_t>& bands,
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
    std::vector<scalar_t>& variogram,
    ProcessingMask& mask,
    Window& window,
    std::vector<LassoResult>& models,
    index_t prev_break
) {
    struct ReverseWindow
    {
        std::ptrdiff_t start;
        std::ptrdiff_t stop;
    };

    auto& options = workspace.options();

    auto masked_dates = 
        masked_.dates_view();

    auto masked_spectral = 
        masked_.spectral_view();

    std::vector<std::int64_t> peek_dates;
    std::vector<scalar_t> peek_spectral;

    std::vector<scalar_t> comp_rmses(options.DETECTION_BANDS.size());
    std::vector<std::vector<scalar_t>> comp_resids(options.DETECTION_BANDS.size());
    std::vector<scalar_t> comp_vario(options.DETECTION_BANDS.size());

    while(window.start > prev_break) {

        // Python:
        // slice(window.start - 1,
        //       window.start - peek_size,
        //       -1)
        ReverseWindow peek_window;
        if ((window.start - prev_break) > options.PEEK_SIZE)
        {
            peek_window.start = static_cast<std::ptrdiff_t>(window.start) - 1;
            peek_window.stop  = static_cast<std::ptrdiff_t>(window.start) - options.PEEK_SIZE;
        }
        else if (window.start <= options.PEEK_SIZE)
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
        peek_dates.reserve(options.PEEK_SIZE);
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
        if (detect_change(magnitude, options.CHANGE_THRESHOLD))
        {   
            return true;
        }
        //----------------------------------------------------------------------
        // Outlier detected
        //----------------------------------------------------------------------
        else if (detect_outlier(magnitude[0], options.OUTLIER_THRESHOLD))
        {
            //----------------------------------------------------------
            // Remove the observation immediately before the window
            //----------------------------------------------------------
            remove_observation(mask, peek_window.start);
            masked_.erase(peek_window.start);
    
            masked_dates = 
                masked_.dates_view();

            masked_spectral = 
                masked_.spectral_view();
            //----------------------------------------------------------
            // Account for coordinate shift
            //----------------------------------------------------------
            --window.start;
            --window.stop;
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
    ProcessingMask& mask,
    Window& window,
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
    auto& options = workspace.options();

    auto masked_dates = 
        masked_.dates_view();

    auto masked_spectral = 
        masked_.spectral_view();

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
    std::vector<scalar_t>& variogram,
    ProcessingMask& mask,
    Window& window
) {
    LOG_DEBUG("initial model window lookforward: " << window.start << ", " << window.stop);
    ChangeModel result;
    auto& options = workspace.options();

    auto masked_dates = 
        masked_.dates_view();

    auto masked_spectral = 
        masked_.spectral_view();

    index_t change = 0;
    auto fit_window = window;
    auto fit_span = span(masked_dates, window);
    Window peek_window;
    index_t num_coef = options.COEFFICIENT_MIN;
    auto num_bands = masked_spectral.extent(0);
    std::vector<std::vector<scalar_t>> full_resids(num_bands);
    std::vector<LassoResult> models(num_bands);
    bool trained = false;

    std::vector<scalar_t> comp_rmses(options.DETECTION_BANDS.size());
    std::vector<scalar_t> comp_vario(options.DETECTION_BANDS.size());
    std::vector<scalar_t> magnitude;

    // masked_.observations is mask.count(): masked_ follows every change
    // to the mask.
    // while model_window.stop + peek_size <= period.shape[0]:
    while (window.stop + options.PEEK_SIZE <= masked_.observations) {
        
        num_coef = coefficient_count(
            masked_dates.slice(range(window.start, window.stop)), options);

        peek_window = 
            Window(window.stop, window.stop + options.PEEK_SIZE);

        // # Used for comparison against fit_span
        auto model_span = span(masked_dates, window);
        
        LOG_DEBUG("detecting change for: " << peek_window.start << ", " << peek_window.stop);

        if (!trained || window.stop - window.start < 24 || model_span >= 1.33 * fit_span) 
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

        if (window.stop - window.start <= 24) {
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
                find_closest_doy(masked_dates, peek_window.stop - 1, fit_window, 24);
            
            LOG_DEBUG("period: " << masked_dates);
            LOG_DEBUG("peek window: (" << peek_window.start << ", " << peek_window.stop << ")");
            LOG_DEBUG("fit window: (" << fit_window.start << ", " << fit_window.stop << ")");
            LOG_DEBUG("closest inds: " << closest_indexes);
   
            for (index_t i = 0; i < options.DETECTION_BANDS.size(); ++i)
            {   
                auto band = options.DETECTION_BANDS[i];
                comp_rmses[i] = seasonal_rmse(models[band], closest_indexes);
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

        if (detect_change(magnitude, options.CHANGE_THRESHOLD))
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
            // Remove the observation immediately before the window
            //----------------------------------------------------------
            remove_observation(mask, peek_window.start);
            masked_.erase(peek_window.start);
    
            masked_dates = 
                masked_.dates_view();

            masked_spectral = 
                masked_.spectral_view();

            continue;
        }

        if (window.stop + options.PEEK_SIZE > masked_.observations) {
            break;
        }

        window = Window(window.start, window.stop + 1);
    }

    result.start_day = masked_dates(window.start);
    result.end_day = masked_dates(window.stop - 1);
    result.break_day = masked_dates(peek_window.start);
    result.observation_count = window.stop - window.start;
    result.change_probability = static_cast<scalar_t>(change);
    result.curve_qa = static_cast<CurveQA>(num_coef);

    for (index_t band = 0; band < 7; ++band) {
        result.bands[band] = std::move(models[band]);
        result.bands[band].score.magn = median(full_resids[band]);

        // The fit-window residuals were for seasonal_rmse(); nothing reads
        // a returned segment's residuals.
        result.bands[band].score.residuals = std::vector<scalar_t>();
    }

    return result;
}

FitResult StandardProcedure::run(
    HarmonicWorkspace& hworkspace,
    LassoWorkspace& lworkspace,
    LassoSolver& solver
) {
    FitResult results;

    auto& options = hworkspace.options();
    //----------------------------------------------------------
    // Build processing mask
    //----------------------------------------------------------
    
    LOG_DEBUG("num obs: " << hworkspace.dates().size());
    ProcessingMask mask =
        select_observations(hworkspace);
    // std::cout << "\n";

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

    //----------------------------------------------------------
    // Mask dates, spectral given ProcessingMask
    //----------------------------------------------------------
    apply_mask(hworkspace, mask, masked_);

    auto masked_dates = 
        masked_.dates_view();

    auto masked_spectral = 
        masked_.spectral_view();

    auto end =
        std::upper_bound(
            masked_dates.data(),
            masked_dates.data() + masked_dates.size(),
            options.STAT_ORD
        );

    auto max_idx =
        static_cast<index_t>(
            end - masked_dates.data()
        );
    LOG_DEBUG("Stat mask count: " << max_idx);
    auto stat_dates = 
        masked_dates.slice(range(index_t{0}, max_idx)); // just view subsets

    auto stat_spectral = 
        masked_spectral.slice(all(), range(index_t{0}, max_idx)); // just view subsets
    
    LOG_DEBUG("stat dates: " << stat_dates);

    options.PEEK_SIZE =
        adjust_peek(
            stat_dates,
            options.PEEK_SIZE
        );
    
    options.CHANGE_THRESHOLD =
        adjust_change_threshold(
            options.PEEK_SIZE,
            options.PEEK_SIZE,
            options.CHANGE_THRESHOLD
        );
    
    LOG_DEBUG("Peek size: " << options.PEEK_SIZE);
    LOG_DEBUG("Chng thresh: " << options.CHANGE_THRESHOLD);


    std::vector<scalar_t> variogram =
        adjusted_variogram(
            stat_dates,
            stat_spectral
        );

    if(!check_variogram(variogram))
    {   
        LOG_DEBUG("variogram has nans");
        return {};
    }
    LOG_DEBUG("Variogram values: " << variogram);

    // initialize processing context
    //
    // masked_.observations is mask.count() throughout: masked_ follows
    // every change the subfunctions make to the mask.
    bool start = true;
    index_t prev_break = 0;
    index_t num_bands = masked_spectral.extent(0);
    auto window = Window(0, options.MEOW_SIZE);

    // window.stop <= observations - MEOW_SIZE, written without the
    // subtraction: index_t is unsigned, and observations < MEOW_SIZE would
    // wrap it to a huge bound.
    while(window.stop + options.MEOW_SIZE <= masked_.observations)
    {   
        LOG_DEBUG("Initialize for change model #: " << results.models.size());
        if (!results.models.empty()) {
            start = false;
        }

        std::vector<LassoResult> init_models(num_bands);
        auto initialized = 
            initialize(hworkspace, lworkspace, solver, variogram, mask, window, init_models);

        if (!initialized) {
            LOG_DEBUG("Model initialization failed");
            break;
        }

        if (window.start > prev_break)
        {
            lookback(hworkspace, lworkspace, variogram, mask, window, init_models, prev_break);
        }

        if ((window.start - prev_break > options.PEEK_SIZE)  && start)
        {   
            auto window_0 = Window(prev_break, window.start);
            ChangeModel result = 
                catch_model(hworkspace, lworkspace, solver, mask, window_0, CurveQA::Start);
            // append result
            start = false;
            results.models.push_back(result);
        }

        if (window.stop + options.PEEK_SIZE > masked_.observations) {
            break;
        }

        ChangeModel result = lookforward(
            hworkspace,
            lworkspace,
            solver,
            variogram,
            mask,
            window
        );
        results.models.push_back(result);
        //--------------------------------------------------
        // iterate
        //--------------------------------------------------
        prev_break = window.stop;
        window = Window(window.stop, window.stop + options.MEOW_SIZE);
    }
    // --------------------------------
    // End catch
    // --------------------------------
    if (prev_break + options.PEEK_SIZE < masked_.observations) {
        auto window_1 = Window(prev_break, masked_.observations);
        ChangeModel result = 
            catch_model(hworkspace, lworkspace, solver, mask, window_1, CurveQA::End);
        results.models.push_back(result);
    }
    results.mask = std::move(mask);
    return results;
}

} // namespace ccd