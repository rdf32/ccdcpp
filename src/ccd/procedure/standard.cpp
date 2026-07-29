#include "ccd/procedure/standard.hpp"

#include <iostream>
#include <algorithm>
#include <vector>

#include "ccd/maths.hpp"
#include "ccd/array_print.hpp"

namespace ccd
{

inline void print_lasso_results(const std::vector<LassoResult>& results)
{
    std::cout << "\nSpectral Models\n";
    std::cout << "----------------------------------------\n";

    for (std::size_t band = 0; band < results.size(); ++band)
    {
        const auto m = results[band].model;
        const auto s = results[band].score;

   
        const auto& coef = m.coefficients();
        const auto& resi = s.residuals;

        std::cout << "Band " << band << '\n';
        std::cout << "  Iterations : " << m.iterations() << '\n';
        std::cout << "  Intercept  : " << m.intercept() << '\n';
        std::cout << "  RMSE       : " << s.rmse << '\n';
        std::cout << "  Magnitude  : " << s.magn << '\n';

        std::cout << "  Coefficients (" << coef.size() << "): ";

        for (std::size_t i = 0; i < coef.size(); ++i)
        {
            std::cout << coef[i];

            if (i + 1 != coef.size())
                std::cout << ", ";
        }
        std::cout << "\n";

        std::cout << "  Residuals (" << resi.size() << "): ";

        for (std::size_t i = 0; i < resi.size(); ++i)
        {
            std::cout << resi[i];

            if (i + 1 != resi.size())
                std::cout << ", ";
        }
        std::cout << "\n";

        std::cout << "\n\n";
    }
    std::cout << "========================================\n";
}

bool StandardProcedure::initialize(
    const HarmonicWorkspace& workspace,
    LassoSolver& solver,
    std::vector<scalar_t>& variogram,
    ProcessingMask& mask,
    Window& window,
    std::vector<LassoResult>& models

)
{   
    auto& options = workspace.options();
    std::cout 
        << "Initial model window: " 
        << window.start 
        << ", " 
        << window.stop 
        << std::endl;
        
    MaskedData masked_data = 
        apply_mask(workspace, mask);
    
    auto masked_dates = 
        masked_data.dates_view();

    auto masked_spectral = 
        masked_data.spectral_view();
    

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
        std:: cout << "Checking window: " << window.start << ", " << window.stop << std::endl;
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
        std:: cout << "Number of Tmask outliers found: " << outliers.count() << std::endl;
        //----------------------------------------------------------------------
        // TMask removed everything
        //----------------------------------------------------------------------
        if (outliers.count() == window.size())
        {   
            std:: cout << "Tmask identified all values as outliers" << std::endl;
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
            // Build a candidate mask first (don't commit yet)
            //--------------------------------------------------------------
            ProcessingMask candidate_mask =
                update_processing_mask(
                    mask,
                    outliers,
                    window
                );

            auto candidate_data =
                apply_mask(
                    workspace,
                    candidate_mask
                );

            auto candidate_dates =
                candidate_data.dates_view();

            auto candidate_spectral =
                candidate_data.spectral_view();

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
                std::cout
                    << "Insufficient time or observations after TMask, "
                    << "extending model window"
                    << std::endl;

                window.grow();
                continue;
            }

            //--------------------------------------------------------------
            // Candidate accepted -> commit everything
            //--------------------------------------------------------------
            mask = std::move(candidate_mask);
            masked_data = std::move(candidate_data);

            masked_dates = masked_data.dates_view();
            masked_spectral = masked_data.spectral_view();

            window = candidate_window;
        }


        //----------------------------------------------------------------------
        // Fit models
        //----------------------------------------------------------------------
        std::cout << "Generating models to check for stability" << std::endl;
        std::cout << " window for stability: " << window.start << ", " << window.stop << std::endl;
        std::cout << "dates into fit models before stable: " << std::endl;
        
        print_array(masked_dates.slice(range(window.start, window.stop)));
        
        std::cout << "spectra into fit models before stable: " << std::endl;
        print_array(masked_spectral.slice(all(), range(window.start, window.stop)));

        //----------------------------------------------------------
        // Determine harmonic complexity
        //----------------------------------------------------------
        const index_t num_coef = 4;

        //----------------------------------------------------------
        // Build harmonic basis
        //----------------------------------------------------------
        auto X_storage = 
            lasso_basis(masked_dates.slice(range(window.start, window.stop)), num_coef);

        auto X = ArrayView<const scalar_t, 2>::contiguous(
            X_storage.data(),
            {masked_dates.slice(range(window.start, window.stop)).size(), 7} // always shape (num_obs, 7) for lasso basis
        );

        //----------------------------------------------------------
        // Fit every spectral band
        //----------------------------------------------------------
        for (index_t band = 0; band < masked_spectral.slice(all(), range(window.start, window.stop)).extent(0); ++band)
        {
            auto y = masked_spectral.slice(all(), range(window.start, window.stop)).slice(
                fixed(band), 
                all()
            );

            LassoModel model = solver.fit(
                X,
                y
            );

            auto preds = model.predict(
                X
            );

            auto metrics = score(
                y,
                preds,
                num_coef,
                true
            );
            LassoResult band_result = {
                model,
                metrics
            };
            models[band] = band_result;
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
            std:: cout << "Stable start found: " << window.start << ", " << window.stop << std::endl;
            print_lasso_results(models);
            return true;
        }

        window.shift();
        std:: cout << "Unstable model, shift window to: " << window.start << ", " << window.stop << std::endl;
    }

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

bool StandardProcedure::lookback(
    const HarmonicWorkspace& workspace,
    std::vector<scalar_t>& variogram,
    ProcessingMask& mask,
    Window& window,
    std::vector<LassoResult>& models,
    index_t prev_break
) {
    auto& options = workspace.options();
    std::cout << "previous break: " << prev_break << std::endl;
    std::cout 
        << "Lookback model window: " 
        << window.start 
        << ", " 
        << window.stop 
        << std::endl;
        
    MaskedData masked_data = 
        apply_mask(workspace, mask);
    
    auto masked_dates = 
        masked_data.dates_view();

    auto masked_spectral = 
        masked_data.spectral_view();

    while(window.start > prev_break) {

        Window peek_window;
        if (window.start - prev_break > options.PEEK_SIZE)
        {
            peek_window.start = window.start - options.PEEK_SIZE;
            peek_window.stop  = window.start;
        }
        else if (window.start - options.PEEK_SIZE <= 0)
        {
            peek_window.start = 0;
            peek_window.stop  = window.start;
        }
        else
        {
            peek_window.start = prev_break;
            peek_window.stop  = window.start;
        }

        std::cout << "Considering index: " << peek_window.start << " using peek window "
            << "(" << peek_window.start << ", " << peek_window.stop << ")" << std::endl;

        std::vector<std::int64_t> peek_dates;
        peek_dates.reserve(peek_window.stop - peek_window.start);

        for (index_t i = peek_window.stop; i-- > peek_window.start;)
        {
            peek_dates.push_back(masked_dates[i]);
        }
        auto masked_dates_window =
            ArrayView<const std::int64_t, 1>::contiguous(
                peek_dates.data(),
                {peek_dates.size()}
            );
        
        std::vector<scalar_t> peek_spectral;
        const index_t bands   = masked_spectral.extent(0);
        const index_t samples = peek_window.stop - peek_window.start;

        peek_spectral.resize(bands * samples);
        for (index_t band = 0; band < bands; ++band)
        {
            for (index_t j = 0; j < samples; ++j)
            {
                index_t source = peek_window.stop - 1 - j;
                peek_spectral[band * samples + j] =
                    masked_spectral(band, source);
            }
        }
        auto masked_spectral_window =
            ArrayView<const scalar_t, 2>::contiguous(
                peek_spectral.data(),
                {bands, samples}
            );
        
        std::vector<scalar_t> comp_rmses;
        std::vector<std::vector<scalar_t>> comp_resids;
        std::vector<scalar_t> comp_vario;

        auto X_storage = 
            lasso_basis(masked_dates_window, 8);

        auto X = ArrayView<const scalar_t, 2>::contiguous(
            X_storage.data(),
            {masked_dates_window.size(), 7} // always shape (num_obs, 7) for lasso basis
        );
        for (auto band: options.DETECTION_BANDS)
        {
            auto y = masked_spectral_window.slice(
                fixed(band), 
                all()
            );

            auto preds = 
                models[band].model.predict(X);

            auto abs_resid = 
                calc_residuals(y, preds);
            // use original model rmse like python code 
            // not sure if this is intended or python bug
            comp_resids.push_back(abs_resid);
            comp_rmses.push_back(models[band].score.rmse);
            comp_vario.push_back(variogram[band]);
        }

        // log.debug('RMSE values for comparison: %s', comp_rmse)
        std::cout << "RMSE values for comparison: " << std::endl;
        std::cout << "[";
        for (const auto num: comp_rmses) {
            std::cout << num << ", ";
        }
        std::cout << "]\n";
        auto magnitude = 
            change_magnitude(comp_resids, comp_vario, comp_rmses);
        // log.debug('Magnitudes of change: %s', change_mag)
        std::cout << "Magnitudes of change: " << std::endl;
        std::cout << "[";
        for (const auto num: magnitude) {
            std::cout << num << ", ";
        }
        std::cout << "]\n";
        //----------------------------------------------------------------------
        // Change detected
        //----------------------------------------------------------------------
        if (detect_change(magnitude, options.CHANGE_THRESHOLD))
        {   
            // log.debug('Change detected for index: %s', peek_window.start)
            std::cout << "Change detected for index: " << peek_window.start << std::endl;
            return true;
        }
        //----------------------------------------------------------------------
        // Outlier detected
        //----------------------------------------------------------------------
        if(detect_outlier(magnitude[0], options.OUTLIER_THRESHOLD))
        {
            std::cout << "Outlier detected for index: " << peek_window.start << std::endl;
            //----------------------------------------------------------
            // Remove the observation immediately before the window
            //----------------------------------------------------------
            mask = 
                update_processing_mask(mask, peek_window.start);

            masked_data = 
                apply_mask(workspace, mask);
    
            masked_dates = 
                masked_data.dates_view();

            masked_spectral = 
                masked_data.spectral_view();
            //----------------------------------------------------------
            // Account for coordinate shift
            //----------------------------------------------------------
            --window.start;
            --window.stop;

            continue;
        }
        window.start = peek_window.start;
    }
    return false;
}

ChangeModel StandardProcedure::catch_model(
    const HarmonicWorkspace& workspace,
    LassoSolver& solver,
    ProcessingMask& mask,
    Window& window,
    CurveQA curve_qa
) {
    ChangeModel result;
    auto& options = workspace.options();
    std::cout 
        << "Catch model window: " 
        << window.start 
        << ", " 
        << window.stop 
        << std::endl;
        
    MaskedData masked_data = 
        apply_mask(workspace, mask);
    
    auto masked_dates = 
        masked_data.dates_view();

    auto masked_spectral = 
        masked_data.spectral_view();

    auto masked_dates_window = 
        masked_dates.slice(range(window.start, window.stop));

    auto masked_spectral_window = 
        masked_spectral.slice(all(), range(window.start, window.stop));

    //----------------------------------------------------------
    // Determine harmonic complexity
    //----------------------------------------------------------
    const index_t num_coef = options.COEFFICIENT_MIN;

    //----------------------------------------------------------
    // Build harmonic basis
    //----------------------------------------------------------
    auto X_storage = 
        lasso_basis(masked_dates_window, num_coef);

    auto X = ArrayView<const scalar_t, 2>::contiguous(
        X_storage.data(),
        {masked_dates_window.size(), 7} // always shape (num_obs, 7) for lasso basis
    );

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
            X,
            y
        );

        auto preds = model.predict(
            X
        );

        auto metrics = score(
            y,
            preds,
            num_coef,
            true
        );

        metrics.magn = 0.0; // set magnitude to 0

        LassoResult band_result = {
            model,
            metrics
        };
        result.bands[band] = band_result;
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
    LassoSolver& solver,
    std::vector<scalar_t>& variogram,
    ProcessingMask& mask,
    Window& window
) {
    ChangeModel result;
    auto& options = workspace.options();
    std::cout 
        << "Catch model window: " 
        << window.start 
        << ", " 
        << window.stop 
        << std::endl;
        
    MaskedData masked_data = 
        apply_mask(workspace, mask);
    
    auto masked_dates = 
        masked_data.dates_view();

    auto masked_spectral = 
        masked_data.spectral_view();

    std::cout << "lookforward initial model window: " 
        << window.start << ", " << window.stop << std::endl;

    index_t change = 0;
    auto fit_window = window;
    auto fit_span = span(masked_dates, window);
    Window peek_window;
    index_t num_coef = options.COEFFICIENT_MIN;
    auto num_bands = masked_spectral.extent(0);
    std::vector<std::vector<scalar_t>> full_resids(num_bands);
    std::vector<LassoResult> models;
    // while model_window.stop + peek_size <= period.shape[0]:
    while (window.stop + options.PEEK_SIZE <= mask.count()) {
        
        num_coef = coefficient_count(
            masked_dates.slice(range(window.start, window.stop)), options);

        peek_window = 
            Window(window.stop, window.stop + options.PEEK_SIZE);

        // # Used for comparison against fit_span
        auto model_span = span(masked_dates, window);

        std::cout << "detecting change for: : " 
            << peek_window.start << ", " << peek_window.stop << std::endl;
        
        if (models.empty() || window.stop - window.start < 24 || model_span >= 1.33 * fit_span) 
        {   
            models.clear();
            fit_window = window;
            fit_span = span(masked_dates, fit_window);
            std::cout << "Retrain models" <<std::endl;

            auto masked_dates_window = 
                masked_dates.slice(range(fit_window.start, fit_window.stop));

            auto masked_spectral_window = 
                masked_spectral.slice(all(), range(fit_window.start, fit_window.stop));
            //----------------------------------------------------------
            // Build harmonic basis
            //----------------------------------------------------------
            auto X_storage = 
                lasso_basis(masked_dates_window, num_coef);

            auto X = ArrayView<const scalar_t, 2>::contiguous(
                X_storage.data(),
                {masked_dates_window.size(), 7} // always shape (num_obs, 7) for lasso basis
            );

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
                    X,
                    y
                );
                auto preds = model.predict(
                    X
                );

                auto metrics = score(
                    y,
                    preds,
                    num_coef,
                    true
                );

                LassoResult band_result = {
                    model,
                    metrics
                };
                models.push_back(band_result);
            }
        }

        auto X_storage = 
            lasso_basis(masked_dates.slice(range(peek_window.start, peek_window.stop)), 8);

        auto X = ArrayView<const scalar_t, 2>::contiguous(
            X_storage.data(),
            {masked_dates.slice(range(peek_window.start, peek_window.stop)).size(), 7} // always shape (num_obs, 7) for lasso basis
        );
        for (index_t band = 0; band < num_bands; ++band)
        {
            auto y = masked_spectral.slice(all(), range(peek_window.start, peek_window.stop)).slice(
                fixed(band), 
                all()
            );

            auto preds = 
                models[band].model.predict(X);

            auto abs_resid = 
                calc_residuals(y, preds);
    
            full_resids[band] = abs_resid;
        }

        std::vector<scalar_t> comp_rmses;
        std::vector<scalar_t> comp_vario;
        std::vector<std::vector<scalar_t>> comp_resids;
        if (window.stop - window.start <= 24) {
            for (auto band: options.DETECTION_BANDS)
            {   
                comp_resids.push_back(full_resids[band]);
                comp_rmses.push_back(models[band].score.rmse);
                comp_vario.push_back(variogram[band]);
            }
        } else {
            auto closest_indexes = 
                find_closest_doy(masked_dates, peek_window.stop - 1, fit_window, 24);
            
            for (auto band : options.DETECTION_BANDS)
            {   
                comp_resids.push_back(full_resids[band]);
                comp_rmses.push_back(seasonal_rmse(models[band], closest_indexes));
                comp_vario.push_back(variogram[band]);
            }
        }

        std::cout << "RMSE values for comparison: " << std::endl;
        std::cout << "[";
        for (const auto num: comp_rmses) {
            std::cout << num << ", ";
        }
        std::cout << "]\n";
        auto magnitude = 
            change_magnitude(comp_resids, comp_vario, comp_rmses);
        std::cout << "Magnitudes of change: " << std::endl;
        std::cout << "[";
        for (const auto num: magnitude) {
            std::cout << num << ", ";
        }
        std::cout << "]\n";

        if (detect_change(magnitude, options.CHANGE_THRESHOLD))
        {   
            // log.debug('Change detected for index: %s', peek_window.start)
            std::cout << "Change detected for index: " << peek_window.start << std::endl;
            change = 1;
            break;
        } 
        else if (detect_outlier(magnitude[0], options.OUTLIER_THRESHOLD))
        {
            std::cout << "Outlier detected for index: " << peek_window.start << std::endl;
            //----------------------------------------------------------
            // Remove the observation immediately before the window
            //----------------------------------------------------------
            mask = 
                update_processing_mask(mask, peek_window.start);

            masked_data = 
                apply_mask(workspace, mask);
    
            masked_dates = 
                masked_data.dates_view();

            masked_spectral = 
                masked_data.spectral_view();

            continue;
        }

        if (window.stop + options.PEEK_SIZE > mask.count()) {
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
    }

    return result;
}

FitResult StandardProcedure::run(
    HarmonicWorkspace& workspace,
    LassoSolver& solver
) {
    FitResult results;

    auto& options = workspace.options();
    //----------------------------------------------------------
    // Build processing mask
    //----------------------------------------------------------
    ProcessingMask mask =
        select_observations(workspace);

    //----------------------------------------------------------
    // Not enough observations
    //----------------------------------------------------------
    if (mask.count() < options.MEOW_SIZE)
    {
        return {};
    }

    //----------------------------------------------------------
    // Mask dates, spectral given ProcessingMask
    //----------------------------------------------------------
    MaskedData masked_data = 
        apply_mask(workspace, mask);

    auto masked_dates = 
        masked_data.dates_view();

    auto masked_spectral = 
        masked_data.spectral_view();

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

    auto stat_dates = 
        masked_dates.slice(range(index_t{0}, max_idx)); // just view subsets

    auto stat_spectral = 
        masked_spectral.slice(all(), range(index_t{0}, max_idx)); // just view subsets
    
    std::cout << "mask count: " << mask.count() << std::endl;
    options.PEEK_SIZE =
        adjust_peek(
            stat_dates,
            options.PEEK_SIZE
        );
    std::cout << "peek size: " << options.PEEK_SIZE << std::endl;
    
    options.CHANGE_THRESHOLD =
        adjust_change_threshold(
            options.PEEK_SIZE,
            options.PEEK_SIZE,
            options.CHANGE_THRESHOLD
        );
    std::cout << "change threshold: " << options.CHANGE_THRESHOLD << std::endl;

    std::vector<scalar_t> variogram =
        adjusted_variogram(
            stat_dates,
            stat_spectral
        );

    if(!check_variogram(variogram))
    {   
        std::cout << "Variogram failed check" << std::endl;
        return {};
    }

    // initialize processing context
    bool start = true;
    index_t prev_break = 0;
    index_t num_bands = masked_spectral.extent(0);
    auto window = Window(0, options.MEOW_SIZE);

    while(window.stop <= mask.count() - options.MEOW_SIZE)
    {   
        // log.debug('Initialize for change model #: %s', len(results) + 1)
        // if len(results) > 0:
        //     start = False for when allowing previous results

        std::cout << "Initialize for change model # " << results.models.size() + 1 << std::endl;
        if (!results.models.empty()) {
            start = false;
        }

        std::vector<LassoResult> init_models(num_bands);
        auto initialized = 
            initialize(workspace, solver, variogram, mask, window, init_models);

        if (!initialized) {
            break;
        }

        std::cout << "after init window: " << std::endl;
        std::cout << window.start << ", " << window.stop << std::endl;

        std::cout << "after init mask: " << std::endl;
        for (ccd::index_t i = 0; i < mask.size(); ++i)
        {
            std::cout << static_cast<int>(mask[i]) << ", ";
        }
        std::cout << '\n';

        if (window.start > prev_break)
        {
            std::cout << "executing lookback: " << std::endl;
            lookback(workspace, variogram, mask, window, init_models, prev_break);
        }

        std::cout << "after lookback window: " << std::endl;
        std::cout << window.start << ", " << window.stop << std::endl;

        std::cout << "after lookback mask: " << std::endl;
        for (ccd::index_t i = 0; i < mask.size(); ++i)
        {
            std::cout << static_cast<int>(mask[i]) << ", ";
        }
        std::cout << '\n';

        if ((window.start - prev_break > options.PEEK_SIZE)  && start)
        {   
            auto window_0 = Window(prev_break, window.start);
            ChangeModel result = 
                catch_model(workspace, solver, mask, window_0, CurveQA::Start);
            // append result
            start = false;
            results.models.push_back(result);
        }

        if (window.stop + options.PEEK_SIZE > mask.count()) {
            break;
        }

        std::cout << "Extend change model" << std::endl;
        ChangeModel result = lookforward(
            workspace,
            solver,
            variogram,
            mask,
            window
        );
        results.models.push_back(result);

        // log.debug('Accumulate results, {} so far'.format(len(results)))
        std::cout << "prev_break: " << prev_break << std::endl;
        std::cout << "window: " << window.start << ", " << window.stop << std::endl;
        std::cout << "Accumulate results " << results.models.size() << " so far" << std::endl;
        //--------------------------------------------------
        // iterate
        //--------------------------------------------------
        prev_break = window.stop;
        window = 
            Window(window.stop, window.stop + options.MEOW_SIZE);
    }
    // --------------------------------
    // End catch
    // --------------------------------
    if (prev_break + options.PEEK_SIZE < mask.count()) {
        auto window_1 = Window(prev_break, mask.count());
        ChangeModel result = 
            catch_model(workspace, solver, mask, window_1, CurveQA::End);
        results.models.push_back(result);
    }
    results.mask = std::move(mask);
    return results;
}

} // namespace ccd