#include "ccd/procedure/standard.hpp"

#include <iostream>
#include <algorithm>
#include <vector>

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

bool initialize(
    const HarmonicWorkspace& workspace,
    LassoSolver& solver,
    std::vector<scalar_t>& variogram,
    ProcessingMask& mask,
    Window& window
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
        // Build result
        //----------------------------------------------------------
        std::vector<LassoResult> bands;
        bands.reserve(masked_spectral.slice(all(), range(window.start, window.stop)).extent(0));

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
            bands.push_back(band_result);
        }

        //----------------------------------------------------------------------
        // Stability check
        //----------------------------------------------------------------------
        if (stable(
                masked_dates.slice(range(window.start, window.stop)),
                variogram,
                bands,
                options.CHANGE_THRESHOLD,
                options.DETECTION_BANDS))
        {
            std:: cout << "Stable start found: " << window.start << ", " << window.stop << std::endl;
            print_lasso_results(bands);
            return true;
        }

        window.shift();
        std:: cout << "Unstable model, shift window to: " << window.start << ", " << window.stop << std::endl;
    }

    return false;
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

    const auto masked_dates = 
        masked_data.dates_view();

    const auto masked_spectral = 
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
    bool start = false;
    index_t previous_break = 0;
    auto window = Window(0, options.MEOW_SIZE);

    while(window.stop <= masked_dates.size() - options.MEOW_SIZE)
    {
        auto initialized = 
            initialize(workspace, solver, variogram, mask, window);

        if (!initialized) {
            break;
        }

        std::cout << "window: " << std::endl;
        std::cout << window.start << ", " << window.stop << std::endl;

        std::cout << "mask: " << std::endl;
        for (ccd::index_t i = 0; i < mask.size(); ++i)
        {
            std::cout << static_cast<int>(mask[i]) << ", ";
        }
        std::cout << '\n';

        break;
        // if (ctx.window.start > ctx.previous_break)
        // {
        //     lookback(ctx);
        // }

        // if (ctx.window.start - ctx.previous_break > 
        //     options.PEEK_SIZE 
        //     && ctx.start
        // )
        // {
        //     ChangeModel result;
        //     catch_model(
        //         ctx,
        //         Window(ctx.previous_break, ctx.window.start),
        //         CurveQA::Start,
        //         result
        //     );
        //     // append result
        //     ctx.start = false;
        //     results.models.push_back(result);
        // }

        // ChangeModel result;
        // lookforward(
        //     ctx, 
        //     result
        // );
        // // store result
        // results.models.push_back(result);

        // //--------------------------------------------------
        // // iterate
        // //--------------------------------------------------
        // ctx.window = Window(
        //     ctx.window.stop,
        //     ctx.window.stop + options.MEOW_SIZE
        // );

        // // --------------------------------
        // // End catch
        // // --------------------------------
        // if (ctx.previous_break + options.PEEK_SIZE < workspace.active_count()) {
            
        //     ChangeModel result;
        //     catch_model(
        //         ctx, 
        //         Window(ctx.previous_break, workspace.active_count()),
        //         CurveQA::End,
        //         result
        //     );
        //     results.models.push_back(result);
        // }
    }
    // results.mask = ctx.mask;
    // return results;
}

} // namespace ccd