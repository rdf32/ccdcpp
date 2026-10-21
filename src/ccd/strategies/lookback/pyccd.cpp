#include "ccd/strategies/lookback/pyccd.hpp"

#include <cassert>

#include "../../core/logger.hpp"
#include "ccd/strategies/pyccd/peek_statistics.hpp"
#include "ccd/strategies/model.hpp"

namespace ccd
{
namespace strategies
{

namespace
{

// One thread's Lookback state: the pixel's peek and threshold (prepare()),
// and buffers kept for their capacity; every one is rewritten before it is
// read.
class LookbackScratch final : public Scratch
{
public:
    index_t peek = 0;
    scalar_t change_threshold = 0;
    ChangeThresholdMemo thresholds;
    std::vector<std::int64_t> peek_dates;
    std::vector<scalar_t> peek_spectral;
    std::vector<std::vector<scalar_t>> residuals;   // by band; detection bands written
    std::vector<scalar_t> comp_rmse;                // by position in the detection bands
    std::vector<scalar_t> magnitude;
};

// The observations of `peek`, nearest the window first, copied into
// s.peek_dates (p) and s.peek_spectral (B, p, row-major).
void reverse_peek(
    ArrayView<const std::int64_t, 1> masked_dates,
    ArrayView<const scalar_t, 2> masked_spectral,
    const ReversePeek& peek_window,
    LookbackScratch& s
)
{
    std::vector<std::int64_t>& peek_dates = s.peek_dates;
    std::vector<scalar_t>& peek_spectral = s.peek_spectral;

    peek_dates.clear();
    peek_dates.reserve(s.peek);
    for (std::ptrdiff_t i = peek_window.start; i > peek_window.stop; --i)
    {
        peek_dates.push_back(masked_dates[i]);
    }

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
}

} // namespace

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

const std::vector<config::Field<PyccdLookback::Params>>& PyccdLookback::schema()
{
    using config::Unit;
    static const std::vector<config::Field<Params>> s = {
        {"change_threshold", &Params::change_threshold, Unit::None, config::any(),
         "the threshold for the change test's reference peek (or `peek`)", config::Source::Chi2,
         "change_probability"},
        {"change_probability", &Params::change_probability, Unit::None,
         config::open_interval(0, 1), "the probability a longer peek's threshold keeps",
         config::Source::Shared, "change_probability"},
        {"outlier_threshold", &Params::outlier_threshold, Unit::None, config::any(),
         "the observation before the window is dropped above this", config::Source::Chi2,
         "outlier_probability"},
        {"peek", &Params::peek, Unit::Observations, config::at_least(1),
         "a peek of N, change_threshold for it; \"change_test\": the change test's",
         config::Source::Own, nullptr, std::nullopt, config::Keyword{"change_test", 0}},
    };
    return s;
}

Capabilities PyccdLookback::needs() const
{
    return Capability::PeekResiduals | Capability::ModelRmse;
}

std::unique_ptr<Scratch> PyccdLookback::make_scratch() const
{
    return std::make_unique<LookbackScratch>();
}

void PyccdLookback::prepare(const PeekPlan& peek, index_t detection_bands,
                            Scratch& scratch) const
{
    LookbackScratch& s = static_cast<LookbackScratch&>(scratch);
    const ThresholdPeek at = threshold_peek(peek, p_.peek);
    s.peek = at.peek;
    s.change_threshold = s.thresholds(at.peek, at.reference, p_.change_threshold,
                                      detection_bands, p_.change_probability);
}

LookbackJudgement PyccdLookback::judge(Model& model, ArrayView<const std::int64_t, 1> masked_dates,
                                       ArrayView<const scalar_t, 2> masked_spectral,
                                       const Window& window, index_t prev_break,
                                       const BandNoise& variogram,
                                       const BandList& detection_bands, Scratch& scratch) const
{
    LookbackScratch& s = static_cast<LookbackScratch&>(scratch);

    // Indexed by band; only the detection bands' are written.
    std::vector<std::vector<scalar_t>>& residuals = s.residuals;
    residuals.resize(masked_spectral.extent(0));
    // Indexed by position in the detection bands.
    std::vector<scalar_t>& comp_rmses = s.comp_rmse;
    comp_rmses.resize(detection_bands.size());
    std::vector<scalar_t>& magnitude = s.magnitude;

    assert(window.start > prev_break);

    const ReversePeek peek_window =
        lookback_peek(window.start, prev_break, s.peek);

    reverse_peek(masked_dates, masked_spectral, peek_window, s);

    auto masked_dates_window =
        ArrayView<const std::int64_t, 1>::contiguous(
            s.peek_dates.data(),
            {s.peek_dates.size()}
        );

    auto masked_spectral_window =
        ArrayView<const scalar_t, 2>::contiguous(
            s.peek_spectral.data(),
            {masked_spectral.extent(0), s.peek_dates.size()}
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
        variogram,
        comp_rmses,
        magnitude
    );

    const auto before = static_cast<index_t>(peek_window.start);

    //--------------------------------------------------------------------------
    // Change detected: the window stays where it is
    //--------------------------------------------------------------------------
    if (detect_change(magnitude, s.change_threshold))
    {
        LOG_DEBUG("Change before " << window.start);
        return LookbackJudgement{LookbackVerdict::Change, before};
    }
    //--------------------------------------------------------------------------
    // Outlier detected: the observation immediately before the window
    //--------------------------------------------------------------------------
    else if (detect_outlier(magnitude[0], p_.outlier_threshold))
    {
        LOG_DEBUG("Outlier dropped at " << peek_window.start);
        return LookbackJudgement{LookbackVerdict::Drop, before};
    }
    //--------------------------------------------------------------------------
    // Otherwise it joins the window
    //--------------------------------------------------------------------------
    LOG_DEBUG("Included " << peek_window.start);
    return LookbackJudgement{LookbackVerdict::Include, before};
}

} // namespace strategies
} // namespace ccd
