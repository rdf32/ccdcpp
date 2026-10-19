#pragma once

#include <vector>

#include "ccd/config/params.hpp"
#include "ccd/strategies/stages.hpp"

namespace ccd
{
namespace strategies
{

//------------------------------------------------------------------------------
// start_search "pyccd": pyccd's search. A candidate is meow_size
// observations from where the search is; it needs meow_size more after it
// to begin (room: window.stop + meow_size <= n) and to go on (exhausted
// unless window.stop + meow_size < n), and a candidate spanning fewer than
// day_delta days grows. A shortened candidate is usable when it still spans
// day_delta days and holds meow_size observations.
//------------------------------------------------------------------------------
struct PyccdSearch final : StartSearch
{
    struct Params
    {
        index_t meow_size = 12;
        std::int64_t day_delta = 365;
    };

    static const std::vector<config::Field<Params>>& schema();

    explicit PyccdSearch(const Params& p) : p_(p) {}

    bool too_few(index_t observations) const override;
    Window first(index_t at) const override;
    bool room(const Window& window, index_t observations) const override;
    bool exhausted(const Window& window, index_t observations) const override;
    bool short_span(ArrayView<const std::int64_t, 1> dates) const override;
    bool usable(ArrayView<const std::int64_t, 1> dates) const override;
    index_t min_window() const override { return p_.meow_size; }

private:
    Params p_;
};

} // namespace strategies
} // namespace ccd
