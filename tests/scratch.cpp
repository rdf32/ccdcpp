#include "ccd/types.hpp"
#include "ccd/array_view.hpp"
#include "ccd/array_print.hpp"
#include "ccd/procedure/fit.hpp"

#include <vector>
#include <iostream>

using namespace ccd;

inline scalar_t span(
    ArrayView<const std::int64_t,1> dates,
    const Window& window
)
{
    assert(window.stop > window.start);

    return static_cast<scalar_t>(
        dates(window.stop - 1) -
        dates(window.start)
    );
}

int main(int argc, char* argv[]) {

    Window window(0, 5);

    const std::vector<std::int64_t> dates_storage{
        724387, 724419, 724451, 724483,
        724547, 724739, 724867, 724915,
        724931, 724947, 725075, 725091
    };
    ArrayView<const std::int64_t, 1> dates = 
        ArrayView<const std::int64_t, 1>::contiguous(
            dates_storage.data(),
          { dates_storage.size() }
        );

    std::cout << span(dates, window) << std::endl;

    auto dates_window = dates.slice(range(window.start, window.stop));
    print_array(dates_window);
    std::cout << dates_window(dates_window.size() - 1) - dates_window(0) << std::endl;
}