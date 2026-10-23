#include "arrays.hpp"

#include <array>
#include <stdexcept>
#include <string>
#include <tuple>

namespace
{

bool usable_strides(const py::array& a)
{
    for (py::ssize_t d = 0; d < a.ndim(); ++d)
    {
        if (a.strides(d) < 0 || a.strides(d) % a.itemsize() != 0)
            return false;
    }
    return true;
}

template<typename T>
bool takes_as_is(const py::array& a)
{
    return py::isinstance<py::array_t<T>>(a) && usable_strides(a);
}

template<typename T>
py::array as_c_array(const py::object& obj, const char* name)
{
    auto converted = py::array_t<T, py::array::c_style>::ensure(obj);

    if (!converted)
        throw py::type_error(
            std::string(name) + " cannot be converted to " +
            py::str(py::dtype::of<T>()).cast<std::string>() +
            " without an unsafe cast"
        );

    return std::move(converted);
}

//------------------------------------------------------------------------------
// `obj` as it is if it is an ndarray of one of the types Ts in a layout
// takes_as_is accepts, else converted to a C-contiguous array of the first.
//------------------------------------------------------------------------------
template<typename... Ts>
py::array accept(const py::object& obj, const char* name)
{
    if (py::isinstance<py::array>(obj))
    {
        auto a = py::reinterpret_borrow<py::array>(obj);

        if ((takes_as_is<Ts>(a) || ...))
            return a;
    }

    using First = std::tuple_element_t<0, std::tuple<Ts...>>;
    return as_c_array<First>(obj, name);
}

template<typename T, std::size_t Rank>
ccd::ArrayView<const T, Rank> strided_view(const py::array& a)
{
    std::array<std::size_t, Rank> shape{};
    std::array<std::size_t, Rank> strides{};

    for (std::size_t d = 0; d < Rank; ++d)
    {
        shape[d] = static_cast<std::size_t>(a.shape(d));
        strides[d] = static_cast<std::size_t>(a.strides(d)) / sizeof(T);
    }

    return ccd::ArrayView<const T, Rank>::strided(
        static_cast<const T*>(a.data()),
        shape,
        strides
    );
}

// `a`, which accept<T, Ts...>() returned, as the view of its element type in
// the variant Views.
template<typename Views, std::size_t Rank, typename T, typename... Ts>
Views view_of(const py::array& a)
{
    if constexpr (sizeof...(Ts) > 0)
    {
        if (!py::isinstance<py::array_t<T>>(a))
            return view_of<Views, Rank, Ts...>(a);
    }
    return strided_view<T, Rank>(a);
}

void require_rank(const py::array& a, py::ssize_t rank, const char* message)
{
    if (a.ndim() != rank)
        throw std::invalid_argument(message);
}

// The three arrays, accepted, rank-checked and viewed.
template<typename Spectral, typename Qa, std::size_t Rank>
Arrays<Spectral, Qa> take(
    const py::object& dates_obj,
    const py::object& spectral_obj,
    const py::object& qas_obj,
    bool with_table,
    const char* dates_shape,
    const char* spectral_shape,
    const char* qas_shape
)
{
    if (!with_table && py::isinstance<py::array_t<std::uint16_t>>(qas_obj))
        throw py::type_error(
            "qas is uint16, which holds raw QA values; give the data spec a qa_table "
            "(see ccdcpp.qa_tables) to decode them, or pass uint8 QA codes"
        );

    py::array dates = as_c_array<std::int64_t>(dates_obj, "dates");
    py::array spectral = accept<double, float, std::uint16_t>(spectral_obj, "spectral");
    py::array qas = with_table ? accept<std::uint16_t, std::uint8_t>(qas_obj, "qas")
                               : accept<std::uint8_t>(qas_obj, "qas");

    require_rank(dates, 1, dates_shape);
    require_rank(spectral, Rank + 1, spectral_shape);
    require_rank(qas, Rank, qas_shape);

    const auto d = ccd::ArrayView<const std::int64_t, 1>::contiguous(
        static_cast<const std::int64_t*>(dates.data()),
        {static_cast<ccd::index_t>(dates.shape(0))});
    const auto s = view_of<Spectral, Rank + 1, double, float, std::uint16_t>(spectral);
    const auto q = view_of<Qa, Rank, std::uint16_t, std::uint8_t>(qas);

    return {std::move(dates), std::move(spectral), std::move(qas), d, s, q};
}

} // namespace

PixelArrays pixel_arrays(
    const py::object& dates,
    const py::object& spectral,
    const py::object& qas,
    bool with_table
)
{
    auto taken = take<ccd::PixelSpectral, ccd::PixelQa, 1>(
        dates, spectral, qas, with_table, "dates must have shape (time)",
        "spectral must have shape (bands,time)", "qas must have shape (time)");

    const auto B = static_cast<ccd::index_t>(taken.spectral_array.shape(0));
    return PixelArrays{std::move(taken), B};
}

CubeArrays cube_arrays(
    const py::object& dates,
    const py::object& spectral,
    const py::object& qas,
    bool with_table
)
{
    auto taken = take<ccd::CubeSpectral, ccd::CubeQa, 3>(
        dates, spectral, qas, with_table, "dates must have shape (timesteps)",
        "spectral must have shape (height,width,bands,timesteps)",
        "qas must have shape (height,width,timesteps)");

    const auto extent = [&taken](py::ssize_t d)
    { return static_cast<ccd::index_t>(taken.spectral_array.shape(d)); };
    const ccd::index_t H = extent(0), W = extent(1), B = extent(2), T = extent(3);
    return CubeArrays{std::move(taken), H, W, B, T};
}
