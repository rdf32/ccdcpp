#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <type_traits>

namespace ccd
{

// ArrayView slicing //
struct All {};

template<typename T>
struct Fixed {
    T value;
};

template<typename T>
struct Range
{
    T begin;
    T end;
};

inline constexpr All all() {
    return {};
}
template<typename T>
constexpr Fixed<T> fixed(T value) {
    return {value};
}
template<typename T>
constexpr Range<T> range(T begin, T end)
{
    return {begin, end};
}


template<typename T>
struct is_fixed : std::false_type {};

template<typename T>
struct is_fixed<Fixed<T>> : std::true_type {};

template<typename T>
struct is_range : std::false_type {};

template<typename T>
struct is_range<Range<T>> : std::true_type {};

template<typename... Args>
constexpr std::size_t count_fixed() {
    return (0 + ... + is_fixed<Args>::value);
}

// ArrayView Template //
template<typename T, std::size_t Rank>
class ArrayView {
public:

    using value_type  = T;
    using index_type  = std::size_t;
    using shape_type  = std::array<index_type, Rank>;
    using stride_type = std::array<index_type, Rank>;

    //===============================================================
    // Converting constructor:
    //
    // ArrayView<T, Rank> -> ArrayView<const T, Rank>
    //
    // Allows:
    //     ArrayView<double,2>
    //         |
    //         v
    //     ArrayView<const double,2>
    //
    //===============================================================
    
    // Constructors //
    template<
        typename U,
        typename = std::enable_if_t<
            std::is_const_v<T> &&
            std::is_same_v<T, const U>
        >
    >
    ArrayView(const ArrayView<U, Rank>& other)
        :
        _data(other.data()),
        _shape(other.shape()),
        _strides(other.strides())
    {}

    static ArrayView contiguous(
        value_type* data,
        const shape_type& shape
    ) noexcept
    {
        stride_type strides;
        strides[Rank - 1] = 1;

        for (std::size_t i = Rank - 1; i > 0; --i) {
            strides[i - 1] = strides[i] * shape[i];
        }
        return ArrayView(data, shape, strides);
    }

    static ArrayView strided(
        value_type* data,
        const shape_type& shape,
        const stride_type& strides
    ) noexcept
    {
        return ArrayView(data, shape, strides);
    }


    // Operators //
    inline value_type& operator[](index_type offset) noexcept
    {
        return _data[offset];
    }

    inline const value_type& operator[](index_type offset) const noexcept
    {
        return _data[offset];
    }

    template<typename... Indices>
    constexpr value_type& operator()(Indices... indices) noexcept {
        static_assert(
            sizeof...(Indices) == Rank,
            "Number of indices must match array rank"
        );

        shape_type idx {static_cast<index_type>(indices)...};

        return _data[compute_offset(idx)];
    }

    template<typename... Indices>
    constexpr const value_type& operator()(Indices... indices) const noexcept {
        static_assert(
            sizeof...(Indices) == Rank,
            "Number of indices must match array rank"
        );

        shape_type idx {static_cast<index_type>(indices)...};

        return _data[compute_offset(idx)];
    }


    // Data access (raw pointer) //
    inline value_type* data() noexcept {
        return _data;
    }
    inline const value_type* data() const noexcept {
        return _data;
    }


    // Element access methods //
    constexpr const value_type& at(shape_type& idx) const noexcept {
        return _data[compute_offset(idx)];
    }

    constexpr value_type& at(shape_type& idx) noexcept {
        return _data[compute_offset(idx)];
    }

    template<typename... Args>
    auto slice(Args... args) const noexcept {
        static_assert(
            sizeof...(Args) == Rank,
            "Must provide one slide argument per dimension."
        );

        constexpr std::size_t NewRank = Rank - count_fixed<Args...>();

        value_type* ptr = _data;

        std::array<index_type, NewRank> new_shape{};
        std::array<index_type, NewRank> new_strides{};

        std::size_t out_dim = 0;
        std::size_t dim = 0;

        auto process = [&](auto arg) {
            using Arg = std::decay_t<decltype(arg)>;

            if constexpr (is_fixed<Arg>::value) 
            {
                assert(arg.value < _shape[dim]);
                ptr += arg.value * _strides[dim];
            } 
            else if constexpr (is_range<Arg>::value)
            {
                assert(arg.begin <= arg.end);
                assert(arg.end <= _shape[dim]);

                ptr += arg.begin * _strides[dim];

                new_shape[out_dim] = arg.end - arg.begin;
                new_strides[out_dim] = _strides[dim];

                ++out_dim;
            }
            else {
                new_shape[out_dim] = _shape[dim];
                new_strides[out_dim] = _strides[dim];
                ++out_dim;
            }
            ++dim;
        };
        (process(args), ...);

        return ArrayView<value_type, NewRank>::strided(
            ptr,
            new_shape,
            new_strides
        );
    }

    // Attribute methods //
    const shape_type& shape() const noexcept {
        return _shape;
    }
    const stride_type& strides() const noexcept {
        return _strides;
    }

    constexpr index_type size() const noexcept {
        index_type total = 1;
        for (auto dim : _shape) {
            total *= dim;
        }
        return total;
    }

    bool empty() const noexcept {
        return size() == 0;
    }

    index_type extent(index_type dim) const noexcept {
        assert(dim < Rank);
        return _shape[dim];
    }

    index_type stride(index_type dim) const noexcept {
        assert(dim < Rank);
        return _strides[dim];
    }
    
    bool is_contiguous() const noexcept {
        index_type expected = 1;
        for (std::size_t i = Rank; i-- > 0;) {
            if (_strides[i] != expected) {
                return false;
            }
            expected *= _shape[i];
        }
        return true;
    }

    // Transformation methods //
    ArrayView transpose(
        const std::array<std::size_t,
        Rank>& permutation
    ) const noexcept {
        
        shape_type new_shape{};
        stride_type new_strides{};

        std::array<bool, Rank> seen{};
        for (std::size_t i = 0; i < Rank; ++i) {
            assert(permutation[i] < Rank);
            assert(!seen[permutation[i]]);

            seen[permutation[i]] = true;
            new_shape[i] = _shape[permutation[i]];
            new_strides[i] = _strides[permutation[i]];
        }
        return ArrayView::strided(
            _data,
            new_shape,
            new_strides
        );
    }



private:
    //-------------------------------------------------------------------------
    // Convert N-dimensional index -> flat memory index
    //
    // Convert N-dimensional index -> flat memory offset
    //
    // Supports contiguous and arbitrary strided layouts
    //
    // Example:
    //
    // (time, band, row, col)
    //
    // index =
    // (((time * bands + band) * rows + row) * cols + col)
    //
    //-------------------------------------------------------------------------
    value_type* _data;
    shape_type _shape;
    stride_type _strides;

    ArrayView(
        value_type* data,
        const shape_type& shape,
        const stride_type& strides
    ) noexcept 
        :
        _data(data),
        _shape(shape),
        _strides(strides){}

    index_type compute_offset(const shape_type& index) const noexcept {
        index_type off = 0;
        for (std::size_t i = 0; i < Rank; ++i) {
            assert(index[i] < _shape[i]);
            off += index[i] * _strides[i];
        }
        return off;
    }
};

}
