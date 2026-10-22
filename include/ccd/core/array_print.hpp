#pragma once

#include <algorithm>
#include <array>
#include <iostream>
#include <string>

#include "ccd/core/array_view.hpp"

namespace ccd
{

// Print metadata
template<typename T, std::size_t Rank>
void print_info(
    const ArrayView<T, Rank>& view,
    std::ostream& os = std::cout
) {
    os << "ArrayView<"
       << typeid(T).name()
       << ", Rank=" << Rank << ">\n";

    os << "Shape   : (";
    for (std::size_t i = 0; i < Rank; ++i)
    {
        if (i) os << ", ";
        os << view.extent(i);
    }
    os << ")\n";

    os << "Strides : (";
    for (std::size_t i = 0; i < Rank; ++i)
    {
        if (i) os << ", ";
        os << view.stride(i);
    }
    os << ")\n";

    os << "Size    : " << view.size() << '\n';
    os << "Contiguous : "
       << std::boolalpha
       << view.is_contiguous()
       << '\n';
}

// Print raw memory (NOT logical order)
// Useful for verifying contiguous layouts.
template<typename T, std::size_t Rank>
void print_memory(
    const ArrayView<T, Rank>& view,
    std::size_t max_values = 100,
    std::ostream& os = std::cout
) {
    std::size_t n = std::min(view.size(), max_values);

    os << "[";
    for (std::size_t i = 0; i < n; ++i) {
        if (i) {
            os << ", ";
        }
        os << view.data()[i];
    }

    if (n < view.size()) {
        os << ", ...";
    }
    os << "]\n";
}

template<typename T, std::size_t Rank>
void print_flat(
    const ArrayView<T, Rank>& view,
    std::size_t max_values = 100,
    std::ostream& os = std::cout)
{
    std::size_t count = 0;
    std::array<std::size_t, Rank> index{};

    os << "[";

    auto recurse = [&](auto&& self, std::size_t dim) -> void {
        if (count >= max_values) {
            return;
        }

        if (dim == Rank) {
            if (count) {
                os << ", ";
            }
            os << view.at(index);
            ++count;
            return;
        }

        for (std::size_t i = 0; i < view.extent(dim); ++i) {
            index[dim] = i;
            self(self, dim + 1);

            if (count >= max_values) {
                return;
            }
        }
    };

    recurse(recurse, 0);

    if (count < view.size()) {
        os << ", ...";
    }
    
    os << "]\n";
}

// Recursive helper
template<typename T, std::size_t Rank>
void print_recursive(
    const ArrayView<T, Rank>& view,
    typename ArrayView<T, Rank>::shape_type& idx,
    std::size_t dim,
    std::ostream& os,
    std::size_t indent
) {
    if (dim == Rank) {
        os << view.at(idx);
        return;
    }
    os << "[";

    if (dim != Rank - 1)
        os << '\n';

    for (std::size_t i = 0; i < view.extent(dim); ++i) {

        idx[dim] = i;
        if (dim != Rank - 1) {
            os << std::string((indent + 1) * 2, ' ');
        }

        print_recursive(view, idx, dim + 1, os, indent + 1);

        if (i + 1 != view.extent(dim)) {
            os << ",";
        }

        if (dim != Rank - 1) {
            os << '\n';
        } else if (i + 1 != view.extent(dim)) {
            os << " ";
        }
    }
    if (dim != Rank - 1) {
        os << std::string(indent * 2, ' ');
    }
    os << "]";
}

//-------------------------------------------------------------
// Pretty multidimensional print
//-------------------------------------------------------------
template<typename T, std::size_t Rank>
void print_array(
    const ArrayView<T, Rank>& view,
    std::ostream& os = std::cout
) {
    typename ArrayView<T, Rank>::shape_type idx{};
    print_recursive(view, idx, 0, os, 0);
    os << '\n';
}

//-------------------------------------------------------------
// ostream support
//-------------------------------------------------------------
template<typename T, std::size_t Rank>
std::ostream&
operator<<(
    std::ostream& os,
    const ArrayView<T, Rank>& view
) {
    print(view, os);
    return os;
}

}