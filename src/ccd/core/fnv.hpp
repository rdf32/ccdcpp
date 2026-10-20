#pragma once

// Internal to the library: the hash behind the digests (inputs,
// configurations, resolutions) and the input fingerprints of the resume
// records.

#include <cstdint>
#include <cstring>
#include <string>

#include "ccd/core/array_view.hpp"
#include "ccd/core/band_list.hpp"
#include "ccd/core/types.hpp"

namespace ccd
{
namespace detail
{

//------------------------------------------------------------------------------
// FNV-1a over a canonical encoding: integers as little-endian 64-bit,
// floating point as the bits of its scalar_t, booleans as one byte, so the
// digest of equal values is the same whatever the platform's layout.
//------------------------------------------------------------------------------
class Fnv
{
public:

    void byte(std::uint8_t b)
    {
        hash_ = (hash_ ^ b) * 1099511628211ull;
    }

    void integer(std::int64_t v)
    {
        auto u = static_cast<std::uint64_t>(v);
        for (int i = 0; i < 8; ++i, u >>= 8)
            byte(static_cast<std::uint8_t>(u & 0xff));
    }

    void real(scalar_t v)
    {
        std::uint64_t bits;
        static_assert(sizeof(bits) == sizeof(v));
        std::memcpy(&bits, &v, sizeof(bits));
        integer(static_cast<std::int64_t>(bits));
    }

    void flag(bool v)
    {
        byte(v ? 1 : 0);
    }

    // Its length, then its bytes.
    void text(const std::string& s)
    {
        integer(static_cast<std::int64_t>(s.size()));
        for (const char c : s)
            byte(static_cast<std::uint8_t>(c));
    }

    void bands(const BandList& list)
    {
        integer(static_cast<std::int64_t>(list.size()));
        for (const index_t b : list)
            integer(static_cast<std::int64_t>(b));
    }

    std::uint64_t value() const
    {
        return hash_;
    }

private:

    std::uint64_t hash_ = 1469598103934665603ull;
};

//------------------------------------------------------------------------------
// input_fingerprint() of a (B, T) view of any element type: each observation
// dated on or before `through` whose QA code is not `qa_fill`, its date, code
// and every band's value as scalar_t.
//------------------------------------------------------------------------------
template<typename Value>
std::uint64_t fingerprint_of(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const Value, 2> spectral,
    ArrayView<const std::uint8_t, 1> qa_codes,
    std::uint8_t qa_fill,
    std::int64_t through
)
{
    Fnv h;
    for (index_t t = 0; t < dates.size() && dates(t) <= through; ++t)
    {
        if (qa_codes(t) == qa_fill)
            continue;
        h.integer(dates(t));
        h.integer(qa_codes(t));
        for (index_t b = 0; b < spectral.extent(0); ++b)
            h.real(static_cast<scalar_t>(spectral(b, t)));
    }
    return h.value();
}

} // namespace detail
} // namespace ccd
