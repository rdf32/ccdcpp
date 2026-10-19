#include "ccd/input/qa_table.hpp"

namespace ccd
{

namespace qa_tables
{

namespace
{

bool bit(std::uint32_t value, unsigned offset)
{
    return (value >> offset) & 1u;
}

// A table whose entry v is decode(v), for every uint16 value.
template<typename Decode>
QaTablePtr build(Decode decode)
{
    auto table = std::make_shared<QaTable>();

    for (std::uint32_t value = 0; value < QA_TABLE_SIZE; ++value)
        table->code[value] = decode(value);

    return table;
}

} // namespace

QaTablePtr landsat_c2_qa_pixel(const HarmonicOptions& o)
{
    return build([&o](std::uint32_t v) -> std::uint8_t
    {
        if (bit(v, 0)) return o.QA_FILL;
        if (bit(v, 3)) return o.QA_CLOUD;
        if (bit(v, 1)) return o.QA_CLOUD;
        if (bit(v, 2)) return o.QA_CLOUD;
        if (bit(v, 4)) return o.QA_SHADOW;
        if (bit(v, 5)) return o.QA_SNOW;
        if (bit(v, 7)) return o.QA_WATER;
        return o.QA_CLEAR;
    });
}

QaTablePtr landsat_c1_pixel_qa(const HarmonicOptions& o)
{
    return build([&o](std::uint32_t v) -> std::uint8_t
    {
        if (bit(v, 0)) return o.QA_FILL;
        if (bit(v, 5)) return o.QA_CLOUD;
        if (bit(v, 3)) return o.QA_SHADOW;
        if (bit(v, 4)) return o.QA_SNOW;
        if (bit(v, 2)) return o.QA_WATER;
        if (bit(v, 1)) return o.QA_CLEAR;
        if (bit(v, 8) && bit(v, 9)) return o.QA_CLEAR;
        if (bit(v, 10)) return o.QA_CLEAR;
        return o.QA_FILL;
    });
}

QaTablePtr hls_fmask(const HarmonicOptions& o, bool exclude_high_aerosol)
{
    return build([&o, exclude_high_aerosol](std::uint32_t v) -> std::uint8_t
    {
        if (v >= 255) return o.QA_FILL;
        if (bit(v, 1)) return o.QA_CLOUD;
        if (bit(v, 2)) return o.QA_CLOUD;
        if (bit(v, 0)) return o.QA_CLOUD;
        if (exclude_high_aerosol && (v >> 6) == 3u) return o.QA_CLOUD;
        if (bit(v, 3)) return o.QA_SHADOW;
        if (bit(v, 4)) return o.QA_SNOW;
        if (bit(v, 5)) return o.QA_WATER;
        return o.QA_CLEAR;
    });
}

QaTablePtr sentinel2_scl(const HarmonicOptions& o)
{
    return build([&o](std::uint32_t v) -> std::uint8_t
    {
        switch (v)
        {
            case 2: case 3:                 return o.QA_SHADOW;
            case 4: case 5:                 return o.QA_CLEAR;
            case 6:                         return o.QA_WATER;
            case 7: case 8: case 9: case 10: return o.QA_CLOUD;
            case 11:                        return o.QA_SNOW;
            default:                        return o.QA_FILL;
        }
    });
}

} // namespace qa_tables

} // namespace ccd
