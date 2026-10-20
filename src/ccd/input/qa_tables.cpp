#include "ccd/input/qa_tables.hpp"

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
    QaTable::Codes codes;

    for (std::uint32_t value = 0; value < QA_TABLE_SIZE; ++value)
        codes[value] = decode(value);

    return std::make_shared<const QaTable>(codes);
}

} // namespace

QaTablePtr landsat_c2_qa_pixel(const Input& o)
{
    return build([&o](std::uint32_t v) -> std::uint8_t
    {
        if (bit(v, 0)) return o.qa_fill;
        if (bit(v, 3)) return o.qa_cloud;
        if (bit(v, 1)) return o.qa_cloud;
        if (bit(v, 2)) return o.qa_cloud;
        if (bit(v, 4)) return o.qa_shadow;
        if (bit(v, 5)) return o.qa_snow;
        if (bit(v, 7)) return o.qa_water;
        return o.qa_clear;
    });
}

QaTablePtr landsat_c1_pixel_qa(const Input& o)
{
    return build([&o](std::uint32_t v) -> std::uint8_t
    {
        if (bit(v, 0)) return o.qa_fill;
        if (bit(v, 5)) return o.qa_cloud;
        if (bit(v, 3)) return o.qa_shadow;
        if (bit(v, 4)) return o.qa_snow;
        if (bit(v, 2)) return o.qa_water;
        if (bit(v, 1)) return o.qa_clear;
        if (bit(v, 8) && bit(v, 9)) return o.qa_clear;
        if (bit(v, 10)) return o.qa_clear;
        return o.qa_fill;
    });
}

QaTablePtr hls_fmask(const Input& o, bool exclude_high_aerosol)
{
    return build([&o, exclude_high_aerosol](std::uint32_t v) -> std::uint8_t
    {
        if (v >= 255) return o.qa_fill;
        if (bit(v, 1)) return o.qa_cloud;
        if (bit(v, 2)) return o.qa_cloud;
        if (bit(v, 0)) return o.qa_cloud;
        if (exclude_high_aerosol && (v >> 6) == 3u) return o.qa_cloud;
        if (bit(v, 3)) return o.qa_shadow;
        if (bit(v, 4)) return o.qa_snow;
        if (bit(v, 5)) return o.qa_water;
        return o.qa_clear;
    });
}

QaTablePtr sentinel2_scl(const Input& o)
{
    return build([&o](std::uint32_t v) -> std::uint8_t
    {
        switch (v)
        {
            case 2: case 3:                 return o.qa_shadow;
            case 4: case 5:                 return o.qa_clear;
            case 6:                         return o.qa_water;
            case 7: case 8: case 9: case 10: return o.qa_cloud;
            case 11:                        return o.qa_snow;
            default:                        return o.qa_fill;
        }
    });
}

} // namespace qa_tables

} // namespace ccd
