#include "ccd/input/qa_table.hpp"

#include "../core/fnv.hpp"

namespace ccd
{

namespace
{

std::uint64_t digest_of(const QaTable::Codes& codes)
{
    detail::Fnv h;
    for (const std::uint8_t code : codes)
        h.byte(code);
    return h.value();
}

} // namespace

QaTable::QaTable(const Codes& codes) : code(codes), digest(digest_of(codes)) {}

} // namespace ccd
