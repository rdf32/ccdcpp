#pragma once

#include <array>
#include <cstdint>
#include <memory>

#include "ccd/core/types.hpp"

namespace ccd
{

//==============================================================================
//
// QaTable
//
// A lookup from one raw QA value -- a bitmask or a class number, as an
// archive stores it -- to one of the codes the procedures read
// (Input::qa_fill, qa_clear, qa_water, qa_shadow, qa_snow, qa_cloud). Indexed
// by the raw value, so it covers every uint16 value, and a uint8 value reads
// its first 256 entries.
//
// An Input holds its table (Input::qa_table), a std::shared_ptr<const
// QaTable> that every pixel and thread shares read-only: the table is
// 64 KiB, and the Input is copied per call. With a table, each pixel's T raw
// values are decoded into per-thread scratch before the procedures run, and
// the procedures see exactly what they would see had the caller passed the
// decoded codes as uint8. Without one, a uint8 `qas` is read as codes, as it
// is, and a uint16 `qas` is rejected.
//
// A table holds the codes of the Input it was built from (qa_tables.hpp).
// Building with one set of qa_* codes and running with another gives codes
// the procedures do not recognise.
//
// A table is immutable: its digest, of every code, is taken once, when it is
// built, so the digest of an Input that holds one costs nothing per call.
//
//==============================================================================

constexpr index_t QA_TABLE_SIZE = 65536;

struct QaTable
{
    using Codes = std::array<std::uint8_t, QA_TABLE_SIZE>;

    explicit QaTable(const Codes& codes);

    // Entry v: the code of raw value v.
    const Codes code;

    // A digest of `code`: equal for equal tables, on any platform.
    const std::uint64_t digest;
};

using QaTablePtr = std::shared_ptr<const QaTable>;

} // namespace ccd
