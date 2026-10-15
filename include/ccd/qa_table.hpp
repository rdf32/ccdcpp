#pragma once

#include <array>
#include <cstdint>
#include <memory>

#include "ccd/types.hpp"
#include "ccd/harmonic/harmonic.hpp"

namespace ccd
{

//==============================================================================
//
// QaTable
//
// A lookup from one raw QA value -- a bitmask or a class number, as an
// archive stores it -- to one of the codes the procedures read
// (HarmonicOptions::QA_FILL, QA_CLEAR, QA_WATER, QA_SHADOW, QA_SNOW,
// QA_CLOUD). Indexed by the raw value, so it covers every uint16 value, and a
// uint8 value reads its first 256 entries.
//
// detect() and detect_cube() take one as a separate argument, never through
// HarmonicOptions: the options are copied once per pixel, and the table is
// 64 KiB that every pixel and thread shares read-only through a
// std::shared_ptr<const QaTable>. With a table, each pixel's T raw values are
// decoded into per-thread scratch before the procedures run, and the
// procedures see exactly what they would see had the caller passed the
// decoded codes as uint8. Without one, a uint8 `qas` is read as codes, as it
// is, and a uint16 `qas` is rejected.
//
// A table holds the codes of the HarmonicOptions it was built from. Building
// with one set of QA_* codes and running with another gives codes the
// procedures do not recognise; build with the options the run uses.
//
//==============================================================================

constexpr index_t QA_TABLE_SIZE = 65536;

struct QaTable
{
    std::array<std::uint8_t, QA_TABLE_SIZE> code;
};

using QaTablePtr = std::shared_ptr<const QaTable>;

//==============================================================================
//
// Table builders
//
// One per QA product. Each decodes in precedence order, most-excluding
// first, so that a value carrying several flags gets the code that keeps the
// observation out of the fit: fill, then cloud, then shadow, then snow, then
// water, then clear. A value the product does not define decodes to QA_FILL,
// which removes the observation from the fit and from the clear and snow
// percentages alike.
//
// The procedures keep an observation for the fit when its code is QA_CLEAR or
// QA_WATER, count QA_SNOW toward PermanentSnow and QA_FILL out of the valid
// total, and read no other code: QA_SHADOW and QA_CLOUD differ only in
// QA_CLOUD's count, which no decision reads.
//
// Each builder takes the HarmonicOptions whose QA_* codes it writes, so a
// caller that renumbers them gets a table that agrees with its run.
//
// The Landsat tables are the ones validated for production use. The HLS and
// Sentinel-2 tables follow the products' published definitions, are tested
// here, and decode as that documentation says on every value present in the
// real sample the HLS and Sentinel-2 presets were validated on (MGRS tile
// T16SGC, 2020-2023: 38 Fmask values; SCL classes 0 and 2-11). No production
// run has used them yet.
//
//==============================================================================

namespace qa_tables
{

//------------------------------------------------------------------------------
// Landsat Collection 2 QA_PIXEL (uint16).
//
//     bit 0       fill               -> QA_FILL
//     bit 3       cloud              -> QA_CLOUD
//     bit 1       dilated cloud      -> QA_CLOUD
//     bit 2       cirrus             -> QA_CLOUD
//     bit 4       cloud shadow       -> QA_SHADOW
//     bit 5       snow               -> QA_SNOW
//     bit 7       water              -> QA_WATER
//     otherwise                      -> QA_CLEAR
//
// Tested in that order. Bit 6 (clear) is not read: it is the complement of
// bits 1 and 3. Bits 8-15 are confidence pairs and are not read. Bit 2 is
// set on Landsat 8 and 9 only.
//------------------------------------------------------------------------------
QaTablePtr landsat_c2_qa_pixel(const HarmonicOptions& options = HarmonicOptions{});

//------------------------------------------------------------------------------
// Landsat Collection 1 pixel_qa (uint16), decoded as pyccd's qabitval does.
//
//     bit 0       fill               -> QA_FILL
//     bit 5       cloud              -> QA_CLOUD
//     bit 3       cloud shadow       -> QA_SHADOW
//     bit 4       snow               -> QA_SNOW
//     bit 2       water              -> QA_WATER
//     bit 1       clear              -> QA_CLEAR
//     bits 8, 9   both set (cirrus)  -> QA_CLEAR
//     bit 10      terrain occlusion  -> QA_CLEAR
//     otherwise                      -> QA_FILL
//
// Tested in that order. pyccd raises on a value none of these match; here it
// decodes to QA_FILL, so one undefined value removes one observation rather
// than the pixel.
//------------------------------------------------------------------------------
QaTablePtr landsat_c1_pixel_qa(const HarmonicOptions& options = HarmonicOptions{});

//------------------------------------------------------------------------------
// HLS v2.0 Fmask (uint8).
//
//     255         fill               -> QA_FILL
//     bit 1       cloud              -> QA_CLOUD
//     bit 2       adjacent to cloud
//                 or shadow          -> QA_CLOUD
//     bit 0       reserved           -> QA_CLOUD
//     bits 6-7    high aerosol (11)  -> QA_CLOUD, when exclude_high_aerosol
//     bit 3       cloud shadow       -> QA_SHADOW
//     bit 4       snow or ice        -> QA_SNOW
//     bit 5       water              -> QA_WATER
//     otherwise                      -> QA_CLEAR
//
// Tested in that order. The aerosol level is ignored unless
// exclude_high_aerosol is set. Values above 255 are not Fmask values and
// decode to QA_FILL. The HLS v2.0 User Guide lists bit 0 as reserved and not
// used in v2.0 (Fmask folds cirrus into cloud); it decodes to QA_CLOUD only
// should it ever be set.
//------------------------------------------------------------------------------
QaTablePtr hls_fmask(
    const HarmonicOptions& options = HarmonicOptions{},
    bool exclude_high_aerosol = false
);

//------------------------------------------------------------------------------
// Sentinel-2 L2A scene classification, SCL (uint8).
//
//     0  no data, 1 saturated or defective         -> QA_FILL
//     2  cast shadows, 3 cloud shadows             -> QA_SHADOW
//     4  vegetation, 5 not vegetated               -> QA_CLEAR
//     6  water                                     -> QA_WATER
//     7  unclassified, 8 cloud medium probability,
//     9  cloud high probability, 10 thin cirrus    -> QA_CLOUD
//     11 snow or ice                               -> QA_SNOW
//     otherwise                                    -> QA_FILL
//------------------------------------------------------------------------------
QaTablePtr sentinel2_scl(const HarmonicOptions& options = HarmonicOptions{});

} // namespace qa_tables

} // namespace ccd
