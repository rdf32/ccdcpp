#pragma once

#include "ccd/input/input.hpp"
#include "ccd/input/qa_table.hpp"

namespace ccd
{

//==============================================================================
//
// Table builders
//
// One QaTable (qa_table.hpp) per QA product. Each decodes in precedence
// order, most-excluding first, so that a value carrying several flags gets
// the code that keeps the observation out of the fit: fill, then cloud, then shadow, then snow, then
// water, then clear. A value the product does not define decodes to qa_fill,
// which removes the observation from the fit and from the clear and snow
// percentages alike.
//
// The procedures keep an observation for the fit when its code is qa_clear or
// qa_water, count qa_snow toward PermanentSnow and qa_fill out of the valid
// total, and read no other code: qa_shadow and qa_cloud differ only in
// qa_cloud's count, which no decision reads.
//
// Each builder takes the Input whose qa_* codes it writes, so a caller that
// renumbers them gets a table that agrees with its run; the table then goes
// into that Input's qa_table.
//
// The Landsat tables are the ones validated for production use. The HLS and
// Sentinel-2 tables follow the products' published definitions, are tested
// here, and decode as that documentation says on every value present in the
// real sample the HLS and Sentinel-2 presets were validated on (MGRS tile
// T16SGC, 2020-2023: 43 Fmask values, 38 in L30 and 43 in S30; SCL classes 0
// and 2-11). No production run has used them yet.
//
//==============================================================================

namespace qa_tables
{

//------------------------------------------------------------------------------
// Landsat Collection 2 QA_PIXEL (uint16).
//
//     bit 0       fill               -> qa_fill
//     bit 3       cloud              -> qa_cloud
//     bit 1       dilated cloud      -> qa_cloud
//     bit 2       cirrus             -> qa_cloud
//     bit 4       cloud shadow       -> qa_shadow
//     bit 5       snow               -> qa_snow
//     bit 7       water              -> qa_water
//     otherwise                      -> qa_clear
//
// Tested in that order. Bit 6 (clear) is not read: it is the complement of
// bits 1 and 3. Bits 8-15 are confidence pairs and are not read. Bit 2 is
// set on Landsat 8 and 9 only.
//------------------------------------------------------------------------------
QaTablePtr landsat_c2_qa_pixel(const Input& input = Input{});

//------------------------------------------------------------------------------
// Landsat Collection 1 pixel_qa (uint16), decoded as pyccd's qabitval does.
//
//     bit 0       fill               -> qa_fill
//     bit 5       cloud              -> qa_cloud
//     bit 3       cloud shadow       -> qa_shadow
//     bit 4       snow               -> qa_snow
//     bit 2       water              -> qa_water
//     bit 1       clear              -> qa_clear
//     bits 8, 9   both set (cirrus)  -> qa_clear
//     bit 10      terrain occlusion  -> qa_clear
//     otherwise                      -> qa_fill
//
// Tested in that order. pyccd raises on a value none of these match; here it
// decodes to qa_fill, so one undefined value removes one observation rather
// than the pixel.
//------------------------------------------------------------------------------
QaTablePtr landsat_c1_pixel_qa(const Input& input = Input{});

//------------------------------------------------------------------------------
// HLS v2.0 Fmask (uint8).
//
//     255         fill               -> qa_fill
//     bit 1       cloud              -> qa_cloud
//     bit 2       adjacent to cloud
//                 or shadow          -> qa_cloud
//     bit 0       reserved           -> qa_cloud
//     bits 6-7    high aerosol (11)  -> qa_cloud, when exclude_high_aerosol
//     bit 3       cloud shadow       -> qa_shadow
//     bit 4       snow or ice        -> qa_snow
//     bit 5       water              -> qa_water
//     otherwise                      -> qa_clear
//
// Tested in that order. The aerosol level is ignored unless
// exclude_high_aerosol is set. Values above 255 are not Fmask values and
// decode to qa_fill. The HLS v2.0 User Guide lists bit 0 as reserved and not
// used in v2.0 (Fmask folds cirrus into cloud); it decodes to qa_cloud only
// should it ever be set.
//------------------------------------------------------------------------------
QaTablePtr hls_fmask(
    const Input& input = Input{},
    bool exclude_high_aerosol = false
);

//------------------------------------------------------------------------------
// Sentinel-2 L2A scene classification, SCL (uint8).
//
//     0  no data, 1 saturated or defective         -> qa_fill
//     2  cast shadows, 3 cloud shadows             -> qa_shadow
//     4  vegetation, 5 not vegetated               -> qa_clear
//     6  water                                     -> qa_water
//     7  unclassified, 8 cloud medium probability,
//     9  cloud high probability, 10 thin cirrus    -> qa_cloud
//     11 snow or ice                               -> qa_snow
//     otherwise                                    -> qa_fill
//------------------------------------------------------------------------------
QaTablePtr sentinel2_scl(const Input& input = Input{});

} // namespace qa_tables

} // namespace ccd
