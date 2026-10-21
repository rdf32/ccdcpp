#include "ccd/input/processing_mask.hpp"

namespace ccd
{

const char* name(Reason reason)
{
    switch (reason)
    {
        case Reason::Kept:            return "kept";
        case Reason::NoFit:           return "no_fit";
        case Reason::NoSegment:       return "no_segment";
        case Reason::Qa:              return "qa";
        case Reason::Thermal:         return "thermal";
        case Reason::BandRange:       return "band_range";
        case Reason::Green:           return "green";
        case Reason::DuplicateDate:   return "duplicate_date";
        case Reason::CustomStep:      return "custom_step";
        case Reason::WindowScreen:    return "window_screen";
        case Reason::LookbackOutlier: return "lookback_outlier";
        case Reason::Outlier:         return "outlier";
        case Reason::BeforeChange:    return "before_change";
        case Reason::Stored:          return "stored";
    }
    return "?";
}

} // namespace ccd
