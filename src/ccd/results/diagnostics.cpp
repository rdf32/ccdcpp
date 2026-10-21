#include "ccd/results/diagnostics.hpp"

namespace ccd
{

const char* to_string(ObserverEvent::Kind kind)
{
    switch (kind)
    {
        case ObserverEvent::Kind::Transition: return "transition";
        case ObserverEvent::Kind::Test:       return "test";
        case ObserverEvent::Kind::Drop:       return "drop";
        case ObserverEvent::Kind::Segment:    return "segment";
        case ObserverEvent::Kind::Exit:       return "exit";
    }
    return "?";
}

void TextObserver::event(index_t pixel, const ObserverEvent& e)
{
    std::lock_guard<std::mutex> lock(mutex_);
    out_ << "pixel " << pixel << ": " << to_string(e.kind);
    switch (e.kind)
    {
        case ObserverEvent::Kind::Transition:
            out_ << " " << e.phase << " -> " << e.next;
            if (!e.outcome.empty())
                out_ << " (" << e.outcome << ")";
            out_ << " window (" << e.window_start << ", " << e.window_stop << ") prev_break "
                 << e.prev_break << " n " << e.observations;
            break;
        case ObserverEvent::Kind::Test:
        case ObserverEvent::Kind::Drop:
            out_ << " " << e.day << " " << e.outcome;
            break;
        case ObserverEvent::Kind::Segment:
            out_ << " (" << e.day << " .. " << e.day_to << ") " << e.outcome << " n "
                 << e.observations;
            break;
        case ObserverEvent::Kind::Exit:
            out_ << " " << e.outcome;
            break;
    }
    if (!e.values.empty())
    {
        out_ << " [";
        for (std::size_t i = 0; i < e.values.size(); ++i)
            out_ << (i ? ", " : "") << e.values[i];
        out_ << "]";
    }
    out_ << '\n';
}

} // namespace ccd
