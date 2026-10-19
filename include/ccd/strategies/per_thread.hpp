#pragma once

#include <memory>

namespace ccd
{

//------------------------------------------------------------------------------
// One thread's object made by a strategy, kept across pixels and made again
// when the run's strategy is another (another run's): get() is the object of
// `strategy`, made by make(*strategy) when it is not. Holding the strategy
// keeps it alive, so a strategy at the address of a freed one is never
// taken for it.
//------------------------------------------------------------------------------
template<class Strategy, class T>
class PerThread
{
public:

    template<class Make>
    T& get(const std::shared_ptr<const Strategy>& strategy, Make make)
    {
        if (owner_ != strategy)
        {
            object_ = make(*strategy);
            owner_ = strategy;
        }
        return *object_;
    }

    // The object last made; get() has made one.
    T& operator*() const { return *object_; }
    T* operator->() const { return object_.get(); }
    explicit operator bool() const { return object_ != nullptr; }

private:

    std::shared_ptr<const Strategy> owner_;
    std::unique_ptr<T> object_;
};

} // namespace ccd
