#ifndef LOGGING_TIMER_HPP
#define LOGGING_TIMER_HPP

#include <chrono>

namespace logging
{

class Timer
{

public:

    explicit Timer();
            ~Timer();
    void     reset();
    float    elapsed() const;

private:
    std::chrono::high_resolution_clock::time_point m_start;
};

}; 

#endif
