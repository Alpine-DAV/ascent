#ifndef DRAY_TIMER_HPP
#define DRAY_TIMER_HPP

#include <chrono>

namespace dray
{

class Timer 
{

  typedef std::chrono::high_resolution_clock high_resolution_clock;
  typedef std::chrono::nanoseconds nanoseconds;
  typedef std::chrono::duration<float> fsec;

public:
    explicit Timer()
    {
      reset();
    }

    void reset()
    {
      m_start = high_resolution_clock::now();
    }

    float elapsed() const
    {
       auto ftime  = std::chrono::duration_cast<fsec>(high_resolution_clock::now() - m_start);
       return ftime.count();
    }

private:
    high_resolution_clock::time_point m_start;
};    

} // namespace dray
#endif


