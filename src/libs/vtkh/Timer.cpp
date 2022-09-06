#include <vtkh/Timer.hpp>

using namespace std::chrono;

namespace vtkh
{

Timer::Timer()
{
  reset();
}

Timer::~Timer()
{

}

void
Timer::reset()
{
  m_start = high_resolution_clock::now();
}

float
Timer::elapsed() const
{
  return duration_cast<duration<float>>(high_resolution_clock::now() - m_start).count();
}

}; // namespace vtkh
