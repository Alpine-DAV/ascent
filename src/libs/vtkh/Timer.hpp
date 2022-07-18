#ifndef VTKH_TIMER_HPP
#define VTKH_TIMER_HPP

#include <chrono>
#include <vtkh/vtkh_exports.h>

namespace vtkh
{

class VTKH_API Timer
{

public:

    explicit Timer();
            ~Timer();
    void     reset();
    float    elapsed() const;

private:
    std::chrono::high_resolution_clock::time_point m_start;
};

}; // namespace vtkh

#endif
