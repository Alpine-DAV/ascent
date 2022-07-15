#ifndef VTK_H_MUTEX_HPP
#define VTK_H_MUTEX_HPP

#include <vtkh/vtkh_exports.h>
#include <memory>

namespace vtkh
{

//Mutex class for both openmp and std::mutex
class VTKH_API Mutex
{

public:
  Mutex();
  ~Mutex();
  void Lock();
  void Unlock();
private:
  struct InternalsType;
  std::shared_ptr<InternalsType> m_internals;
};

} //namespace vtkh

#endif //VTK_H_MUTEX_HPP
