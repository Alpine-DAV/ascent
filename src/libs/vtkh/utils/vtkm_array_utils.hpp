#ifndef VTKH_VTKM_ARRAY_UTILS_HPP
#define VTKH_VTKM_ARRAY_UTILS_HPP

#include <vtkm/cont/ArrayHandle.h>

namespace vtkh {

template<typename T>
T *
GetVTKMPointer(vtkm::cont::ArrayHandle<T> &handle)
{
  return handle.WritePortal().GetArray();
}

}//namespace vtkh
#endif
