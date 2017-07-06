#ifndef VTKH_VTKM_ARRAY_UTILS_HPP
#define VTKH_VTKM_ARRAY_UTILS_HPP

#include <vtkm/cont/serial/internal/ArrayManagerExecutionSerial.h>
#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/serial/internal/DeviceAdapterTagSerial.h>

namespace vtkh {

template<typename T>
T *
GetVTKMPointer(vtkm::cont::ArrayHandle<T> &handle)
{
  typedef typename vtkm::cont::ArrayHandle<T> HandleType;
  typedef typename HandleType::template ExecutionTypes<vtkm::cont::DeviceAdapterTagSerial>::Portal PortalType;
  typedef typename vtkm::cont::ArrayPortalToIterators<PortalType>::IteratorType IteratorType;
  IteratorType iter = vtkm::cont::ArrayPortalToIterators<PortalType>(handle.GetPortalControl()).GetBegin();
  return &(*iter);
}

}//namespace vtkh
#endif
