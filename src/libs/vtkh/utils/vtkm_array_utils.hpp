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

struct VTKmTypeCast : public vtkm::worklet::WorkletMapField
{
    using ControlSignature = void(FieldIn input, FieldOut output);
    using ExecutionSignature = void(_1, _2);

    // Use VTKM_EXEC for the operator() function to make it run on both host and device
    template<typename InType, typename OutType>
    VTKM_EXEC void operator()(const InType& input, OutType& output) const
    {
        // Cast input to the output type and assign it
        output = static_cast<OutType>(input);
    }
};

}//namespace vtkh
#endif
