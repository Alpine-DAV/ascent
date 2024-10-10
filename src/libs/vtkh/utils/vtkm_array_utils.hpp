#ifndef VTKH_VTKM_ARRAY_UTILS_HPP
#define VTKH_VTKM_ARRAY_UTILS_HPP

#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/worklet/WorkletMapField.h>

namespace vtkh {

template<typename T>
T *
GetVTKMPointer(vtkm::cont::ArrayHandle<T> &handle)
{
  return handle.WritePortal().GetArray();
}

class VTKmTypeCast : public vtkm::worklet::WorkletMapField
{
public:
    VTKM_CONT
    VTKmTypeCast() = default;

    using ControlSignature = void(FieldIn, FieldOut);
    using ExecutionSignature = void( _1, _2);
    //using ExecutionSignature = void(InputIndex, _1, _2);

    //void operator()(const vtkm::Id idx, const vtkm::cont::ArrayHandle<InType> &input, vtkm::cont::ArrayHandle<OutType> &output) const
    template<typename InType, typename OutType>
    VTKM_EXEC
    void operator()(const InType &input, OutType &output) const
    {
        //output.Set(idx, static_cast<OutType>(input[idx]));
        output = static_cast<OutType>(input);
    }
};

}//namespace vtkh
#endif
