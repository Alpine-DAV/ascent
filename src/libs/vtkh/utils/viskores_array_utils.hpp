#ifndef VTKH_VISKORES_ARRAY_UTILS_HPP
#define VTKH_VISKORES_ARRAY_UTILS_HPP

#include <viskores/cont/ArrayHandle.h>
#include <viskores/worklet/WorkletMapField.h>

namespace vtkh {

template<typename T>
T *
GetVISKORESPointer(viskores::cont::ArrayHandle<T> &handle)
{
  return handle.WritePortal().GetArray();
}

class ViskoresTypeCast : public viskores::worklet::WorkletMapField
{
public:
    VISKORES_CONT
    ViskoresTypeCast() = default;

    using ControlSignature = void(FieldIn, FieldOut);
    using ExecutionSignature = void( _1, _2);
    //using ExecutionSignature = void(InputIndex, _1, _2);

    //void operator()(const viskores::Id idx, const viskores::cont::ArrayHandle<InType> &input, viskores::cont::ArrayHandle<OutType> &output) const
    template<typename InType, typename OutType>
    VISKORES_EXEC
    void operator()(const InType &input, OutType &output) const
    {
        //output.Set(idx, static_cast<OutType>(input[idx]));
        output = static_cast<OutType>(input);
    }
};

}//namespace vtkh
#endif
