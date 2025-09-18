#ifndef VTK_H_VISKORES_TETRAHEDRALIZE_HPP
#define VTK_H_VISKORES_TETRAHEDRALIZE_HPP

#include <viskores/cont/DataSet.h>
#include <viskores/filter/FieldSelection.h>

namespace vtkh
{

class viskoresTetrahedralize
{
public:
  viskores::cont::DataSet Run(viskores::cont::DataSet &input,
                          viskores::filter::FieldSelection map_fields);
};
}
#endif
