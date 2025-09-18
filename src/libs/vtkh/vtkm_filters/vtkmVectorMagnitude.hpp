#ifndef VTK_H_VISKORES_VECTOR_MAGNITUDE_HPP
#define VTK_H_VISKORES_VECTOR_MAGNITUDE_HPP

#include <viskores/cont/DataSet.h>
#include <viskores/filter/FieldSelection.h>

namespace vtkh
{

class viskoresVectorMagnitude
{
public:
  viskores::cont::DataSet Run(viskores::cont::DataSet &input,
                          std::string field_name,
                          std::string out_field_name,
                          viskores::filter::FieldSelection map_fields);
};
}
#endif
