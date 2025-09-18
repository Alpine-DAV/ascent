#ifndef VTK_H_VISKORES_CELL_AVERAGE_HPP
#define VTK_H_VISKORES_CELL_AVERAGE_HPP

#include <viskores/cont/DataSet.h>
#include <viskores/filter/FieldSelection.h>

namespace vtkh
{

class viskoresCellAverage
{
public:
  viskores::cont::DataSet Run(viskores::cont::DataSet &input,
                     std::string field_name,
                     std::string output_field_name,
                     viskores::filter::FieldSelection map_fields);
};
}
#endif
