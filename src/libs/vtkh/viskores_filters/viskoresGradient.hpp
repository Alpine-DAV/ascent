#ifndef VTK_H_VISKORES_GRADIENT_HPP
#define VTK_H_VISKORES_GRADIENT_HPP

#include <viskores/cont/DataSet.h>
#include <viskores/filter/FieldSelection.h>
#include "GradientParameters.hpp"

namespace vtkh
{

class viskoresGradient
{
public:
  viskores::cont::DataSet Run(viskores::cont::DataSet &input,
                          std::string field_name,
                          GradientParameters params,
                          viskores::filter::FieldSelection map_fields);
};
}
#endif
