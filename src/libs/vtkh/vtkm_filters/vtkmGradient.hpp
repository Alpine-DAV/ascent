#ifndef VTK_H_VTKM_GRADIENT_HPP
#define VTK_H_VTKM_GRADIENT_HPP

#include <vtkm/cont/DataSet.h>
#include <vtkm/filter/FieldSelection.h>
#include "GradientParameters.hpp"

namespace vtkh
{

class vtkmGradient
{
public:
  vtkm::cont::DataSet Run(vtkm::cont::DataSet &input,
                          std::string field_name,
                          GradientParameters params,
                          vtkm::filter::FieldSelection map_fields);
};
}
#endif
