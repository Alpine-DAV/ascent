#ifndef VTK_H_VTKM_COMPOSITEVECTOR_HPP
#define VTK_H_VTKM_COMPOSITEVECTOR_HPP

#include <vtkm/cont/DataSet.h>
#include <vtkm/filter/field_transform/CompositeVectors.h>

namespace vtkh
{

class vtkmCompositeVector
{
public:
  vtkm::cont::DataSet Run(vtkm::cont::DataSet &input,
                          std::vector<std::string> input_field_names,
                          std::string output_field_name);
};
}
#endif
