#ifndef VTK_H_VISKORES_COMPOSITEVECTOR_HPP
#define VTK_H_VISKORES_COMPOSITEVECTOR_HPP

#include <viskores/cont/DataSet.h>
#include <viskores/filter/field_transform/CompositeVectors.h>

namespace vtkh
{

class viskoresCompositeVector
{
public:
  viskores::cont::DataSet Run(viskores::cont::DataSet &input,
                          std::vector<std::string> input_field_names,
                          std::string output_field_name,
			  viskores::cont::Field::Association assoc);
};
}
#endif
