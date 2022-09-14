#ifndef VTK_H_VTKM_LOG_HPP
#define VTK_H_VTKM_LOG_HPP

#include <vtkm/cont/DataSet.h>
#include <vtkm/filter/FieldSelection.h>
#include <vtkm/filter/field_transform/LogValues.h>

namespace vtkh
{

typedef vtkm::filter::field_transform::LogValues vtkmLogFilter;

class vtkmLog
{
public:
  vtkm::cont::DataSet Run(vtkm::cont::DataSet &input,
		          const std::string in_field_name,
			  const std::string out_field_name,
			  vtkm::cont::Field::Association in_assoc,
		  	  vtkmLogFilter::LogBase log_base,
                          const vtkm::Float32 min_value);
};
}
#endif

