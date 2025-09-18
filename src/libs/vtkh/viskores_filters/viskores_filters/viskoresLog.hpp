#ifndef VTK_H_VISKORES_LOG_HPP
#define VTK_H_VISKORES_LOG_HPP

#include <viskores/cont/DataSet.h>
#include <viskores/filter/FieldSelection.h>
#include <viskores/filter/field_transform/LogValues.h>

namespace vtkh
{

typedef viskores::filter::field_transform::LogValues viskoresLogFilter;

class viskoresLog
{
public:
  viskores::cont::DataSet Run(viskores::cont::DataSet &input,
		          const std::string in_field_name,
			  const std::string out_field_name,
			  viskores::cont::Field::Association in_assoc,
		  	  viskoresLogFilter::LogBase log_base,
                          const viskores::Float32 min_value);
};
}
#endif

