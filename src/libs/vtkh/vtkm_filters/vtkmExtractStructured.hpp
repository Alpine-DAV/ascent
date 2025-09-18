#ifndef VTK_H_VISKORES_EXTRACT_STRUCTURED_HPP
#define VTK_H_VISKORES_EXTRACT_STRUCTURED_HPP

#include <viskores/RangeId3.h>
#include <viskores/cont/DataSet.h>
#include <viskores/filter/FieldSelection.h>

namespace vtkh
{

class viskoresExtractStructured
{
public:
  viskores::cont::DataSet Run(viskores::cont::DataSet &input,
                           viskores::RangeId3 range,
                           viskores::Id3 sample_rate,
                           viskores::filter::FieldSelection map_fields);
};
}
#endif
