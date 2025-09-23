#ifndef VTK_H_VISKORES_CLEAN_GRID_HPP
#define VTK_H_VISKORES_CLEAN_GRID_HPP

#include <viskores/cont/DataSet.h>
#include <viskores/filter/FieldSelection.h>

namespace vtkh
{

class viskoresCleanGrid
{
protected:
  viskores::Float64 m_tolerance = -1.;
public:
  void tolerance(const viskores::Float64 tol);

  viskores::cont::DataSet Run(viskores::cont::DataSet &input,
                          viskores::filter::FieldSelection map_fields);
};
}
#endif
