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
  bool m_merge_points = true;
public:
  void tolerance(const viskores::Float64 tol);
  void merge_points(bool merge);
  viskores::cont::DataSet Run(viskores::cont::DataSet &input,
                          viskores::filter::FieldSelection map_fields);
};
}
#endif
