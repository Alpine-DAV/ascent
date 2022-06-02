#ifndef VTKH_DIY_UTILS_HPP
#define VTKH_DIY_UTILS_HPP

#include <diy/decomposition.hpp>
#include <vtkm/Bounds.h>

namespace vtkh
{

static
vtkm::Bounds DIYBoundsToVTKM(const vtkhdiy::DiscreteBounds &bounds)
{
  vtkm::Bounds vtkm_bounds;

  vtkm_bounds.X.Min = bounds.min[0];
  vtkm_bounds.Y.Min = bounds.min[1];
  vtkm_bounds.Z.Min = bounds.min[2];

  vtkm_bounds.X.Max = bounds.max[0];
  vtkm_bounds.Y.Max = bounds.max[1];
  vtkm_bounds.Z.Max = bounds.max[2];
  return vtkm_bounds;
}

static
vtkhdiy::DiscreteBounds VTKMBoundsToDIY(const vtkm::Bounds &bounds)
{
  vtkhdiy::DiscreteBounds diy_bounds;

  diy_bounds.min[0] = bounds.X.Min;
  diy_bounds.min[1] = bounds.Y.Min;

  diy_bounds.max[0] = bounds.X.Max;
  diy_bounds.max[1] = bounds.Y.Max;

  if(bounds.Z.IsNonEmpty())
  {
    diy_bounds.min[2] = bounds.Z.Min;
    diy_bounds.max[2] = bounds.Z.Max;
  }
  else
  {
    diy_bounds.min[2] = 0;
    diy_bounds.max[2] = 0;
  }
  return diy_bounds;
}

} //namespace vtkh

#endif
