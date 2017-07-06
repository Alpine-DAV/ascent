#ifndef VTKH_DIY_UTILS_HPP
#define VTKH_DIY_UTILS_HPP

#include <diy/decomposition.hpp>
#include <vtkm/Bounds.h>

namespace vtkh 
{
  
static  
vtkm::Bounds DIYBoundsToVTKM(const diy::DiscreteBounds &bounds)
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
diy::DiscreteBounds VTKMBoundsToDIY(const vtkm::Bounds &bounds)
{
  diy::DiscreteBounds diy_bounds;

  diy_bounds.min[0] = bounds.X.Min;
  diy_bounds.min[1] = bounds.Y.Min;
  diy_bounds.min[2] = bounds.Z.Min;
                                 
  diy_bounds.max[0] = bounds.X.Max;
  diy_bounds.max[1] = bounds.Y.Max;
  diy_bounds.max[2] = bounds.Z.Max;
  return diy_bounds;
}

} //namespace vtkh

#endif
