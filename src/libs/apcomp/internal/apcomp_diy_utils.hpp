#ifndef APCOMP_DIY_UTILS_HPP
#define APCOMP_DIY_UTILS_HPP

#include <diy/decomposition.hpp>
#include <apcomp/bounds.hpp>

namespace apcomp
{

static
Bounds DIYToBounds(const apcompdiy::DiscreteBounds &bounds)
{
  Bounds res;

  res.m_min_x = bounds.min[0];
  res.m_min_y = bounds.min[1];

  res.m_max_x = bounds.max[0];
  res.m_max_y = bounds.max[1];
  return res;
}

static
apcompdiy::DiscreteBounds BoundsToDIY(const Bounds &bounds)
{
  apcompdiy::DiscreteBounds diy_bounds(2);

  diy_bounds.min[0] = bounds.m_min_x;
  diy_bounds.min[1] = bounds.m_min_y;

  diy_bounds.max[0] = bounds.m_max_x;
  diy_bounds.max[1] = bounds.m_max_y;
  return diy_bounds;
}

} //namespace apcomp

#endif
