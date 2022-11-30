#ifndef APCOMP_SCALAR_IMAGE_COMPOSITOR_HPP
#define APCOMP_SCALAR_IMAGE_COMPOSITOR_HPP

#include <apcomp/apcomp_config.h>
#include <apcomp/scalar_image.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>

#include<apcomp/apcomp_exports.h>

namespace apcomp
{

class APCOMP_API ScalarImageCompositor
{
public:

void ZBufferComposite(apcomp::ScalarImage &front, const apcomp::ScalarImage &image)
{
  if(front.m_payload_bytes != image.m_payload_bytes)
  {
    std::cout<<"very bad\n";
  }
  assert(front.m_depths.size() == front.m_payloads.size() / front.m_payload_bytes);
  assert(front.m_bounds.m_min_x == image.m_bounds.m_min_x);
  assert(front.m_bounds.m_min_y == image.m_bounds.m_min_y);
  assert(front.m_bounds.m_max_x == image.m_bounds.m_max_x);
  assert(front.m_bounds.m_max_y == image.m_bounds.m_max_y);

  const int size = static_cast<int>(front.m_depths.size());
  const bool nan_check = image.m_default_value != image.m_default_value;
#ifdef APCOMP_OPENMP_ENABLED
  #pragma omp parallel for
#endif
  for(int i = 0; i < size; ++i)
  {
    const float depth = image.m_depths[i];
    const float fdepth = front.m_depths[i];
    // this should handle NaNs correctly
    const bool take_back = fmin(depth, fdepth) == depth;

    if(take_back)
    {
      const int offset = i * 4;
      front.m_depths[i] = depth;
      const size_t p_offset = i * front.m_payload_bytes;
      std::copy(&image.m_payloads[p_offset],
                &image.m_payloads[p_offset] + front.m_payload_bytes,
                &front.m_payloads[p_offset]);
    }
  }
}


};

} // namespace apcomp
#endif
