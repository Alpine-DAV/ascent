#ifndef DRAY_SURFACE_TRIANGLE_HPP
#define DRAY_SURFACE_TRIANGLE_HPP

/*
 * This filter class is just a placeholder to render Bezier triangles
 * before formally adding them to the Element system.
 */

#include <dray/array.hpp>
#include <dray/vec.hpp>
#include <dray/types.hpp>

namespace dray
{

class SurfaceTriangle
{
  public:
    // This filter rasterizes samples from the triangle(s).
    template<typename T>
    Array<Vec<float32,4>> execute(int32 im_width,
                                  int32 im_height,
                                  const Array<Vec<T,2>> &triangle_dofs,
                                  int32 poly_order,
                                  int32 num_samples = 1000);
};

}//namespace dray

#endif//DRAY_SURFACE_TRIANGLE_HPP
