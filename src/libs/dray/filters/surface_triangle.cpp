
#include <dray/dray_config.h>
#include <dray/dray_exports.h>

#include <dray/filters/surface_triangle.hpp>
#include <dray/data_model/bezier_simplex.hpp>
#include <dray/array.hpp>
#include <dray/array_utils.hpp>
#include <dray/vec.hpp>
#include <dray/types.hpp>
#include <dray/error_check.hpp>

namespace dray
{


template<typename T>
Array<Vec<float32,4>> SurfaceTriangle::execute(int32 im_width, int32 im_height, const Array<Vec<T,2>> &triangle_dofs, int32 poly_order, int32 num_samples)
{
  const Vec<T,2> world_viewport_sz = {1.1, 1.1};
  const Vec<float32,4> out_color = {0.0f, 0.0f, 0.0f, 0.0f};
  const Vec<float32,4> in_color = {0.0f, 0.0f, 1.0f, 1.0f};

  int32 n = ceil(sqrt(2*num_samples + 0.25) - 0.5);    // Quadratic formula
  num_samples = n * (n+1) / 2;                         // Smallest upper-bound triangle number.

  // Create image buffer and initialize to out_color.
  Array<Vec<float32,4>> img_array;
  img_array.resize(im_width * im_height);
  array_memset_vec(img_array, out_color);

  const Vec<T,2> *triangle_dofs_ptr = triangle_dofs.get_device_ptr_const();
  Vec<float32,4> *img_ptr = img_array.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_samples), [=] DRAY_LAMBDA (int32 sample_idx)
  {
    // s == j*(n - (j+1)/2) + i
    // (n+1)*n/2 - (s+1) >= (n-j)*(n-j-1)/2
    //
    // Solving for (n-j) using this upper bound makes the discriminant nonnegative.
    const int32 jj = n - floor(sqrt((n+1)*n - 2*sample_idx - 1.75) + 0.5);
    const int32 ii = sample_idx - jj*n + jj*(jj-1)/2;

    // Define sample.
    Vec<T,2> ref_coords = {(T) ii / n, (T) jj / n};

    // Transform coordinates.
    /// Vec<T,2> world_coords = ref_coords;
    Vec<T,2> world_coords = BezierSimplex<T,2>::template eval<Vec<T,2>>(ref_coords, triangle_dofs_ptr, poly_order);

    if (!(-epsilon<T>() < world_coords[0] && world_coords[0] < world_viewport_sz[0] + epsilon<T>() &&
          -epsilon<T>() < world_coords[1] && world_coords[1] < world_viewport_sz[1] + epsilon<T>()))
      return;

    world_coords *= im_width;  // Image space.

    // Rasterize as 1px point.
    int32 im_ii = 0.5 + world_coords[0];
    int32 im_jj = 0.5 + world_coords[1];

    if (0 <= im_ii && im_ii < im_width &&
        0 <= im_jj && im_jj < im_height)
      img_ptr[im_jj * im_height + im_ii] = in_color;
  });
  DRAY_ERROR_CHECK();
  return img_array;
}


template
Array<Vec<float32,4>> SurfaceTriangle::execute<float32>(int32 im_width,
                              int32 im_height,
                              const Array<Vec<float32,2>> &triangle_dofs,
                              int32 poly_order,
                              int32 num_samples);

template
Array<Vec<float32,4>> SurfaceTriangle::execute<float64>(int32 im_width,
                              int32 im_height,
                              const Array<Vec<float64,2>> &triangle_dofs,
                              int32 poly_order,
                              int32 num_samples);


}//namespace dray
