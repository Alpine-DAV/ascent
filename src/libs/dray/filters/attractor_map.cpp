#include <dray/filters/attractor_map.hpp>

#include <dray/GridFunction/mesh.hpp>
#include <dray/GridFunction/device_mesh.hpp>
#include <dray/data_model/data_set.hpp>

#include <dray/array.hpp>
#include <dray/array_utils.hpp>
#include <dray/error_check.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>
#include <dray/math.hpp>

namespace dray
{


  //
  // AttractorMap::execute()
  //
  template<class ElemT>
  Array<Vec<float32,4>> AttractorMap::execute( bool output_color_buffer,
                                               const Vec<Float,3> world_query_point,
                                               const Array<RefPoint<3>> &guesses,
                                               Array<Vec<Float,3>> &solutions,
                                               Array<int32> &iterations,
                                               DataSet<ElemT> &data_set)
  {
    using Color = Vec<float32, 4>;

    // Resize outputs.
    solutions.resize(guesses.size());
    iterations.resize(guesses.size());

    Array<Color> color_buffer;
    if (output_color_buffer)
    {
      color_buffer.resize(guesses.size());

      // Initialize the color buffer to (0,0,0,0).
      const Color init_color = make_vec4f(0.f, 0.f, 0.f, 0.f);
      array_memset_vec(color_buffer, init_color);
    }

    // Get mesh.
    const Mesh<ElemT> &mesh = data_set.get_mesh();
    DeviceMesh<ElemT> device_mesh(mesh);

    // Set shader uniforms (for color buffer output).
    AttractorMapShader shader;
    shader.set_uniforms({0,0,1,1}, {1,0,0,1}, {1,1,1,1}, 0.05, 3.0);

    // Read-only pointers.
    const RefPoint<3> *guess_ptr = guesses.get_device_ptr_const();

    // Writable pointers.
    Vec<Float,3> *solutions_ptr = solutions.get_device_ptr();
    int32 *iterations_ptr = iterations.get_device_ptr();
    Color *color_buffer_ptr;
    if (output_color_buffer)
      color_buffer_ptr = color_buffer.get_device_ptr();

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, guesses.size()), [=] DRAY_LAMBDA(const int32 sample_idx)
    {
      // Use ref_point.m_el_coords as initial guess, then receive into m_el_coords the solution.
      RefPoint<3> ref_point = guess_ptr[sample_idx];

      /// std::cout << "Before: " << ref_point.m_el_coords << "  ";
      stats::Stats stats;
      stats.construct();

      // TODO: we currently don't really get any stats other than number of
      // iterations, so we could just make this an int
      device_mesh.get_elem(ref_point.m_el_id)
          .eval_inverse_local(stats,
                              world_query_point,
                              ref_point.m_el_coords);

      /// std::cout << "After: " << ref_point.m_el_coords << "\n";

      // Store outputs.
      solutions_ptr[sample_idx] = ref_point.m_el_coords;
      iterations_ptr[sample_idx] = stats.iters() ;
    });
    DRAY_ERROR_CHECK();

    if (output_color_buffer)
      RAJA::forall<for_policy>(RAJA::RangeSegment(0, guesses.size()), [=] DRAY_LAMBDA (const int32 sample_idx)
      {
        color_buffer_ptr[sample_idx] = shader(solutions_ptr[sample_idx]);
      });
    DRAY_ERROR_CHECK();

    return color_buffer;
  }



  Array<RefPoint<3>>
  AttractorMap::domain_grid_3d(uint32 grid_depth_x,
                               uint32 grid_depth_y,
                               uint32 grid_depth_z,
                               int32 el_id)
  {
    const int32 grid_size_x = 1u << grid_depth_x;
    const int32 grid_size_y = 1u << grid_depth_y;
    const int32 grid_size_z = 1u << grid_depth_z;

    // The number of subintervals is the number of sample points - 1.
    const Float grid_divisor_x = (grid_size_x - 1);
    const Float grid_divisor_y = (grid_size_y - 1);
    const Float grid_divisor_z = (grid_size_z - 1);

    const int32 total_num_samples = grid_size_x * grid_size_y * grid_size_z;

    Array<RefPoint<3>> guess_grid;
    guess_grid.resize(total_num_samples);
    RefPoint<3> *guess_grid_ptr = guess_grid.get_device_ptr();

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, total_num_samples), [=] DRAY_LAMBDA (const int32 sample_idx)
    {
      // Index with x innermost and z outermost.
      const int32 xi = sample_idx & (grid_size_x - 1);
      const int32 yi = (sample_idx >> grid_depth_x) & (grid_size_y - 1);
      const int32 zi = (sample_idx >> grid_depth_x + grid_depth_y) /* & (grid_size_z - 1) */;

      guess_grid_ptr[sample_idx].m_el_id = el_id;
      guess_grid_ptr[sample_idx].m_el_coords = {((Float) xi)/grid_divisor_x,
                                                ((Float) yi)/grid_divisor_y,
                                                ((Float) zi)/grid_divisor_z};
    });
    DRAY_ERROR_CHECK();

    return guess_grid;
  }


  Array<RefPoint<3>>
  AttractorMap::domain_grid_slice_xy(uint32 grid_depth_x,
                                     uint32 grid_depth_y,
                                     Float ref_z_val,
                                     int32 el_id)
  {
    const int32 grid_size_x = 1u << grid_depth_x;
    const int32 grid_size_y = 1u << grid_depth_y;

    // The number of subintervals is the number of sample points - 1.
    const Float grid_divisor_x = (grid_size_x - 1);
    const Float grid_divisor_y = (grid_size_y - 1);

    const int32 total_num_samples = grid_size_x * grid_size_y;

    Array<RefPoint<3>> guess_grid;
    guess_grid.resize(total_num_samples);
    RefPoint<3> *guess_grid_ptr = guess_grid.get_device_ptr();

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, total_num_samples), [=] DRAY_LAMBDA (const int32 sample_idx)
    {
      // Index with x innermost and z outermost.
      const int32 xi = sample_idx & (grid_size_x - 1);
      const int32 yi = (sample_idx >> grid_depth_x) /* & (grid_size_y - 1) */;

      guess_grid_ptr[sample_idx].m_el_id = el_id;
      guess_grid_ptr[sample_idx].m_el_coords = {((Float) xi)/grid_divisor_x,
                                                ((Float) yi)/grid_divisor_y,
                                                ref_z_val};
    });
    DRAY_ERROR_CHECK();

    return guess_grid;
  }



  //
  // Template instantiations.
  //

template
Array<Vec<float32,4>>
AttractorMap::execute<MeshElem<3u, ElemType::Tensor, Order::General>>( bool output_color_buffer,
                                                      const Vec<Float,3> world_query_point,
                                                      const Array<RefPoint<3>> &guesses,
                                                      Array<Vec<Float,3>> &solutions,
                                                      Array<int32> &iterations,
                                                      DataSet<MeshElem<3u, ElemType::Tensor, Order::General>> &data_set);

}//namespace dray
