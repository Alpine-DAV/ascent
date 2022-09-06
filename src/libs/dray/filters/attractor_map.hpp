#ifndef DRAY_ATTRACTOR_MAP_HPP
#define DRAY_ATTRACTOR_MAP_HPP

#include <dray/GridFunction/mesh.hpp>
#include <dray/data_model/data_set.hpp>
#include <dray/types.hpp>
#include <dray/ref_point.hpp>
#include <dray/vec.hpp>
#include <dray/array.hpp>


namespace dray
{

  // This filter takes a mesh, element id, and a world-space query point.
  // It produces an image where each pixel location represents an initial guess, and the
  // pixel color represents whether the solver converged to an element-interior solution or to a
  // some point outside the element.
class AttractorMap
{
public:
  template<class ElemT>
  Array<Vec<float32,4>> execute(
      bool output_color_buffer,
      const Vec<Float,3> world_query_point,
      const Array<RefPoint<3>> &guesses,
      Array<Vec<Float,3>> &solutions,
      Array<int32> &iterations,
      DataSet<ElemT> &data_set);

  // grid_depth becomes the exponent leading to grid size being a power of 2.
  // Makes it easier to find 3-tuple-valued indices from a linearized index.
  static Array<RefPoint<3>>
  domain_grid_3d(uint32 grid_depth_x,
                 uint32 grid_depth_y,
                 uint32 grid_depth_z,
                 int32 el_id = 0);

  static Array<RefPoint<3>>
  domain_grid_slice_xy(uint32 grid_depth_x,
                       uint32 grid_depth_y,
                       Float ref_z_val = 0.5,
                       int32 el_id = 0);

  // Could also do other slices, but then have to decide whether x comes before z or after.
};


class AttractorMapShader
{
  protected:
    Vec4f u_inner_color;
    Vec4f u_outer_color;
    Vec4f u_edge_color;
    float32 u_inner_edge_radius_rcp;
    float32 u_outer_edge_radius_rcp;

  public:
    void set_uniforms(Vec4f inner_color,
                      Vec4f outer_color,
                      Vec4f edge_color,
                      float32 inner_edge_radius,
                      float32 outer_edge_radius)
    {
      u_inner_color = inner_color;
      u_outer_color = outer_color;
      u_edge_color = edge_color;

      u_inner_edge_radius_rcp = (inner_edge_radius > 0.0 ? 1.0 / inner_edge_radius : 0.05);
      u_outer_edge_radius_rcp = (outer_edge_radius > 0.0 ? 1.0 / outer_edge_radius : 3.0);
    }

    DRAY_EXEC static Float edge_dist(const Vec<Float,3> &rcoords)
    {
      // 0 on element boundary, +0.5 in center, negative outside the element.
      Float dist0 = 0.5 - fabs(rcoords[0] - 0.5);
      Float dist1 = 0.5 - fabs(rcoords[1] - 0.5);
      Float dist2 = 0.5 - fabs(rcoords[2] - 0.5);
      return min(min(dist0, dist1), dist2);
    }

    DRAY_EXEC Vec4f operator()(const Vec<Float,3> &rcoords) const
    {
      // For now, piecewise linear interpolation on the distance to nearest face.
      // TODO output a single channel for vtk image.
      Float edge_dist = AttractorMapShader::edge_dist(rcoords);

      Vec4f color0 = u_edge_color;
      Vec4f color1 = (edge_dist >= 0.0 ? u_inner_color : u_outer_color);
      edge_dist = (edge_dist >= 0.0 ? edge_dist * u_inner_edge_radius_rcp :
                                      -edge_dist * u_outer_edge_radius_rcp);

      edge_dist = min(edge_dist, Float(1.0f));

      return color0 * (1 - edge_dist) + color1 * edge_dist;
    }
};




}//namespace dray


#endif//DRAY_ATTRACTOR_MAP_HPP
