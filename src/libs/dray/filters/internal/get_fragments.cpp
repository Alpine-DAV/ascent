#include <dray/filters/internal/get_fragments.hpp>
#include <dray/GridFunction/device_mesh.hpp>
#include <dray/GridFunction/device_field.hpp>
#include <dray/policies.hpp>
#include <dray/error_check.hpp>

namespace dray
{
namespace internal
{

  // --------------------------------------------------------------------
  // MatInvHack: Don't call inverse() if we can't, but get past compiler.
  // --------------------------------------------------------------------
  template <typename T, int32 M, int32 N>
  struct MatInvHack
  {
    DRAY_EXEC static Matrix<T,N,M>
    get_inverse(const Matrix<T,M,N> &m, bool &valid)
    {
      Matrix<T,N,M> a;
      a.identity();
      valid = false;
      return a;
    }
  };
  // ----------------------------------------
  template <typename T, int32 S>
  struct MatInvHack<T, S, S>
  {
    DRAY_EXEC static Matrix<T,S,S>
    get_inverse(const Matrix<T,S,S> &m, bool &valid)
    {
      return matrix_inverse(m, valid);
    }
  };
  // ----------------------------------------

template <class MeshElem, class FieldElem>
Array<Fragment>
get_fragments(Array<Ray> &rays,
              Range scalar_range,
              Field<FieldElem> &field,
              Mesh<MeshElem> &mesh,
              Array<RayHit> &hits)
{
  // Ray (read)    RefPoint (read)      ShadingContext (write)
  // ---           -----             --------------
  // m_pixel_id    m_el_id           m_pixel_id
  // m_dir         m_el_coords       m_ray_dir
  // m_orig
  // m_dist                          m_hit_pt
  //                                 m_is_valid
  //                                 m_sample_val
  //                                 m_normal
  //                                 m_gradient_mag

  // Convention: If dim==2, use surface normal as direction.
  //             If dim==3, use field gradient as direction.
  //             In any case, make sure it faces the camera.

  constexpr int32 dim = MeshElem::get_dim();

  const int32 size_rays = rays.size();
  //const int32 size_active_rays = rays.m_active_rays.size();

  Array<Fragment> fragments;
  fragments.resize(size_rays);
  Fragment *fragments_ptr = fragments.get_device_ptr();

  // Initialize other outputs to well-defined dummy values.
  constexpr Vec<Float,3> one_two_three = {123., 123., 123.};

  const int32 size = rays.size();

  const Float field_min = scalar_range.min();
  const Float field_range_rcp = rcp_safe( scalar_range.length() );

  const Ray *ray_ptr = rays.get_device_ptr_const();
  const RayHit *hit_ptr = hits.get_device_ptr_const();

  DeviceMesh<MeshElem> device_mesh(mesh);
  DeviceField<FieldElem> device_field(field);

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    Fragment frag;
    frag.m_scalar = -1.f;
    frag.m_normal = {-1.f, -1.f, -1.f};

    const Ray &ray = ray_ptr[i];
    const RayHit &hit = hit_ptr[i];

    if(hit.m_hit_idx > -1)
    {
      const int32 el_id = hit.m_hit_idx;
      Vec<Float, dim> ref_pt;
      ref_pt[0] = hit.m_ref_pt[0];
      ref_pt[1] = hit.m_ref_pt[1];
      if(dim == 3)
      {
        ref_pt[2] = hit.m_ref_pt[2];
      }
      // Evaluate element transformation and scalar field.
      Vec<Vec<Float, 3>, dim> jac_vec;
      Vec<Float, 3> world_pos = device_mesh.get_elem(el_id).eval_d(ref_pt, jac_vec);

      Vec<Float, 1> field_val;
      Vec<Vec<Float, 1>, dim> field_deriv;  // Only init'd if dim==3.

      if (dim == 2)
        frag.m_scalar = device_field.get_elem(el_id).eval_d(ref_pt, field_deriv)[0];
      else if (dim == 3)
        frag.m_scalar = device_field.get_elem(el_id).eval_d(ref_pt, field_deriv)[0];

      // What we output as the normal depends if dim==2 or 3.
      if (dim == 2)
      {
        // Use the normalized cross product of the jacobian
        frag.m_normal = cross(jac_vec[0], jac_vec[1]);
      }
      else if (dim == 3)
      {
        // Use the gradient of the scalar field relative to world axes.
        Matrix<Float, 3, dim> jacobian_matrix;
        Matrix<Float, 1, dim> gradient_ref;
        for (int32 rdim = 0; rdim < 3; rdim++)
        {
          jacobian_matrix.set_col(rdim, jac_vec[rdim]);
          gradient_ref.set_col(rdim, field_deriv[rdim]);
        }

        // To convert to world coords, use g = gh * J_inv.
        bool inv_valid;
        const Matrix<Float, dim, 3> j_inv =
            MatInvHack<Float, 3, dim>::get_inverse(jacobian_matrix, inv_valid);
        //TODO How to handle the case that inv_valid == false?
        const Matrix<Float, 1, 3> gradient_mat = gradient_ref * j_inv;
        Vec<Float,3> gradient_world = gradient_mat.get_row(0);

        // Output.
        frag.m_normal = gradient_world;
        //TODO What if the gradient is (0,0,0)? (Matt: it will be bad)
      }
    }

    fragments_ptr[i] = frag;

  });
  DRAY_ERROR_CHECK();

  return fragments;
}

// <2D>
template
Array<Fragment>
get_fragments<>(Array<Ray> &rays,
                      Range scalar_range,
                      Field<Element<2u, 1u, ElemType::Tensor, Order::General>> &field,
                      Mesh<MeshElem<2u, ElemType::Tensor, Order::General>> &mesh,
                      Array<RayHit> &hits);
template
Array<Fragment>
get_fragments<>(Array<Ray> &rays,
                      Range scalar_range,
                      Field<Element<3u, 1u, ElemType::Tensor, Order::General>> &field,
                      Mesh<MeshElem<3u, ElemType::Tensor, Order::General>> &mesh,
                      Array<RayHit> &hits);

} // namespace internal
} // namespace dray
