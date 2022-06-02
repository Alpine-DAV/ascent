// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#include <dray/rendering/traceable.hpp>
#include <dray/rendering/device_framebuffer.hpp>
#include <dray/dispatcher.hpp>
#include <dray/error_check.hpp>
#include <dray/device_color_map.hpp>

#include <dray/utils/data_logger.hpp>

#include <dray/data_model/device_mesh.hpp>
#include <dray/data_model/device_field.hpp>

namespace dray
{
namespace detail
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

// ------------------------------------------------------------------------
template <typename T, int32 S>
struct MatInvHack<T, S, S>
{
  DRAY_EXEC static Matrix<T,S,S>
  get_inverse(const Matrix<T,S,S> &m, bool &valid)
  {
    return matrix_inverse(m, valid);
  }
};

// ------------------------------------------------------------------------
template <class MeshElem, class FieldElem>
Array<Fragment>
get_fragments(UnstructuredMesh<MeshElem> &mesh,
              UnstructuredField<FieldElem> &field,
              Array<RayHit> &hits)
{
  // Convention: If dim==2, use surface normal as direction.
  //             If dim==3, use field gradient as direction.

  const int32 size = hits.size();

  Array<Fragment> fragments;
  fragments.resize(size);
  Fragment *fragments_ptr = fragments.get_device_ptr();

  const RayHit *hit_ptr = hits.get_device_ptr_const();
  const int32 elems = mesh.cells();

  DeviceMesh<MeshElem> device_mesh(mesh);
  DeviceField<FieldElem> device_field(field);

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    Fragment frag;
    frag.m_scalar = -1.f;
    frag.m_normal = {-1.f, -1.f, -1.f};

    // this has to be inside the lambda for gcc8.1 otherwise:
    // error: use of 'this' in a constant expression
    constexpr int32 dim = MeshElem::get_dim();

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
      (void)world_pos;

      Vec<Vec<Float, 1>, dim> field_deriv;  // Only init'd if dim==3.

      if (dim == 2)
      {
        frag.m_scalar = device_field.get_elem(el_id).eval_d(ref_pt, field_deriv)[0];
      }
      else if (dim == 3)
      {
        frag.m_scalar = device_field.get_elem(el_id).eval_d(ref_pt, field_deriv)[0];
      }

      // What we output as the normal depends if dim==2 or 3.
      if (dim == 2)
      {
        // Use the normalized cross product of the jacobian
        frag.m_normal = cross(jac_vec[0], jac_vec[1]);
        frag.m_normal.normalize();
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

struct FragmentFunctor
{
  Array<RayHit> *m_hits;
  Array<Fragment> m_fragments;
  FragmentFunctor(Array<RayHit> *hits)
    : m_hits(hits)
  {
  }

  template<typename MeshType, typename FieldType>
  void operator()(MeshType &mesh, FieldType &field)
  {
    m_fragments = detail::get_fragments(mesh, field, *m_hits);
  }
};

} // namespace detail

// ------------------------------------------------------------------------
Traceable::Traceable(Collection &collection)
  : m_collection(collection),
    m_active_domain(0)
{
}

// ------------------------------------------------------------------------
Traceable::~Traceable()
{
}

// ------------------------------------------------------------------------
void Traceable::input(Collection &collection)
{
  m_collection = collection;
  m_active_domain = 0;
}

// ------------------------------------------------------------------------
void Traceable::field(const std::string &field_name)
{
  m_field_name = field_name;
  m_field_range = m_collection.range(m_field_name);
}

// ------------------------------------------------------------------------
void Traceable::color_map(ColorMap &color_map)
{
  m_color_map = color_map;
}

// ------------------------------------------------------------------------
ColorMap& Traceable::color_map()
{
  if(!m_color_map.range_set())
  {
    m_color_map.scalar_range(m_field_range);
  }
  return m_color_map;
}

// ------------------------------------------------------------------------
std::string Traceable::field() const
{
  return m_field_name;
}

// ------------------------------------------------------------------------
Collection& Traceable::collection()
{
  return m_collection;
}

// ------------------------------------------------------------------------
Array<Fragment>
Traceable::fragments(Array<RayHit> &hits)
{
  DRAY_LOG_OPEN("fragments");
  if(m_field_name == "")
  {
    DRAY_ERROR("Field name never set");
  }

  DataSet data_set = m_collection.domain(m_active_domain);

  Mesh *mesh = data_set.mesh();
  Field *field = data_set.field(m_field_name);

  detail::FragmentFunctor func(&hits);
  dispatch(mesh, field, func);
  DRAY_LOG_CLOSE();
  return func.m_fragments;
}

void Traceable::shade(const Array<Ray> &rays,
                      const Array<RayHit> &hits,
                      const Array<Fragment> &fragments,
                      const Array<PointLight> &lights,
                      Framebuffer &framebuffer)
{
  DataSet data_set = m_collection.domain(m_active_domain);
  if(!m_color_map.range_set())
  {
    m_color_map.scalar_range(m_field_range);
  }

  DRAY_LOG_OPEN("fragments");
  const RayHit *hit_ptr = hits.get_device_ptr_const ();
  const Ray *ray_ptr = rays.get_device_ptr_const ();
  const Fragment *frag_ptr = fragments.get_device_ptr_const ();
  const PointLight *light_ptr = lights.get_device_ptr_const();
  const int32 num_lights = lights.size();

  DeviceFramebuffer d_framebuffer(framebuffer);
  DeviceColorMap d_color_map (m_color_map);

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, hits.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
    const RayHit &hit = hit_ptr[ii];
    const Fragment &frag = frag_ptr[ii];
    const Ray &ray = ray_ptr[ii];

    if (hit.m_hit_idx > -1)
    {
      const int32 pid = ray.m_pixel_id;
      const Float sample_val = frag.m_scalar;
      Vec4f sample_color = d_color_map.color (sample_val);

      Vec<float32, 3> look_dir;
      look_dir[0] = float32(ray.m_dir[0]);
      look_dir[1] = float32(ray.m_dir[1]);
      look_dir[2] = float32(ray.m_dir[2]);

      Vec<float32, 3> view_pos;;
      view_pos[0] = float32(ray.m_orig[0]);
      view_pos[1] = float32(ray.m_orig[1]);
      view_pos[2] = float32(ray.m_orig[2]);

      Vec<float32, 3> fnormal = frag.m_normal;
      fnormal.normalize ();

      const Vec<float32, 3> normal = dot (look_dir, fnormal) >= 0 ? -fnormal : fnormal;
      const Vec<float32, 3> hit_pt = view_pos + look_dir * hit.m_dist;
      const Vec<float32, 3> view_dir = -look_dir;

      Vec4f acc = {0.f, 0.f, 0.f, 0.f};
      for(int l = 0; l < num_lights; ++l)
      {
        const PointLight light = light_ptr[l];

        Vec<float32, 3> light_dir = light.m_pos - hit_pt;
        light_dir.normalize ();
        const float32 diffuse = clamp (dot (light_dir, normal), 0.f, 1.f);

        Vec4f shaded_color;
        shaded_color[0] = light.m_amb[0] * sample_color[0];
        shaded_color[1] = light.m_amb[1] * sample_color[1];
        shaded_color[2] = light.m_amb[2] * sample_color[2];
        shaded_color[3] = sample_color[3];

        // add the diffuse component
        for (int32 c = 0; c < 3; ++c)
        {
          shaded_color[c] += diffuse * light.m_diff[c] * sample_color[c];
        }

        Vec<float32, 3> half_vec = view_dir + light_dir;
        half_vec.normalize ();
        float32 doth = clamp (dot (normal, half_vec), 0.f, 1.f);
        float32 intensity = pow (doth, light.m_spec_pow);

        // add the specular component
        for (int32 c = 0; c < 3; ++c)
        {
          //shaded_color[c] += intensity * light_color[c] * sample_color[c];
          shaded_color[c] += intensity * light.m_spec[c] * sample_color[c];
        }

        acc += shaded_color;
      }

      for (int32 c = 0; c < 3; ++c)
      {
        acc[c] = clamp (acc[c], 0.0f, 1.0f);
      }

      d_framebuffer.m_colors[pid] = acc;
      d_framebuffer.m_depths[pid] = hit.m_dist;
    }

  });
  DRAY_ERROR_CHECK();
  DRAY_LOG_CLOSE();
}

void Traceable::shade(const Array<Ray> &rays,
                      const Array<RayHit> &hits,
                      const Array<Fragment> &fragments,
                      Framebuffer &framebuffer)
{
  DataSet data_set = m_collection.domain(m_active_domain);

  if(!m_color_map.range_set())
  {
    std::vector<Range> ranges = data_set.field(m_field_name)->range();
    if(ranges.size() != 1)
    {
      DRAY_ERROR("Expected 1 range component, got "<<ranges.size());
    }
    m_color_map.scalar_range(ranges[0]);
  }

  DRAY_LOG_OPEN("fragments");
  const RayHit *hit_ptr = hits.get_device_ptr_const ();
  const Ray *ray_ptr = rays.get_device_ptr_const ();
  const Fragment *frag_ptr = fragments.get_device_ptr_const ();

  DeviceFramebuffer d_framebuffer(framebuffer);
  DeviceColorMap d_color_map (m_color_map);

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, hits.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
    const RayHit &hit = hit_ptr[ii];
    const Fragment &frag = frag_ptr[ii];
    const Ray &ray = ray_ptr[ii];

    if (hit.m_hit_idx > -1)
    {
      const int32 pid = ray.m_pixel_id;
      const Float sample_val = frag.m_scalar;
      Vec4f sample_color = d_color_map.color (sample_val);

      d_framebuffer.m_colors[pid] = sample_color;
      d_framebuffer.m_depths[pid] = hit.m_dist;
    }

  });
  DRAY_ERROR_CHECK();
  DRAY_LOG_CLOSE();
}

void Traceable::colors(const Array<Ray> &rays,
                       const Array<RayHit> &hits,
                       const Array<Fragment> &fragments,
                       Array<Vec<float32,4>> &colors)
{
  DataSet data_set = m_collection.domain(m_active_domain);

  if(!m_color_map.range_set())
  {
    std::vector<Range> ranges = data_set.field(m_field_name)->range();
    if(ranges.size() != 1)
    {
      DRAY_ERROR("Expected 1 range component, got "<<ranges.size());
    }
    m_color_map.scalar_range(ranges[0]);
  }

  DRAY_LOG_OPEN("colors");
  const RayHit *hit_ptr = hits.get_device_ptr_const ();
  const Ray *ray_ptr = rays.get_device_ptr_const ();
  const Fragment *frag_ptr = fragments.get_device_ptr_const ();

  colors.resize(rays.size());
  Vec<float32,4> *color_ptr = colors.get_device_ptr();
  DeviceColorMap d_color_map (m_color_map);

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, hits.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
    const RayHit &hit = hit_ptr[ii];
    const Fragment &frag = frag_ptr[ii];
    const Ray &ray = ray_ptr[ii];

    Vec4f sample_color = {{0.f,0.f,0.f,0.f}};
    if (hit.m_hit_idx > -1)
    {
      const Float sample_val = frag.m_scalar;
      sample_color = d_color_map.color (sample_val);
    }

    color_ptr[ii] = sample_color;
  });
  DRAY_ERROR_CHECK();
  DRAY_LOG_CLOSE();
}

int32
Traceable::num_domains()
{
  return m_collection.local_size();
}

void
Traceable::active_domain(int32 domain_index)
{
  m_active_domain = domain_index;
}

int32
Traceable::active_domain()
{
  return m_active_domain;
}

} // namespace dray
