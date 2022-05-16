// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_VOLUME_SHADER_HPP
#define DRAY_VOLUME_SHADER_HPP

#include <dray/device_color_map.hpp>
#include <dray/data_model/device_mesh.hpp>
#include <dray/data_model/device_field.hpp>

namespace dray
{

template<typename Element, typename FieldElement>
struct VolumeShader
{
  DeviceMesh<Element> m_mesh;
  DeviceField<FieldElement> m_field;
  DeviceColorMap m_color_map;
  const PointLight *m_lights;
  const int32 m_num_lights;

  VolumeShader() = delete;

  VolumeShader(UnstructuredMesh<Element> &mesh,
               UnstructuredField<FieldElement> &field,
               ColorMap &color_map,
               Array<PointLight> lights)
    : m_mesh(mesh),
      m_field(field),
      m_color_map(color_map),
      m_lights(lights.get_device_ptr_const()),
      m_num_lights(lights.size())
  {
  }

  DRAY_EXEC
  void scalar_gradient(const Location &loc,
                       Float &scalar,
                       Vec<Float,3> &gradient,
                       Vec<Float,3> &world_pos) const
  {

    // i think we need this to oreient the deriv
    Vec<Vec<Float, 3>, 3> jac_vec;
    world_pos = m_mesh.get_elem(loc.m_cell_id).eval_d(loc.m_ref_pt, jac_vec);

    Vec<Vec<Float, 1>, 3> field_deriv;
    scalar = m_field.get_elem(loc.m_cell_id).eval_d(loc.m_ref_pt, field_deriv)[0];

    Matrix<Float, 3, 3> jacobian_matrix;
    Matrix<Float, 1, 3> gradient_ref;
    for(int32 rdim = 0; rdim < 3; ++rdim)
    {
      jacobian_matrix.set_col(rdim, jac_vec[rdim]);
      gradient_ref.set_col(rdim, field_deriv[rdim]);
    }

    bool inv_valid;
    const Matrix<Float, 3, 3> j_inv = matrix_inverse(jacobian_matrix, inv_valid);
    //TODO How to handle the case that inv_valid == false?
    const Matrix<Float, 1, 3> gradient_mat = gradient_ref * j_inv;
    gradient = gradient_mat.get_row(0);
  }

  DRAY_EXEC
  Vec<float32,4> shaded_color(const Location &loc, const Ray &ray) const
  {

    Vec<Float,3> gradient;
    Vec<Float,3> world_pos;
    Float scalar;
    scalar_gradient(loc, scalar, gradient, world_pos);
    Vec4f sample_color = m_color_map.color(scalar);
    Vec4f acc = {0.f, 0.f, 0.f, 0.f};
    if(sample_color[3] > 0.01)
    {

      gradient.normalize();

      Vec<float32,3> fgradient;
      fgradient[0] = float32(gradient[0]);
      fgradient[1] = float32(gradient[1]);
      fgradient[2] = float32(gradient[2]);

      const Vec<float32, 3> view_dir = { float32(-ray.m_dir[0]),
                                         float32(-ray.m_dir[1]),
                                         float32(-ray.m_dir[2])};

      for(int32 l = 0; l < m_num_lights; ++l)
      {
        const PointLight light = m_lights[l];

        Vec<float32, 3> light_dir = light.m_pos - world_pos;
        light_dir.normalize ();

        // now it might seem silly to calculate the dot twice, but
        // cuda 11.0.2 and gcc 7.3 are somehow optimizing away
        // code such that we are getting negative values. Even
        // manualy tring to ensure the value is positive fails.
        // Flipping the gradient so that the result is positive
        // seems to do the trick. This cost a lot of time to
        // track down.

        float32 ldir_dot_grad = dot(light_dir, fgradient);
        if(ldir_dot_grad < 0.f)
        {
          fgradient = -fgradient;
        }
        const float32 diffuse = clamp (dot(light_dir, fgradient), 0.f, 1.f);
        //const float32 diffuse = clamp (fabsf(dot(light_dir, fgradient)), 0.f, 1.f);

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

        Vec<float32, 3> half_vec = 0.5f * (view_dir + light_dir);
        half_vec.normalize ();
        // doing this for the same reason as above
        float32 h_dot_g = dot(fgradient, half_vec);
        if(h_dot_g < 0.f)
        {
          fgradient = -fgradient;
        }
        float32 doth = clamp (dot (fgradient, half_vec), 0.f, 1.f);
        // this is the old line that works on the cpu
        //float32 doth = clamp (fabsf(dot (fgradient, half_vec)), 0.f, 1.f);
        float32 intensity = pow (doth, light.m_spec_pow);

        //intensity *= sample_color[3];

        // add the specular component
        for (int32 c = 0; c < 3; ++c)
        {
          shaded_color[c] += intensity * light.m_spec[c] * sample_color[c];
        }


        acc += shaded_color;

        for (int32 c = 0; c < 3; ++c)
        {
          acc[c] = clamp (acc[c], 0.0f, 1.0f);
        }
      }
    }
    return acc;
  }

  DRAY_EXEC
  Vec<float32,4> color(const Location &loc) const
  {
    Vec<Vec<Float, 1>, 3> field_deriv;
    Float scalar =
      m_field.get_elem(loc.m_cell_id).eval_d(loc.m_ref_pt, field_deriv)[0];
    return m_color_map.color(scalar);
  }

};

} // namespace dray
#endif
