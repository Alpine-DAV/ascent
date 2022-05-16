// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_ISOSURFACE_INTERSECTION_HPP
#define DRAY_ISOSURFACE_INTERSECTION_HPP

#include <dray/data_model/device_mesh.hpp>
#include <dray/data_model/field.hpp>
#include <dray/data_model/mesh.hpp>
#include <dray/newton_solver.hpp>
#include <dray/ray.hpp>
#include <dray/utils/ray_utils.hpp>
#include <dray/vec.hpp>

namespace dray
{


template <ElemType eshape, int32 mesh_P, int32 field_P>
DRAY_EXEC bool intersect_ray_isosurf_local( stats::Stats &stats,
                                            const Element<3, 3, eshape, mesh_P> &mesh_elem,
                                            const Element<3, 1, eshape, field_P> &field_elem,
                                            const Ray &ray,
                                            Float isoval,
                                            Vec<Float, 3> &ref_coords,
                                            Float &ray_dist,
                                            bool use_init_guess = false)
{
  using MElemT = Element<3, 3, eshape, mesh_P>;
  using FElemT = Element<3, 1, eshape, field_P>;

  // TODO would be nicer as a lambda.

  // Newton step to solve the in-element isosurface intersection problem.
  struct Stepper
  {
    DRAY_EXEC typename IterativeMethod::StepStatus operator() (Vec<Float, 3 + 1> &xt) const
    {
      Vec<Float, 3> &x = *(Vec<Float, 3> *)&xt[0];
      Float &rdist = *(Float *)&xt[3];

      // project back onto the element
      for(int i = 0; i < 3; ++i)
      {
        x[i] = fminf(Float(1.f), fmaxf(x[i], Float(0.f)));
      }

      // Space jacobian and spatial residual.
      Vec<Float, 3> delta_y;
      Vec<Vec<Float, 3>, 3> j_col;
      delta_y = m_transf.eval_d (x, j_col);
      delta_y = m_ray_orig + m_ray_dir * rdist - delta_y;

      // Field gradient and field residual.
      Vec<Float, 1> _delta_f;
      Float &delta_f = _delta_f[0];
      Vec<Vec<Float, 1>, 3> _grad_f;
      Vec<Float, 3> &grad_f = *(Vec<Float, 3> *)&_grad_f[0];
      _delta_f = m_field.eval_d (x, _grad_f);
      delta_f = m_isovalue - delta_f;

      // Inverse of Jacobian (LU decomposition).
      bool inverse_valid;
      Matrix<Float, 3, 3> jacobian;
      for (int32 rdim = 0; rdim < 3; rdim++)
        jacobian.set_col (rdim, j_col[rdim]);
      MatrixInverse<Float, 3> jac_inv (jacobian, inverse_valid);

      // Compute adjustments using first-order approximation.
      Vec<Float, 3> delta_x = jac_inv * delta_y;
      Vec<Float, 3> delta_x_r = jac_inv * m_ray_dir;

      Float delta_r = (delta_f - dot (grad_f, delta_x)) / dot (grad_f, delta_x_r);
      delta_x = delta_x + delta_x_r * delta_r;

      //if (!inverse_valid) return IterativeMethod::Abort;

      // Apply the step.
      x = x + delta_x;
      rdist = rdist + delta_r;
      return IterativeMethod::Continue;
    }

    MElemT m_transf;
    FElemT m_field;
    Vec<Float, 3> m_ray_orig;
    Vec<Float, 3> m_ray_dir;
    Float m_isovalue;
  } stepper{ mesh_elem, field_elem, ray.m_orig, ray.m_dir, isoval };

  Vec<Float, 4> vref_coords{ ref_coords[0], ref_coords[1], ref_coords[2], ray_dist };
  if (!use_init_guess)
    for (int32 d = 0; d < 3; d++) // Always use the ray_dist coordinate, for now.
      vref_coords[d] = 0.5;

  // TODO somewhere else in the program, figure out how to set the precision
  const Float tol_ref = 1e-4;
  const int32 max_steps = 10;

  // Find solution.
  bool converged = (IterativeMethod::solve (stats, stepper, vref_coords, max_steps,
                                            tol_ref) == IterativeMethod::Converged);

  ref_coords = { vref_coords[0], vref_coords[1], vref_coords[2] };
  ray_dist = vref_coords[3];

  return (converged && mesh_elem.is_inside (ref_coords) &&
          ray.m_near <= ray_dist && ray_dist < ray.m_far);
}

} // namespace dray
#endif
