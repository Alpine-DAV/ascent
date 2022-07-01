// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_FACE_INTERSECTION_HPP
#define DRAY_FACE_INTERSECTION_HPP

#include <dray/data_model/device_mesh.hpp>
#include <dray/data_model/field.hpp>
#include <dray/data_model/mesh.hpp>
#include <dray/newton_solver.hpp>
#include <dray/ray.hpp>
#include <dray/utils/ray_utils.hpp>
#include <dray/vec.hpp>

namespace dray
{

// TODO i don't think this should be a class

template <class ElemT> struct Intersector_RayFace
{
  DRAY_EXEC static bool intersect (stats::Stats &stats,
                                   const ElemT &surf_elem,
                                   const Ray &ray,
                                   const AABB<2> &guess_domain,
                                   Vec<Float, 2> &ref_coords,
                                   Float &ray_dist,
                                   bool use_init_guess = false)
  {
    using QueryT = Ray;
    using RefBoxT = AABB<2>;
    using SolT = Vec<Float, 3>; // Parametric coordinates + ray distance.

    // TODO?
    const Float tol_refbox = 1.0 / 64;
    constexpr int32 subdiv_budget = 0; // 0 means initial_guess = face_guess_domain.center();
    constexpr int32 stack_cap = 13;

    RefBoxT domain = (use_init_guess ? guess_domain : RefBoxT::ref_universe ());

    // For subdivision search, test whether the sub-element possibly contains
    // the query point. Strict test because the bounding boxes are approximate.
    struct FInBounds
    {
      DRAY_EXEC bool
      operator() (stats::Stats &stats, const QueryT &qray, const ElemT &elem, const RefBoxT &ref_box)
      {
        // Test intersection of bounding box with ray.
        dray::AABB<3> space_bounds;
        elem.get_sub_bounds (ref_box, space_bounds);
        return intersect_ray_aabb (qray, space_bounds);
      }
    };

    // Get solution when close enough: Iterate using Newton's method.
    struct FGetSolution
    {
      DRAY_EXEC bool operator() (stats::Stats &stats,
                                 const QueryT &qray,
                                 const ElemT &elem,
                                 const RefBoxT &ref_box,
                                 SolT &solution)
      {
        Vec<Float, 2> sol_ref_coords = ref_box.template center<Float> ();
        Float sol_ray_dist = qray.m_near;

        bool sol_found =
        Intersector_RayFace<ElemT>::intersect_local (stats, elem, qray, sol_ref_coords,
                                                     sol_ray_dist, true);
        solution[0] = sol_ref_coords[0]; // Pack
        solution[1] = sol_ref_coords[1];
        solution[2] = sol_ray_dist;

        return sol_found;
      }
    };

    // Initiate subdivision search.
    SolT solution;
    uint32 ret_code;
    int32 num_solutions =
    SubdivisionSearch::subdivision_search<QueryT, ElemT, RefBoxT, SolT, FInBounds, FGetSolution, subdiv_budget, stack_cap> (
    ret_code, stats, ray, surf_elem, tol_refbox, &domain, &solution, 1);

    ref_coords[0] = solution[0]; // Unpack
    ref_coords[1] = solution[1];
    ray_dist = solution[2];

    return num_solutions > 0;
  }


  // Returns true if an intersection was found.
  DRAY_EXEC static bool intersect_local (stats::Stats &stats,
                                         const ElemT &surf_elem,
                                         const Ray &ray,
                                         Vec<Float, 2> &ref_coords,
                                         Float &ray_dist,
                                         bool use_init_guess = false)
  {
    // TODO would be nicer as a lambda.

    // Newton step to solve the element face-ray intersection problem.
    struct Stepper
    {
      DRAY_EXEC typename IterativeMethod::StepStatus operator() (Vec<Float, 2 + 1> &xt) const
      {
        Vec<Float, 2> &x = *(Vec<Float, 2> *)&xt[0];
        Float &rdist = *(Float *)&xt[2];
        // project back onto the element
        for(int i = 0; i < 2; ++i)
        {
          x[i] = fminf(Float(1.f), fmaxf(x[i], Float(0.f)));
        }

        // Space jacobian and spatial residual.
        Vec<Float, 3> delta_y;
        Vec<Vec<Float, 3>, 2> j_col;
        delta_y = m_transf.eval_d (x, j_col);
        delta_y = m_ray_orig - delta_y;

        Matrix<Float, 3, 3> jacobian;
        jacobian.set_col (0, j_col[0]);
        jacobian.set_col (1, j_col[1]);
        jacobian.set_col (2, -m_ray_dir);

        // Inverse of Jacobian (LU decomposition) --> spatial adjustment and new ray dist.
        bool inverse_valid;
        Vec<Float, 3> delta_xt = MatrixInverse<Float, 3> (jacobian, inverse_valid) * delta_y;

        //  TODO: figure out what to do about tolerance issues
        //if (!inverse_valid)
        //{
        //  // std::cout<<"ABORT\n";
        //  return IterativeMethod::Abort;
        //}

        // Apply the step.
        x = x + (*(Vec<Float, 2> *)&delta_xt);
        rdist = delta_xt[2];
        // std::cout<<"x "<<x<<"\n";
        // std::cout<<"dxt "<<delta_xt<<"\n";
        return IterativeMethod::Continue;
      }

      ElemT m_transf;
      Vec<Float, 3> m_ray_orig;
      Vec<Float, 3> m_ray_dir;
    }

    stepper{ surf_elem, ray.m_orig, ray.m_dir };

    Vec<Float, 2 + 1> vref_coords{ ref_coords[0], ref_coords[1], ray_dist };
    if (!use_init_guess)
      for (int32 d = 0; d < 2; d++) // Always use the ray_dist coordinate, for now.
        vref_coords[d] = 0.5;

    // TODO somewhere else in the program, figure out how to set the precision
    const Float tol_ref = 1e-4;
    const int32 max_steps = 10;

    // Find solution.
    bool converged = (IterativeMethod::solve (stats, stepper, vref_coords, max_steps,
                                              tol_ref) == IterativeMethod::Converged);

    ref_coords = { vref_coords[0], vref_coords[1] };
    ray_dist = vref_coords[2];

    return (converged && surf_elem.is_inside (ref_coords) &&
            ray.m_near <= ray_dist && ray_dist < ray.m_far);
  }

  /*
   * Adapters to conform to simpler interfaces.
   */

  // Returns true if an intersection was found.
  // Adapter: Substitutes element (dmesh.get_elem(el_idx)).
  //  DRAY_EXEC static bool intersect( stats::IterativeProfile &iter_prof,
  //                                   const DeviceMesh<ElemT> &dmesh,  // Should be 2D device mesh.
  //                                   int32 el_idx,
  //                                   const Ray &ray,
  //                                   const AABB<2> &guess_domain,
  //                                   Vec<Float,2> &ref_coords,
  //                                   Float &ray_dist,
  //                                   bool use_init_guess = false)
  //  {
  //    return intersect(iter_prof,
  //                     dmesh.get_elem(el_idx),  // <--- substitute element.
  //                     ray,
  //                     guess_domain,
  //                     ref_coords,
  //                     ray_dist,
  //                     use_init_guess);
  //  }
  //
  //  // Returns true if an intersection was found.
  //  // Adapter: Substitutes dummy iter_prof (iteration counter).
  //  DRAY_EXEC static bool intersect( const ElemT &mesh_elem,
  //                                   const Ray &ray,
  //                                   const AABB<2> &guess_domain,
  //                                   Vec<Float,2> &ref_coords,
  //                                   Float &ray_dist,
  //                                   bool use_init_guess = false)
  //  {
  //    stats::IterativeProfile iter_prof;
  //    iter_prof.construct();
  //    return intersect(iter_prof,          // <--- substitute dummy iter_prof.
  //                     mesh_elem,
  //                     ray,
  //                     guess_domain,
  //                     ref_coords,
  //                     ray_dist,
  //                     use_init_guess);
  //  }
  //
  //  // Returns true if an intersection was found.
  //  // Adapter: Both substitutes mesh element and provides dummy iter_prof.
  //  DRAY_EXEC static bool intersect( const DeviceMesh<ElemT> &dmesh,
  //                                   int32 el_idx,
  //                                   const Ray &ray,
  //                                   const AABB<2> &guess_domain,
  //                                   Vec<Float,2> &ref_coords,
  //                                   Float &ray_dist,
  //                                   bool use_init_guess = false)
  //  {
  //    stats::IterativeProfile iter_prof;
  //    iter_prof.construct();
  //    return intersect(iter_prof,          // <--- substitute dummy iter_prof.
  //                     dmesh,              // <--- (dmesh, el_idx) will be substituted.
  //                     el_idx,             //
  //                     ray,
  //                     guess_domain,
  //                     ref_coords,
  //                     ray_dist,
  //                     use_init_guess);
  //  }
  //
  //
  //
  //  // Returns true if an intersection was found.
  //  // Adapter: Substitutes element (dmesh.get_elem(el_idx)).
  //  DRAY_EXEC static bool intersect_local( stats::IterativeProfile &iter_prof,
  //                                         const DeviceMesh<ElemT> &dmesh,
  //                                         int32 el_idx,
  //                                         const Ray &ray,
  //                                         Vec<Float,2> &ref_coords,
  //                                         Float &ray_dist,
  //                                         bool use_init_guess = false)
  //  {
  //    return intersect_local(iter_prof,
  //                           dmesh.get_elem(el_idx),   // <--- substitute mesh element.
  //                           ray,
  //                           ref_coords,
  //                           ray_dist,
  //                           use_init_guess);
  //  }
  //
  //  // Returns true if an intersection was found.
  //  // Adapter: Substitutes dummy iter_prof (iteration counter).
  //  DRAY_EXEC static bool intersect_local( const ElemT &mesh_elem,
  //                                         const Ray &ray,
  //                                         Vec<Float,2> &ref_coords,
  //                                         Float &ray_dist,
  //                                         bool use_init_guess = false)
  //  {
  //    stats::IterativeProfile iter_prof;
  //    iter_prof.construct();
  //    return intersect_local(iter_prof,       // <--- substitute dummy iter_prof.
  //                           mesh_elem,
  //                           ray,
  //                           ref_coords,
  //                           ray_dist,
  //                           use_init_guess);
  //  }
  //
  //  // Returns true if an intersection was found.
  //  // Adapter: Both substitutes mesh element and provides dummy iter_prof.
  //  DRAY_EXEC static bool intersect_local( const DeviceMesh<ElemT> &dmesh,
  //                                         int32 el_idx,
  //                                         const Ray &ray,
  //                                         Vec<Float,2> &ref_coords,
  //                                         Float &ray_dist,
  //                                         bool use_init_guess = false)
  //  {
  //    stats::IterativeProfile iter_prof;
  //    iter_prof.construct();
  //    return intersect_local(iter_prof,     // <--- substitute dummy iter_prof.
  //                           dmesh,         // <--- (dmesh, el_idx) will be substituted.
  //                           el_idx,        //
  //                           ray,
  //                           ref_coords,
  //                           ray_dist,
  //                           use_init_guess);
  //  }
};

} // namespace dray
#endif
