// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_DEVICE_MESH_HPP
#define DRAY_DEVICE_MESH_HPP

#include <dray/dray_config.h>
#include <dray/dray_exports.h>

#include <dray/data_model/subref.hpp>
#include <dray/data_model/element.hpp>
#include <dray/data_model/grid_function.hpp>
#include <dray/data_model/unstructured_mesh.hpp>
#include <dray/aabb.hpp>
#include <dray/array_utils.hpp>
#include <dray/device_bvh.hpp>
#include <dray/location.hpp>
#include <dray/subdivision_search.hpp>
#include <dray/vec.hpp>

#include <dray/utils/appstats.hpp>

namespace dray
{

template <int32 dim, ElemType etype, int32 P_order>
using MeshElem = Element<dim, 3u, etype, P_order>;

/*
 * @class DeviceMesh
 * @brief Device-safe access to a collection of elements
 * (just knows about the geometry, not fields).
 */
template <class ElemT> struct DeviceMesh
{
  static constexpr auto dim = ElemT::get_dim ();
  static constexpr auto etype = ElemT::get_etype ();

  DeviceMesh (UnstructuredMesh<ElemT> &mesh, bool use_bvh = true);
  DeviceMesh () = delete;

  //TODO use a DeviceGridFunction

  const int32 *m_idx_ptr;
  const Vec<Float, 3u> *m_val_ptr;
  const int32 m_poly_order;
  // bvh related data
  const DeviceBVH m_bvh;
  const SubRef<dim, etype> *m_ref_boxs;
  // if the element was subdivided m_ref_boxs
  // contains the sub-ref box of the original element
  // TODO: this should be married with BVH

  DRAY_EXEC_ONLY typename AdaptGetOrderPolicy<ElemT>::type get_order_policy() const
  {
    return adapt_get_order_policy(ElemT{}, m_poly_order);
  }

  DRAY_EXEC_ONLY ElemT get_elem (int32 el_idx) const;
  DRAY_EXEC_ONLY Location locate (const Vec<Float, 3> &point) const;
};


// ------------------ //
// DeviceMesh methods //
// ------------------ //

template <class ElemT>
DeviceMesh<ElemT>::DeviceMesh (UnstructuredMesh<ElemT> &mesh, bool use_bvh)
: m_idx_ptr (mesh.m_dof_data.m_ctrl_idx.get_device_ptr_const ()),
  m_val_ptr (mesh.m_dof_data.m_values.get_device_ptr_const ()),
  m_poly_order (mesh.m_poly_order),
  // hack to get around that constructing the bvh needs the device mesh
  m_bvh (use_bvh ? mesh.get_bvh(): BVH()),
  m_ref_boxs (mesh.m_ref_aabbs.get_device_ptr_const ())
{
}

template <class ElemT>
DRAY_EXEC_ONLY ElemT DeviceMesh<ElemT>::get_elem (int32 el_idx) const
{
  // We are just going to assume that the elements in the data store
  // are in the same position as their id, el_id==el_idx.
  ElemT ret;

  auto shape = adapt_get_shape(ElemT{});
  auto order_p = get_order_policy();
  const int32 dofs_per  = eattr::get_num_dofs(shape, order_p);

  const int32 elem_offset = dofs_per * el_idx;

  using DofVec = Vec<Float, 3u>;
  SharedDofPtr<DofVec> dof_ptr{ elem_offset + m_idx_ptr, m_val_ptr };
  ret.construct (el_idx, dof_ptr, m_poly_order);
  return ret;
}

//
// HACK to avoid calling eval_inverse() on 2x3 elements.
//
namespace detail
{

template <int32 d> struct LocateHack
{
  template <class ElemT>
  static bool DRAY_EXEC_ONLY eval_inverse (const ElemT &elem,
                                           stats::Stats &stats,
                                           const Vec<typename ElemT::get_precision, 3u> &world_coords,
                                           const SubRef<2, ElemT::get_etype()> &guess_domain,
                                           Vec<typename ElemT::get_precision, 2> &ref_coords,
                                           bool use_init_guess = false)
  {
    return false;
  }

  // non-stats version
  template <class ElemT>
  static bool DRAY_EXEC_ONLY eval_inverse (const ElemT &elem,
                                           const Vec<typename ElemT::get_precision, 3u> &world_coords,
                                           const SubRef<2, ElemT::get_etype()> &guess_domain,
                                           Vec<typename ElemT::get_precision, 2> &ref_coords,
                                           bool use_init_guess = false)
  {

    return false;
  }
};

// 3D: Works.
template <> struct LocateHack<3u>
{
  template <class ElemT>
  static bool DRAY_EXEC_ONLY eval_inverse (const ElemT &elem,
                                           stats::Stats &stats,
                                           const Vec<typename ElemT::get_precision, 3u> &world_coords,
                                           const SubRef<3, ElemT::get_etype()> &guess_domain,
                                           Vec<typename ElemT::get_precision, 3u> &ref_coords,
                                           bool use_init_guess = false)
  {
    /// return elem.eval_inverse (stats, world_coords, guess_domain, ref_coords, use_init_guess);

    // Bypass the subdivision search.
    //
    if (!use_init_guess)
      ref_coords = subref_center(guess_domain);
    return elem.eval_inverse_local (stats, world_coords, ref_coords);
  }

  // non-stats version
  template <class ElemT>
  static bool DRAY_EXEC_ONLY eval_inverse (const ElemT &elem,
                                           const Vec<typename ElemT::get_precision, 3u> &world_coords,
                                           const SubRef<3, ElemT::get_etype()> &guess_domain,
                                           Vec<typename ElemT::get_precision, 3u> &ref_coords,
                                           bool use_init_guess = false)
  {
    /// return elem.eval_inverse (world_coords, guess_domain, ref_coords, use_init_guess);

    // Bypass the subdivision search.
    if (!use_init_guess)
      ref_coords = subref_center(guess_domain);
    return elem.eval_inverse_local (world_coords, ref_coords);
  }
};

// This is a better hack. Current state of things is we always treat 2d data as
// having a 0 z-coordinate. In the past, we would return false if the element was
// topologically 2d since the resulting element had a 2x3 matrix that was not invertable.
// This is a workaround that ingores the z-coordinate and return the correct result
// for a point location for a '2d' mesh. This still won't work on data like external faces,
// where the z-coordinate means something, but its better than not finding anything at all.
// One day, we might try to natively handle 2d data where the element matrix would be 2x2,
// but that is another day
template <> struct LocateHack<2>
{
  template <class ElemT>
  static bool DRAY_EXEC_ONLY eval_inverse (const ElemT &elem,
                                           stats::Stats &stats,
                                           const Vec<typename ElemT::get_precision, 3u> &world_coords,
                                           const SubRef<2, ElemT::get_etype()> &guess_domain,
                                           Vec<typename ElemT::get_precision, 2> &ref_coords,
                                           bool use_init_guess = false)
  {
    struct Stepper
    {
      DRAY_EXEC typename IterativeMethod::StepStatus operator() (Vec<Float, 2> &x) const
      {
        // project back onto the element
        for(int i = 0; i < 2; ++i)
        {
          x[i] = fminf(Float(1.f), fmaxf(x[i], Float(0.f)));
        }

        Vec<Float, 3> delta_y;
        Vec<Vec<Float, 3>, 2> j_col;
        Matrix<Float, 3, 3> jacobian;
        delta_y = m_transf.eval_d (x, j_col);
        delta_y = m_target - delta_y;

        for (int32 rdim = 0; rdim < 2; rdim++)
          jacobian.set_col (rdim, j_col[rdim]);
        // set this so the matrix is invertable
        jacobian.set_col(2, {{0.f, 0.f, 1.f}});

        bool inverse_valid;
        Vec<Float, 3> delta_x;
        delta_x = matrix_mult_inv (jacobian, delta_y, inverse_valid);

        //if (!inverse_valid) return IterativeMethod::Abort;

        // get the two coordinates we care about
        Vec<Float, 2> delta_x_2d = {{delta_x[0], delta_x[1]}};
        x = x + delta_x_2d;
        return IterativeMethod::Continue;
      }

      Element_impl<ElemT::get_dim(), 3, ElemT::get_etype(), ElemT::get_P()> m_transf;
      Vec<Float, 3> m_target;

    } stepper{ elem, world_coords };
    // TODO somewhere else in the program, figure out how to set the precision
    // based on the gradient and the image resolution.
    const Float tol_ref = 1e-4f;
    const int32 max_steps = 20;
    // Find solution.
    bool found = (IterativeMethod::solve (stats, stepper, ref_coords, max_steps,
                                          tol_ref) == IterativeMethod::Converged &&
                  elem.is_inside (ref_coords, tol_ref));
    return found;
  }

  // non-stats version
  template <class ElemT>
  static bool DRAY_EXEC_ONLY eval_inverse (const ElemT &elem,
                                           const Vec<typename ElemT::get_precision, 3u> &world_coords,
                                           const SubRef<2, ElemT::get_etype()> &guess_domain,
                                           Vec<typename ElemT::get_precision, 2> &ref_coords,
                                           bool use_init_guess = false)
  {
    stats::Stats stats;
    return eval_inverse(elem, stats, world_coords, guess_domain, ref_coords, use_init_guess);
  }
};
} // namespace detail

template <class ElemT>
DRAY_EXEC_ONLY Location DeviceMesh<ElemT>::locate (const Vec<Float, 3> &point) const
{
  constexpr auto etype = ElemT::get_etype ();  //TODO use type trait instead

  Location loc{ -1, { -1.f, -1.f, -1.f } };

  int32 todo[64];
  int32 current_node = 0;
  int32 stackptr = 0;

  constexpr int32 barrier = -2000000000;
  todo[stackptr] = barrier;
  while (current_node != barrier)
  {
    if (current_node > -1)
    {
      // inner node
      const Vec<float32, 4> first4 =
      const_get_vec4f (&m_bvh.m_inner_nodes[current_node + 0]);
      const Vec<float32, 4> second4 =
      const_get_vec4f (&m_bvh.m_inner_nodes[current_node + 1]);
      const Vec<float32, 4> third4 =
      const_get_vec4f (&m_bvh.m_inner_nodes[current_node + 2]);

      bool in_left = true;
      if (point[0] < first4[0]) in_left = false;
      if (point[1] < first4[1]) in_left = false;
      if (point[2] < first4[2]) in_left = false;

      if (point[0] > first4[3]) in_left = false;
      if (point[1] > second4[0]) in_left = false;
      if (point[2] > second4[1]) in_left = false;

      bool in_right = true;
      if (point[0] < second4[2]) in_right = false;
      if (point[1] < second4[3]) in_right = false;
      if (point[2] < third4[0]) in_right = false;

      if (point[0] > third4[1]) in_right = false;
      if (point[1] > third4[2]) in_right = false;
      if (point[2] > third4[3]) in_right = false;

      if (!in_left && !in_right)
      {
        // pop the stack and continue
        current_node = todo[stackptr];
        stackptr--;
      }
      else
      {
        const Vec<float32, 4> children =
        const_get_vec4f (&m_bvh.m_inner_nodes[current_node + 3]);
        int32 l_child;
        constexpr int32 isize = sizeof (int32);
        // memcpy the int bits hidden in the floats
        memcpy (&l_child, &children[0], isize);
        int32 r_child;
        memcpy (&r_child, &children[1], isize);

        current_node = (in_left) ? l_child : r_child;

        if (in_left && in_right)
        {
          stackptr++;
          todo[stackptr] = r_child;
          // TODO: if we are in both children we could
          // go down the "closer" first by perhaps the distance
          // from the point to the center of the aabb
        }
      }
    }
    else
    {
      // leaf node
      // leafs are stored as negative numbers
      current_node = -current_node - 1; // swap the neg address
      const int32 el_idx = m_bvh.m_leaf_nodes[current_node];
      const int32 ref_box_id = m_bvh.m_aabb_ids[current_node];
      SubRef<dim, etype> ref_start_box = m_ref_boxs[ref_box_id];
      bool use_init_guess = true;
      // locate the point

      Vec<Float, dim> el_coords;

      bool found;
      found = detail::LocateHack<ElemT::get_dim ()>::template eval_inverse<ElemT> (
      get_elem (el_idx), point, ref_start_box, el_coords, use_init_guess);

      if (found)
      {
        loc.m_cell_id = el_idx;
        loc.m_ref_pt[0] = el_coords[0];
        loc.m_ref_pt[1] = el_coords[1];
        if (dim == 3)
        {
          loc.m_ref_pt[2] = el_coords[2];
        }
        break;
      }

      current_node = todo[stackptr];
      stackptr--;
    }
  } // while

  return loc;
}

} // namespace dray


#endif
