// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include <dray/dray_config.h>
#include <dray/dray_exports.h>

#include <dray/filters/isosurfacing.hpp>
#include <dray/filters/marching_cubes.hpp>
#include <dray/error.hpp>
#include <dray/dispatcher.hpp>
#include <dray/array_utils.hpp>

#include <RAJA/RAJA.hpp>
#include <dray/policies.hpp>

#include <dray/data_model/unstructured_mesh.hpp>
#include <dray/data_model/mesh.hpp>
#include <dray/data_model/field.hpp>
#include <dray/data_model/device_mesh.hpp>
#include <dray/data_model/device_field.hpp>
#include <dray/data_model/elem_attr.hpp>
#include <dray/data_model/iso_ops.hpp>
#include <dray/data_model/detached_element.hpp>

#include <sstream>


  // internal, will be undef'd at end of file.
#ifdef DRAY_CUDA_ENABLED
#define THROW_LOGIC_ERROR(msg) assert(!(msg) && false);
#elif defined(DRAY_HIP_ENABLED)
#define THROW_LOGIC_ERROR(msg) assert(!(msg) && false);
#else
#define THROW_LOGIC_ERROR(msg) throw std::logic_error(msg);
#endif


// ----------------------------------------------------
// Isosurfacing approach based on
//   https://dx.doi.org/10.1016/j.cma.2016.10.019
//
// @article{FRIES2017759,
//   title = "Higher-order meshing of implicit geometries—Part I: Integration and interpolation in cut elements",
//   journal = "Computer Methods in Applied Mechanics and Engineering",
//   volume = "313",
//   pages = "759 - 784",
//   year = "2017",
//   issn = "0045-7825",
//   doi = "https://doi.org/10.1016/j.cma.2016.10.019",
//   url = "http://www.sciencedirect.com/science/article/pii/S0045782516308696",
//   author = "T.P. Fries and S. Omerović and D. Schöllhammer and J. Steidl",
// }
// ----------------------------------------------------

namespace dray
{
  // -----------------------
  // Getter/setters
  // -----------------------
  void ExtractIsosurface::iso_field(const std::string field_name)
  {
    m_iso_field_name = field_name;
  }

  std::string ExtractIsosurface::iso_field() const
  {
    return m_iso_field_name;
  }

  void ExtractIsosurface::iso_value(const float32 iso_value)
  {
    m_iso_value = iso_value;
  }

  Float ExtractIsosurface::iso_value() const
  {
    return m_iso_value;
  }
  // -----------------------


  // LocationSet for remapping non-iso fields.
  struct LocationSet
  {
    GridFunction<3> m_rcoords;     // one per dof per element.
    Array<int32> m_host_cell_id;   // one per element.
  };

  template <typename OutShape, int32 MP, class FElemT>
  std::shared_ptr<Field> ReMapField_execute(const LocationSet &location_set,
                                            OutShape,
                                            OrderPolicy<MP> mesh_order_p,
                                            UnstructuredField<FElemT> &in_field);

  // ReMapFieldFunctor
  template <typename OutShape, int32 P>
  struct ReMapFieldFunctor
  {
    LocationSet m_location_set;
    OrderPolicy<P> m_mesh_order_p;

    std::shared_ptr<Field> m_out_field_ptr;

    ReMapFieldFunctor(const LocationSet &ls, OrderPolicy<P> mesh_order_p)
      : m_location_set(ls),
        m_mesh_order_p(mesh_order_p),
        m_out_field_ptr(nullptr)
    { }

    template <typename FieldT>
    void operator()(FieldT &field)
    {
      m_out_field_ptr = ReMapField_execute(m_location_set, OutShape(), m_mesh_order_p, field);
    }
  };


  /*
  template <typename T>
  T array_sum(const Array<T> &arr)
  {
    RAJA::ReduceSum<reduce_policy, T> asum;
    const T *aptr = arr.get_device_ptr_const();
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, arr.size()), [=] DRAY_LAMBDA (int32 i) {
      asum += aptr[i];
    });
    return asum.get();
  }
  */

  // View2D{};
  template <typename PtrT>
  struct View2D
  {
    PtrT m_base;
    int32 m_sz1d;

    DRAY_EXEC PtrT operator[](int32 r)
    {
      return m_base + (r * m_sz1d);
    }
  };

  // view_2d()
  template <typename T>
  DRAY_EXEC View2D<T*> view_2d(T *base, int32 sz1d)
  {
    return View2D<T*>{base, sz1d};
  }

  // view_2d()
  template <typename T>
  DRAY_EXEC View2D<WriteDofPtr<T>> view_2d(WriteDofPtr<T> &base, int32 sz1d)
  {
    return View2D<WriteDofPtr<T>>{base, sz1d};
  }

  // view_2d()
  template <typename T>
  DRAY_EXEC View2D<ReadDofPtr<T>> view_2d(ReadDofPtr<T> &base, int32 sz1d)
  {
    return View2D<ReadDofPtr<T>>{base, sz1d};
  }


  // my_copy_n()
  template <typename DofT>
  DRAY_EXEC void my_copy_n(DofT *dest_elem, const DofT *src_elem, const int32 npe)
  {
    for (int32 nidx = 0; nidx < npe; ++nidx)
      dest_elem[nidx] = src_elem[nidx];
  }

  // my_copy_n()
  template <typename DofT>
  DRAY_EXEC void my_copy_n(DofT *dest_elem, const ReadDofPtr<DofT> &src_elem, const int32 npe)
  {
    for (int32 nidx = 0; nidx < npe; ++nidx)
      dest_elem[nidx] = src_elem[nidx];
  }

  // my_copy_n()
  template <typename DofT>
  DRAY_EXEC void my_copy_n(WriteDofPtr<DofT> dest_elem, const ReadDofPtr<DofT> &src_elem, const int32 npe)
  {
    for (int32 nidx = 0; nidx < npe; ++nidx)
      dest_elem[nidx] = src_elem[nidx];
  }

  // my_swap()
  template <typename T>
  DRAY_EXEC void my_swap(T &x, T &y)
  {
    T tmp = x;
    x = y;
    y = tmp;
  }

  // my_swap_n()
  template <typename DofT>
  DRAY_EXEC void my_swap_n(WriteDofPtr<DofT> elemA, WriteDofPtr<DofT> elemB, const int32 npe)
  {
    for (int32 nidx = 0; nidx < npe; ++nidx)
    {
      DofT tmp = elemA[nidx];
      elemA[nidx] = elemB[nidx];
      elemB[nidx] = tmp;
    }
  }


  /** pick_candidate() */
  template <ElemType etype>
  DRAY_EXEC int32 pick_candidate(const SubRef<3, etype> * subrefs,
                                 const eops::IsocutInfo * infos,
                                 const int32 n)
  {
    int32 picked = 0;
    Float picked_length = subref_length(subrefs[picked]);
    Float maybe_length;
    for (int32 maybe = 1; maybe < n; ++maybe)
      if ((maybe_length = subref_length(subrefs[maybe])) > picked_length)
      {
        picked = maybe;
        picked_length = maybe_length;
      }
    return picked;
  }

  /** is_empty() */
  DRAY_EXEC bool is_empty(const eops::IsocutInfo &info)
  {
    return !bool(info.m_cut_type_flag & eops::IsocutInfo::Cut);
  }

  /** is_simple() */
  DRAY_EXEC bool is_simple(const eops::IsocutInfo &info)
  {
    using eops::IsocutInfo;
    return bool(info.m_cut_type_flag & (IsocutInfo::CutSimpleTri | IsocutInfo::CutSimpleQuad));
  }

  /**
   * subdivide_host_elem()
   *
   * @pre out_dofs[0] .. out_dofs[npe*budget - 1] is available to be overwritten.
   * @post The first out_size subelements in the budget are valid.
   * @post budget_insufficient is true or the entire element was subdivided into simple cuts.
   */
  template <typename ShapeT, int32 P>
  DRAY_EXEC void subdivide_host_elem( const ShapeT,
                                      const OrderPolicy<P> order_p,
                                      const int32 budget,
                                      const ReadDofPtr<Vec<Float, 1>> & in_elem,
                                      const Float isoval,
                                      WriteDofPtr<Vec<Float, 1>> out_dofs,
                                      SubRef<3, eattr::get_etype(ShapeT())> * subrefs,
                                      eops::IsocutInfo * infos,
                                      bool &budget_insufficient,
                                      int32 &out_size )
  {
    //   |      |        .              |
    //   | kept | (cand) .              | (q_end)
    //   |      |       queue           |
    //   |      |        .              |

    assert(budget >= 1);

    constexpr ElemType etype = eattr::get_etype(ShapeT());
    using eops::IsocutInfo;
    using SubRefT = SubRef<3, etype>;
    using eops::measure_isocut;

    const int32 p = eattr::get_order(order_p);
    const int32 npe = eattr::get_num_dofs(ShapeT(), order_p);

    ReadDofPtr<Vec<Float, 1>> out_dofs_read = out_dofs.to_readonly_dof_ptr();

    my_copy_n(view_2d(out_dofs, npe)[0], in_elem, npe);
    subrefs[0] = ref_universe(RefSpaceTag<3, etype>());
    infos[0] = measure_isocut(ShapeT(), view_2d(out_dofs_read, npe)[0], isoval, p);

    int32 q_sz = 1;
    int32 kept_sz = 0;

    budget_insufficient = false;

    while (q_sz > 0 && budget > q_sz + kept_sz)
    {
      const int32 picked = kept_sz + pick_candidate(subrefs + kept_sz,
                                                    infos + kept_sz,
                                                    q_sz);
      const int32 candidate = kept_sz;
      const int32 q_end = kept_sz + q_sz;
      assert(candidate <= picked  &&  picked < q_end);


      if (picked != candidate)
      {
        my_swap_n(view_2d(out_dofs, npe)[candidate], view_2d(out_dofs, npe)[picked], npe);
        my_swap(subrefs[candidate], subrefs[picked]);
        my_swap(infos[candidate], infos[picked]);
      }
      q_sz--;

      // The pre-loop invariant (budget > q_sz + kept_sz)
      // ensures there is enough room to perform a binary split.

      if (is_simple(infos[candidate]))
      {
        kept_sz++;
      }
      else if (is_empty(infos[candidate]))
      {
        if (q_sz > 0)
        {
          my_copy_n(view_2d(out_dofs, npe)[candidate], view_2d(out_dofs_read, npe)[q_end-1], npe);
          subrefs[candidate] = subrefs[q_end-1];
          infos[candidate] = infos[q_end-1];
        }
      }
      else
      {
        const Split<etype> binary_split = pick_iso_simple_split(ShapeT(), infos[candidate]);

        // Prepare for in-place splits.
        my_copy_n(view_2d(out_dofs, npe)[q_end], view_2d(out_dofs_read, npe)[candidate], npe);
        subrefs[q_end] = subrefs[candidate];

        // Split subrefs.
        subrefs[q_end] = split_subref(subrefs[candidate], binary_split);

        // Split subelements.
        split_inplace(ShapeT(), order_p, view_2d(out_dofs, npe)[candidate], binary_split);
        split_inplace(ShapeT(), order_p, view_2d(out_dofs, npe)[q_end], binary_split.get_complement());

        // Update infos after split.
        infos[candidate] = measure_isocut(ShapeT(), view_2d(out_dofs_read, npe)[candidate], isoval, p);
        infos[q_end]     = measure_isocut(ShapeT(), view_2d(out_dofs_read, npe)[q_end], isoval, p);

        q_sz += 2;
      }
    }

    out_size = kept_sz;

    if (q_sz > 0)
      budget_insufficient = true;
  }








  //
  // execute(topo, field)
  //
  template <class MElemT, class FElemT>
  std::pair<DataSet,DataSet> ExtractIsosurface_execute( UnstructuredMesh<MElemT> &mesh,
                                                        UnstructuredField<FElemT> &field,
                                                        Float iso_value,
                                                        DataSet *input_dataset)
  {
    // Overview:
    //   - Subdivide field elements until isocut is simple. Yields sub-ref and sub-coeffs.
    //   - Extract isopatch coordinates relative to the coord-system of sub-ref.
    //   - Transform isopatch coordinates to reference space via (sub-ref)^{-1}.
    //   - Transform isopatch coordinates to world space via mesh element.
    //   - Outside this function, the coordinates are converted to Bernstein ctrl points.

    static_assert(FElemT::get_ncomp() == 1, "Can't take isosurface of a vector field");

    using eops::IsocutInfo;

    const Float isoval = iso_value;  // Local for capture
    const int32 n_el_in = field.get_num_elem();
    DeviceField<FElemT> dfield(field);

    constexpr int32 init_budget = 5;
    constexpr int32 budget_factor = 2;
    constexpr int32 pass_limit = 4;

    // host_elem_budgets[]
    Array<int32> host_elem_budgets;
    host_elem_budgets.resize(n_el_in);
    array_memset(host_elem_budgets, init_budget);
    int32 * host_elem_budget_ptr = host_elem_budgets.get_device_ptr();

    int32 total_budgeted_subelems;
    Array<int32> offsets_array = array_exc_scan_plus(host_elem_budgets, total_budgeted_subelems);

    Array<int32> budget_exceeded;
    budget_exceeded.resize(n_el_in);
    array_memset(budget_exceeded, int32(false));
    int32 * budget_exceeded_ptr = budget_exceeded.get_device_ptr();

    Array<int32> out_sizes_array;
    out_sizes_array.resize(n_el_in);
    int32 * out_sizes_ptr = out_sizes_array.get_device_ptr();

    const auto field_order_p = dfield.get_order_policy();
    constexpr auto shape3d = adapt_get_shape<FElemT>();
    const int32 field3d_npe = eattr::get_num_dofs(shape3d, field_order_p);

    // Allocate and get ptrs to sub-element field dofs.
    GridFunction<1> field_sub_elems;
    field_sub_elems.resize_counting(total_budgeted_subelems, field3d_npe);

    // Allocate and get ptrs to sub-element coords within host elements.
    using SubRefT = typename get_subref<FElemT>::type;
    Array<SubRefT> subref_array;
    subref_array.resize(total_budgeted_subelems);

    // Allocate and get ptrs to sub-element isocut metrics.
    Array<IsocutInfo> info_array;
    info_array.resize(total_budgeted_subelems);

    Array<int32> pending_host_elems = array_counting(n_el_in, 0, 1);

    int32 num_passes = 0;
    while (pending_host_elems.size() > 0 && num_passes < pass_limit)
    {
      printf("------------------------------------------\n"
             "Pass #%d\n"
             "------------------------------------------\n", num_passes);

      const int32 * pending_host_elem_ptr = pending_host_elems.get_device_ptr_const();
      const int32 * offset_ptr = offsets_array.get_device_ptr_const();

      DeviceGridFunction<1> out_field_dgf(field_sub_elems);
      SubRefT * subref_ptr = subref_array.get_device_ptr();
      IsocutInfo * info_ptr = info_array.get_device_ptr();

      const int32 num_pending = pending_host_elems.size();
      RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_pending), [=] DRAY_LAMBDA (int32 pend_host_idx) {
        const int32 host_elem_id = pending_host_elem_ptr[pend_host_idx];

        const ReadDofPtr<Vec<Float, 1>> rdp = dfield.get_elem(host_elem_id).read_dof_ptr();

        const int32 budget = host_elem_budget_ptr[host_elem_id];
        int32 out_sz;
        bool exceeded;

        constexpr auto shape = adapt_get_shape<FElemT>();
        const auto forder_p = dfield.get_order_policy();

        const int32 offset = offset_ptr[host_elem_id];
        WriteDofPtr<Vec<Float, 1>> out_dofs = out_field_dgf.get_wdp(offset);

        SubRefT * subrefs = subref_ptr + offset;
        IsocutInfo * infos = info_ptr + offset;

        subdivide_host_elem(shape, forder_p, budget, rdp, isoval, out_dofs, subrefs, infos, exceeded, out_sz);

        if (exceeded)
          host_elem_budget_ptr[host_elem_id] *= budget_factor;
        else
          host_elem_budget_ptr[host_elem_id] = out_sz;
        out_sizes_ptr[host_elem_id] = out_sz;
        budget_exceeded_ptr[host_elem_id] = exceeded;
      });

      // Update list of pending host elements.
      pending_host_elems = index_flags(budget_exceeded);

      if (pending_host_elems.size() > 0)
      {
        Array<int32> new_offsets_array = array_exc_scan_plus(host_elem_budgets, total_budgeted_subelems);
        const int32 * new_offset_ptr = new_offsets_array.get_device_ptr_const();

        printf("Re-budgeting with total_budgeted_sublems==%d\n", total_budgeted_subelems);

        // Move finished outputs to make room for unfinished outputs.

        // new_field_sub_elems
        GridFunction<1> new_field_sub_elems;
        new_field_sub_elems.resize_counting(total_budgeted_subelems, field3d_npe);
        Vec<Float, 1> * new_out_dof_ptr = new_field_sub_elems.m_values.get_device_ptr();

        // new_subref_array
        Array<SubRefT> new_subref_array;
        new_subref_array.resize(total_budgeted_subelems);
        SubRefT * new_subref_ptr = new_subref_array.get_device_ptr();

        // new_info_array
        Array<IsocutInfo> new_info_array;
        new_info_array.resize(total_budgeted_subelems);
        IsocutInfo * new_info_ptr = new_info_array.get_device_ptr();

        const Vec<Float, 1> * out_dof_ptr = field_sub_elems.m_values.get_device_ptr();
        RAJA::forall<for_policy>(RAJA::RangeSegment(0, n_el_in), [=] DRAY_LAMBDA (int32 host_elem_id) {
          const int32 old_offset = offset_ptr[host_elem_id];
          const int32 new_offset = new_offset_ptr[host_elem_id];

          const Vec<Float, 1> * old_out_dofs = out_dof_ptr + old_offset * field3d_npe;
          const SubRefT * old_subrefs = subref_ptr + old_offset;
          const IsocutInfo * old_infos = info_ptr + old_offset;

          Vec<Float, 1> * new_out_dofs = new_out_dof_ptr + new_offset * field3d_npe;
          SubRefT * new_subrefs = new_subref_ptr + new_offset;
          IsocutInfo * new_infos = new_info_ptr + new_offset;

          const int32 out_sz = out_sizes_ptr[host_elem_id];
          for (int32 i = 0; i < out_sz; ++i)
          {
            my_copy_n(view_2d(new_out_dofs, field3d_npe)[i], view_2d(old_out_dofs, field3d_npe)[i], field3d_npe);
            new_subrefs[i] = old_subrefs[i];
            new_infos[i] = old_infos[i];
          }
        });
        field_sub_elems = new_field_sub_elems;
        subref_array = new_subref_array;
        info_array = new_info_array;

        offsets_array = new_offsets_array;
      }

      num_passes++;

      printf("\n");
    }

    // Record tri-shaped and quad-shaped cuts.

    // Allocate and get ptrs to sub-element output activations.
    Array<int32> keepme_tri, keepme_quad;
    keepme_tri.resize(total_budgeted_subelems);
    keepme_quad.resize(total_budgeted_subelems);
    array_memset_zero(keepme_tri);
    array_memset_zero(keepme_quad);
    int32 *keepme_tri_ptr = keepme_tri.get_device_ptr();
    int32 *keepme_quad_ptr = keepme_quad.get_device_ptr();

    Array<int32> host_cell_array;
    host_cell_array.resize(total_budgeted_subelems);
    int32 * host_cell_ptr = host_cell_array.get_device_ptr();

    const IsocutInfo * info_ptr = info_array.get_device_ptr_const();
    const int32 * offset_ptr = offsets_array.get_device_ptr_const();
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, n_el_in), [=] DRAY_LAMBDA (int32 host_elem_id) {
      const bool exceeded = budget_exceeded_ptr[host_elem_id];
      const int32 out_sz = out_sizes_ptr[host_elem_id];
      const int32 offset = offset_ptr[host_elem_id];

      for (int32 sub_i = 0; sub_i < out_sz; ++sub_i)
      {
        const IsocutInfo info = info_ptr[offset + sub_i];
        host_cell_ptr[offset + sub_i] = host_elem_id;
        if (info.m_cut_type_flag & IsocutInfo::CutSimpleTri)
          keepme_tri_ptr[offset + sub_i] = true;
        if (info.m_cut_type_flag & IsocutInfo::CutSimpleQuad)
          keepme_quad_ptr[offset + sub_i] = true;
      }

      if (exceeded)
      {
        printf("(num_passes==%d) Very complex geometry in host cell %d.\n", num_passes, host_elem_id);
      }
    });

    GridFunction<1> field_sub_elems_tri;
    Array<int32> kept_indices_tri = index_flags(keepme_tri);
    Array<SubRefT> subrefs_tri = gather(subref_array, kept_indices_tri);
    Array<int32> host_cells_tri = gather(host_cell_array, kept_indices_tri);
    field_sub_elems_tri.m_values = gather(field_sub_elems.m_values, field3d_npe, kept_indices_tri);
    field_sub_elems_tri.m_ctrl_idx = array_counting(field3d_npe * kept_indices_tri.size(), 0, 1);
    field_sub_elems_tri.m_el_dofs = field3d_npe;
    field_sub_elems_tri.m_size_el = kept_indices_tri.size();
    field_sub_elems_tri.m_size_ctrl = field_sub_elems_tri.m_values.size();

    GridFunction<1> field_sub_elems_quad;
    Array<int32> kept_indices_quad = index_flags(keepme_quad);
    Array<SubRefT> subrefs_quad = gather(subref_array, kept_indices_quad);
    Array<int32> host_cells_quad = gather(host_cell_array, kept_indices_quad);
    field_sub_elems_quad.m_values = gather(field_sub_elems.m_values, field3d_npe, kept_indices_quad);
    field_sub_elems_quad.m_ctrl_idx = array_counting(field3d_npe * kept_indices_quad.size(), 0, 1);
    field_sub_elems_quad.m_el_dofs = field3d_npe;
    field_sub_elems_quad.m_size_el = kept_indices_quad.size();
    field_sub_elems_quad.m_size_ctrl = field_sub_elems_quad.m_values.size();

    const int32 num_sub_elems_tri = field_sub_elems_tri.m_size_el;
    const int32 num_sub_elems_quad = field_sub_elems_quad.m_size_el;

    // Now have the field values of each sub-element.
    // Create an output isopatch for each sub-element.
    // Use the FIELD order for the approximate isopatches.

    constexpr int32 out_order_policy_id = FElemT::get_P();
    const auto out_order_p = field_order_p;
    /// constexpr Order out_order_policy_id = General;     //
    /// const auto out_order_p = OrderPolicy<General>{3};  // Can use another order.
    const int32 out_order = eattr::get_order(out_order_p);
    const int32 out_tri_npe = eattr::get_num_dofs(ShapeTri(), out_order_p);
    const int32 out_quad_npe = eattr::get_num_dofs(ShapeQuad(), out_order_p);

    // Outputs for physical mesh coords of new surface elements.
    GridFunction<3> isopatch_coords_tri;
    GridFunction<3> isopatch_coords_quad;
    isopatch_coords_tri.resize_counting(num_sub_elems_tri, out_tri_npe);
    isopatch_coords_quad.resize_counting(num_sub_elems_quad, out_quad_npe);

    // Intermediate arrays to later map additional fields onto new surface elements.
    LocationSet locset_tri;
    LocationSet locset_quad;
    locset_tri.m_rcoords.resize_counting(num_sub_elems_tri, out_tri_npe);
    locset_quad.m_rcoords.resize_counting(num_sub_elems_quad, out_quad_npe);
    locset_tri.m_host_cell_id = host_cells_tri;
    locset_quad.m_host_cell_id = host_cells_quad;

    DeviceMesh<MElemT> dmesh(mesh);
    const Float iota = iso_value;

    // Extract triangle isopatches.
    {
      const SubRefT * subref_ptr = subrefs_tri.get_device_ptr_const();
      DeviceGridFunction<1> field_subel_dgf(field_sub_elems_tri);
      DeviceGridFunction<3> isopatch_dgf(isopatch_coords_tri);
      const int32 * host_cell_id_ptr = host_cells_tri.get_device_ptr_const();

      DeviceGridFunction<3> isopatch_r_dgf(locset_tri.m_rcoords);

      RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_sub_elems_tri), [=] DRAY_LAMBDA (int32 neid) {

        ReadDofPtr<Vec<Float, 1>> field_vals = field_subel_dgf.get_rdp(neid);
        WriteDofPtr<Vec<Float, 3>> coords = isopatch_dgf.get_wdp(neid);
        eops::reconstruct_isopatch(shape3d, ShapeTri(), field_vals, coords, iota, field_order_p, out_order_p);

        WriteDofPtr<Vec<Float, 3>> rcoords = isopatch_r_dgf.get_wdp(neid);

        const int32 host_cell_id = host_cell_id_ptr[neid];
        const MElemT melem = dmesh.get_elem(host_cell_id);
        for (int32 nidx = 0; nidx < out_tri_npe; ++nidx)
        {
          const Vec<Float, 3> rcoord = subref2ref(subref_ptr[neid], coords[nidx]);
          rcoords[nidx] = rcoord;
          coords[nidx] = melem.eval(rcoord);
        }
      });
    }

    // Extract quad isopatches.
    {
      const SubRefT * subref_ptr = subrefs_quad.get_device_ptr_const();
      DeviceGridFunction<1> field_subel_dgf(field_sub_elems_quad);
      DeviceGridFunction<3> isopatch_dgf(isopatch_coords_quad);
      const int32 * host_cell_id_ptr = host_cells_quad.get_device_ptr_const();

      int32 *host_cell_id_quad_ptr = locset_quad.m_host_cell_id.get_device_ptr();
      DeviceGridFunction<3> isopatch_r_dgf(locset_quad.m_rcoords);

      RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_sub_elems_quad), [=] DRAY_LAMBDA (int32 neid) {

        ReadDofPtr<Vec<Float, 1>> field_vals = field_subel_dgf.get_rdp(neid);
        WriteDofPtr<Vec<Float, 3>> coords = isopatch_dgf.get_wdp(neid);
        eops::reconstruct_isopatch(shape3d, ShapeQuad(), field_vals, coords, iota, field_order_p, out_order_p);

        WriteDofPtr<Vec<Float, 3>> rcoords = isopatch_r_dgf.get_wdp(neid);

        const int32 host_cell_id = host_cell_id_ptr[neid];
        const MElemT melem = dmesh.get_elem(host_cell_id);
        for (int32 nidx = 0; nidx < out_quad_npe; ++nidx)
        {
          const Vec<Float, 3> rcoord = subref2ref(subref_ptr[neid], coords[nidx]);
          rcoords[nidx] = rcoord;
          coords[nidx] = melem.eval(rcoord);
        }
      });
    }

    using IsoPatchTriT = Element<2, 3, Simplex, out_order_policy_id>;
    using IsoPatchQuadT = Element<2, 3, Tensor, out_order_policy_id>;
    UnstructuredMesh<IsoPatchTriT> isosurface_tris(isopatch_coords_tri, out_order);
    UnstructuredMesh<IsoPatchQuadT> isosurface_quads(isopatch_coords_quad, out_order);
    DataSet isosurface_tri_ds(std::make_shared<UnstructuredMesh<IsoPatchTriT>>(isosurface_tris));
    DataSet isosurface_quad_ds(std::make_shared<UnstructuredMesh<IsoPatchQuadT>>(isosurface_quads));

    // Remap input fields onto surfaces.
    // Need to dispatch order policy for each input field.
    ReMapFieldFunctor<ShapeTri,  out_order_policy_id> rmff_tri(locset_tri, out_order_p);
    ReMapFieldFunctor<ShapeQuad, out_order_policy_id> rmff_quad(locset_quad, out_order_p);
    for (const std::string &fname : input_dataset->fields())
    {
      // TODO: we should probably map vectors
      dispatch_3d(input_dataset->field(fname), rmff_tri);
      dispatch_3d(input_dataset->field(fname), rmff_quad);

      isosurface_tri_ds.add_field(rmff_tri.m_out_field_ptr);
      isosurface_quad_ds.add_field(rmff_quad.m_out_field_ptr);
    }

    return {isosurface_tri_ds, isosurface_quad_ds};
  }


  /** remap_element() (Hex, Quad) */
  template <int32 IP, int32 MP, int32 OP, int32 ncomp>
  DRAY_EXEC void remap_element(const ShapeHex,
                               OrderPolicy<IP> in_order_p,
                               const ReadDofPtr<Vec<Float, ncomp>> &in_field_rdp,
                               const ShapeQuad,
                               OrderPolicy<MP> mesh_order_p,
                               const ReadDofPtr<Vec<Float, 3>> &mesh_rdp,
                               OrderPolicy<OP> out_order_p,
                               WriteDofPtr<Vec<Float, ncomp>> out_field_wdp)
  {
    const int32 ip = eattr::get_order(in_order_p);
    const int32 mp = eattr::get_order(mesh_order_p);
    const int32 op = eattr::get_order(out_order_p);

    //TODO use eval and don't assert equal.
    assert(mp == op);

    for (int32 j = 0; j <= op; ++j)
      for (int32 i = 0; i <= op; ++i)
      {
        Vec<Vec<Float, ncomp>, 3> UN_d = {{ {{0}}, {{0}}, {{0}} }};  // unused derivative.
        const Vec<Float, 3> host_ref_pt = mesh_rdp[j*(op+1) + i];  //TODO eval()
        const Vec<Float, ncomp> field_val =
            eops::eval_d(ShapeHex(), in_order_p, in_field_rdp, host_ref_pt, UN_d);

        out_field_wdp[j*(op+1) + i] = field_val;
      }
  }

  /** remap_element() (Hex, Tri) */
  template <int32 IP, int32 MP, int32 OP, int32 ncomp>
  DRAY_EXEC void remap_element(const ShapeHex,
                               OrderPolicy<IP> in_order_p,
                               const ReadDofPtr<Vec<Float, ncomp>> &in_field_rdp,
                               const ShapeTri,
                               OrderPolicy<MP> mesh_order_p,
                               const ReadDofPtr<Vec<Float, 3>> &mesh_rdp,
                               OrderPolicy<OP> out_order_p,
                               WriteDofPtr<Vec<Float, ncomp>> out_field_wdp)
  {
    const int32 ip = eattr::get_order(in_order_p);
    const int32 mp = eattr::get_order(mesh_order_p);
    const int32 op = eattr::get_order(out_order_p);

    //TODO use eval and don't assert equal.
    assert(mp == op);

    for (int32 j = 0; j <= op; ++j)
      for (int32 i = 0; i <= op-j; ++i)
      {
        const int32 nidx = detail::cartesian_to_tri_idx(i, j, op+1);

        Vec<Vec<Float, ncomp>, 3> UN_d = {{ {{0}}, {{0}}, {{0}} }};  // unused derivative.
        const Vec<Float, 3> host_ref_pt = mesh_rdp[nidx];  //TODO eval()
        const Vec<Float, ncomp> field_val =
            eops::eval_d(ShapeHex(), in_order_p, in_field_rdp, host_ref_pt, UN_d);

        out_field_wdp[nidx] = field_val;
      }
  }

  /** remap_element() (Tet, Quad) */
  template <int32 IP, int32 MP, int32 OP, int32 ncomp>
  DRAY_EXEC void remap_element(const ShapeTet,
                               OrderPolicy<IP> in_order_p,
                               const ReadDofPtr<Vec<Float, ncomp>> &in_field_rdp,
                               const ShapeQuad,
                               OrderPolicy<MP> mesh_order_p,
                               const ReadDofPtr<Vec<Float, 3>> &mesh_rdp,
                               OrderPolicy<OP> out_order_p,
                               WriteDofPtr<Vec<Float, ncomp>> out_field_wdp)
  {
        // TODO this is where we should evaluate instead of merely lookup.
    THROW_LOGIC_ERROR("Not implemented in " __FILE__ " remap_element<ShapeTet, ShapeQuad>")
  }

  /** remap_element() (Tet, Tri) */
  template <int32 IP, int32 MP, int32 OP, int32 ncomp>
  DRAY_EXEC void remap_element(const ShapeTet,
                               OrderPolicy<IP> in_order_p,
                               const ReadDofPtr<Vec<Float, ncomp>> &in_field_rdp,
                               const ShapeTri,
                               OrderPolicy<MP> mesh_order_p,
                               const ReadDofPtr<Vec<Float, 3>> &mesh_rdp,
                               OrderPolicy<OP> out_order_p,
                               WriteDofPtr<Vec<Float, ncomp>> out_field_wdp)
  {
        // TODO this is where we should evaluate instead of merely lookup.
    THROW_LOGIC_ERROR("Not implemented in " __FILE__ " remap_element<ShapeTet, ShapeTri>")
  }



  template <typename OutShape, int32 MP, class FElemT>
  std::shared_ptr<Field> ReMapField_execute(const LocationSet &location_set,
                                            OutShape,
                                            OrderPolicy<MP> _mesh_order_p,
                                            UnstructuredField<FElemT> &in_field)
  {
    // The output field type is based on the input field type,
    // but can also depend on the mesh order policy.

    const OrderPolicy<MP> mesh_order_p = _mesh_order_p;

    using InShape = typename AdaptGetShape<FElemT>::type;
    using InOrderPolicy = typename AdaptGetOrderPolicy<FElemT>::type;
    const InOrderPolicy in_order_p = adapt_get_order_policy(FElemT(), in_field.order());

    // TODO Evaluate (in Lagrange) on the surface to find out-field dof ref coords.
    // Then the output field type can match the input field order policy.

    ///using OutOrderPolicy = typename AdaptGetOrderPolicy<FElemT>::type;
    using OutOrderPolicy = OrderPolicy<MP>;

    ///const OutOrderPolicy out_order_p = adapt_get_order_policy(FElemT(), field.order());
    const OutOrderPolicy out_order_p = mesh_order_p;

    const int32 out_order = eattr::get_order(out_order_p);
    const int32 out_npe = eattr::get_num_dofs(OutShape(), out_order_p);

    constexpr int32 ncomp = FElemT::get_ncomp();

    using OutFElemT = Element<2,
                              ncomp,
                              eattr::get_etype(OutShape()),
                              eattr::get_policy_id(OutOrderPolicy())>;

    // Inputs.
    DeviceField<FElemT> device_in_field(in_field);
    DeviceGridFunctionConst<3> device_rcoords(location_set.m_rcoords);
    const int32 *host_cell_id_ptr = location_set.m_host_cell_id.get_device_ptr_const();

    const int32 num_out_elems = location_set.m_rcoords.get_num_elem();

    // Output.
    GridFunction<ncomp> out_field_gf;
    out_field_gf.resize_counting(num_out_elems, out_npe);
    DeviceGridFunction<ncomp> out_field_dgf(out_field_gf);

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_out_elems), [=] DRAY_LAMBDA (int32 oeid)
    {
        // workaournd for gcc 8.1 bug
        constexpr int32 ncomp_l = FElemT::get_ncomp();
        const int32 host_cell_id = host_cell_id_ptr[oeid];
        FElemT in_felem = device_in_field.get_elem(host_cell_id);
        const ReadDofPtr<Vec<Float, ncomp_l>> in_field_rdp = in_felem.read_dof_ptr();
        const ReadDofPtr<Vec<Float, 3>> mesh_rdp = device_rcoords.get_rdp(oeid);
        WriteDofPtr<Vec<Float, ncomp_l>> out_field_wdp = out_field_dgf.get_wdp(oeid);

        remap_element(InShape(),
                      in_order_p,
                      in_field_rdp,
                      OutShape(),
                      mesh_order_p,
                      mesh_rdp,
                      out_order_p,
                      out_field_wdp);
    });

    return std::make_shared<UnstructuredField<OutFElemT>>(out_field_gf, out_order, in_field.name());
  }



  // ExtractIsosurfaceFunctor
  struct ExtractIsosurfaceFunctor
  {
    Float m_iso_value;
    DataSet *m_input_dataset;

    DataSet m_output_tris;
    DataSet m_output_quads;

    ExtractIsosurfaceFunctor(Float iso_value, DataSet *input_dataset)
      : m_iso_value(iso_value),
        m_input_dataset(input_dataset)
    { }

    template <typename MeshType, typename FieldT>
    void operator()(MeshType &mesh, FieldT &field)
    {
      auto output = ExtractIsosurface_execute(mesh,
                                              field,
                                              m_iso_value,
                                              m_input_dataset);
      m_output_tris = output.first;
      m_output_quads = output.second;
    }
  };

  bool
  ExtractIsosurface::all_linear(Collection &collxn)
  {
    bool retval = true;
    for(DataSet ds : collxn.domains())
    {
      Mesh *mesh = ds.mesh();
      Field *field = ds.field(m_iso_field_name);
      if(mesh->order() != 1 || field->order() != 1)
      {
        retval = false;
        break;
      }
    }
    return retval;
  }

  // execute() wrapper
  std::pair<DataSet, DataSet> ExtractIsosurface::execute(DataSet &data_set)
  {
    // Extract isosurface mesh.
    ExtractIsosurfaceFunctor func(m_iso_value, &data_set);

    dispatch_3d_min_linear(data_set.mesh(),
                           data_set.field(m_iso_field_name),
                           func);

    // Return dataset.
    return {func.m_output_tris, func.m_output_quads};
  }

  std::pair<Collection, Collection> ExtractIsosurface::execute(Collection &collxn)
  {
    Collection out_collxn_first;
    Collection out_collxn_second;
    bool use_marching_cubes = all_linear(collxn);
    if(use_marching_cubes)
    {
      // Fast-path for linear mesh/field types.
      MarchingCubes filter;
      out_collxn_first = filter.execute(collxn);
    }
    else
    {
      for (DataSet ds : collxn.domains())
      {
        std::pair<DataSet, DataSet> ds_pair = this->execute(ds);
        out_collxn_first.add_domain(ds_pair.first);
        out_collxn_second.add_domain(ds_pair.second);
      }
    }
    return {out_collxn_first, out_collxn_second};
  }


}

#undef THROW_LOGIC_ERROR
