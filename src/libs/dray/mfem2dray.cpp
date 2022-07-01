// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/data_model/field.hpp>
#include <dray/data_model/mesh.hpp>
#include <dray/dray.hpp>
#include <dray/mfem2dray.hpp>
#include <dray/policies.hpp>
#include <dray/error.hpp>
#include <dray/error_check.hpp>
#include <dray/types.hpp>
#include <dray/utils/mfem_utils.hpp>
#include <dray/utils/data_logger.hpp>

#include <dray/data_model/unstructured_mesh.hpp>
#include <dray/data_model/unstructured_field.hpp>

#include <iostream>

namespace dray
{

namespace detail
{

// copied from mfem since they do not give us access
// https://github.com/mfem/mfem/blob/d9c4909da3766e49e66187ee063b8504e9d5138b/fem/fe.cpp#L8519
mfem::Array<int> tri_dof_map(int p)
{
  int dof_size = ((p + 1)*(p + 2))/2;
  struct Index
  {
     int p2p3;
     Index(int p) { p2p3 = 2*p + 3; }
     int operator()(int i, int j) { return ((p2p3-j)*j)/2+i; }
  };
  Index idx(p);

  mfem::Array<int> dof_map;
  dof_map.SetSize(dof_size);
  // vertices
  dof_map[idx(0,0)] = 0;
  //Nodes.IntPoint(0).Set2(0., 0.);
  dof_map[idx(p,0)] = 1;
  //Nodes.IntPoint(1).Set2(1., 0.);
  dof_map[idx(0,p)] = 2;
  //Nodes.IntPoint(2).Set2(0., 1.);

   // edges
  int o = 3;
  for (int i = 1; i < p; i++)
  {
     dof_map[idx(i,0)] = o;
     o++;
     //Nodes.IntPoint(o++).Set2(double(i)/p, 0.);
  }
  for (int i = 1; i < p; i++)
  {
     dof_map[idx(p-i,i)] = o;
     o++;
     //Nodes.IntPoint(o++).Set2(double(p-i)/p, double(i)/p);
  }
  for (int i = 1; i < p; i++)
  {
     dof_map[idx(0,p-i)] = o;
     o++;
     //Nodes.IntPoint(o++).Set2(0., double(p-i)/p);
  }

  // interior
  for (int j = 1; j < p; j++)
     for (int i = 1; i + j < p; i++)
     {
        dof_map[idx(i,j)] = o;
        o++;
        //Nodes.IntPoint(o++).Set2(double(i)/p, double(j)/p);
     }
  return dof_map;
}

mfem::Array<int> tet_dof_map(int p)
{
  int dof_size = ((p + 1)*(p + 2)*(p + 3))/6;

  struct Index
   {
      int p, dof;
      int tri(int k) { return (k*(k + 1))/2; }
      int tet(int k) { return (k*(k + 1)*(k + 2))/6; }
      Index(int p_) { p = p_; dof = tet(p + 1); }
      int operator()(int i, int j, int k)
      { return dof - tet(p - k) - tri(p + 1 - k - j) + i; }
   };

   Index idx(p);
   mfem::Array<int> dof_map;
   dof_map.SetSize(dof_size);
// vertices
   dof_map[idx(0,0,0)] = 0;
   //Nodes.IntPoint(0).Set3(0., 0., 0.);
   dof_map[idx(p,0,0)] = 1;
   //Nodes.IntPoint(1).Set3(1., 0., 0.);
   dof_map[idx(0,p,0)] = 2;
   //Nodes.IntPoint(2).Set3(0., 1., 0.);
   dof_map[idx(0,0,p)] = 3;
   //Nodes.IntPoint(3).Set3(0., 0., 1.);

   // edges (see Tetrahedron::edges in mesh/tetrahedron.cpp)
   int o = 4;
   for (int i = 1; i < p; i++)  // (0,1)
   {
      dof_map[idx(i,0,0)] = o;
      //Nodes.IntPoint(o++).Set3(double(i)/p, 0., 0.);
      o++;
   }
   for (int i = 1; i < p; i++)  // (0,2)
   {
      dof_map[idx(0,i,0)] = o;
      //Nodes.IntPoint(o++).Set3(0., double(i)/p, 0.);
      o++;
   }
   for (int i = 1; i < p; i++)  // (0,3)
   {
      dof_map[idx(0,0,i)] = o;
      //Nodes.IntPoint(o++).Set3(0., 0., double(i)/p);
      o++;
   }
   for (int i = 1; i < p; i++)  // (1,2)
   {
      dof_map[idx(p-i,i,0)] = o;
      //Nodes.IntPoint(o++).Set3(double(p-i)/p, double(i)/p, 0.);
      o++;
   }
   for (int i = 1; i < p; i++)  // (1,3)
   {
      dof_map[idx(p-i,0,i)] = o;
      //Nodes.IntPoint(o++).Set3(double(p-i)/p, 0., double(i)/p);
      o++;
   }
   for (int i = 1; i < p; i++)  // (2,3)
   {
      dof_map[idx(0,p-i,i)] = o;
      //Nodes.IntPoint(o++).Set3(0., double(p-i)/p, double(i)/p);
      o++;
   }

   // faces (see Mesh::GenerateFaces in mesh/mesh.cpp)
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (1,2,3)
      {
         dof_map[idx(p-i-j,i,j)] = o;
         //Nodes.IntPoint(o++).Set3(double(p-i-j)/p, double(i)/p, double(j)/p);
          o++;
      }
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (0,3,2)
      {
         dof_map[idx(0,j,i)] = o;
         //Nodes.IntPoint(o++).Set3(0., double(j)/p, double(i)/p);
         o++;
      }
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (0,1,3)
      {
         dof_map[idx(i,0,j)] = o;
         //Nodes.IntPoint(o++).Set3(double(i)/p, 0., double(j)/p);
         o++;
      }
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (0,2,1)
      {
         dof_map[idx(j,i,0)] = o;
         //Nodes.IntPoint(o++).Set3(double(j)/p, double(i)/p, 0.);
         o++;
      }

   // interior
   for (int k = 1; k < p; k++)
      for (int j = 1; j + k < p; j++)
         for (int i = 1; i + j + k < p; i++)
         {
            dof_map[idx(i,j,k)] = o;
            //Nodes.IntPoint(o++).Set3(double(i)/p, double(j)/p, double(k)/p);
            o++;
         }
   return dof_map;
}
// Assumes that values is allocated
template <int32 PhysDim, int32 RefDim>
void
import_dofs(const mfem::GridFunction &mfem_gf,
            Array<Vec<Float, PhysDim>> &values,
            int comp)
{

  // Access to degree of freedom mapping.
  const mfem::FiniteElementSpace *fespace = mfem_gf.FESpace ();

  // Access to control point data.
  const double * ctrl_vals = mfem_gf.HostRead();

  mfem::Array<int> zeroth_dof_set;
  fespace->GetElementDofs (0, zeroth_dof_set);

  const int32 vdim = fespace->GetVDim();
  const int32 dofs_per_element = zeroth_dof_set.Size ();
  const int32 num_elements = fespace->GetNE ();
  const int32 num_ctrls = mfem_gf.Size () / vdim;

  mfem::Table el_dof_table (fespace->GetElementToDofTable ());
  el_dof_table.Finalize ();
  const int32 all_el_dofs = el_dof_table.Size_of_connections ();
  if(all_el_dofs != num_elements * dofs_per_element)
  {
    DRAY_ERROR("Elements do not have the same number of dofs");
  }

  // if this is a vector or points in 2d,
  // we will convert this into 3d space with 0 for z;
  bool fill_z = false;
  if(RefDim == 2 && vdim > 1 && PhysDim == 3)
  {
    fill_z = true;
  }

  // Former attempt at the above assertion.
  //const int32 mfem_num_dofs = fespace->GetNDofs ();

  int32 stride_pdim;
  int32 stride_ctrl;
  if (fespace->GetOrdering () == mfem::Ordering::byNODES) // XXXX YYYY ZZZZ
  {
    DRAY_INFO("Ordering by nodes\n");
    // stride_pdim = num_elements;
    stride_pdim = num_ctrls;
    stride_ctrl = 1;
  }
  else // XYZ XYZ XYZ XYZ
  {
    DRAY_INFO("Ordering interleaved\n");
    stride_pdim = 1;
    stride_ctrl = vdim;
  }

  //
  // Import degree of freedom values.
  //
  Vec<Float, PhysDim> *ctrl_val_ptr = values.get_host_ptr();
  // import all components
  if(comp == -1)
  {
    /// RAJA::forall<for_cpu_policy>(RAJA::RangeSegment(0, num_ctrls), [=] (int32 ctrl_id)
    for (int32 ctrl_id = 0; ctrl_id < num_ctrls; ctrl_id++)
    {
      // TODO get internal representation of the mfem memory, so we can access in a device function.
      //
      for (int32 pdim = 0; pdim < PhysDim; pdim++)
      {
        if(fill_z && pdim == 2)
        {
          ctrl_val_ptr[ctrl_id][pdim] = Float(0.f);
        }
        else
        {
          const int index = pdim * stride_pdim + ctrl_id * stride_ctrl;
          ctrl_val_ptr[ctrl_id][pdim] = ctrl_vals[index];
        }
      }
    }
    ///});
    DRAY_ERROR_CHECK();
  }
  else
  {
    if(comp >= vdim)
    {
      DRAY_ERROR("vector dim is greater then requested component");
    }
    //import only a single component
    for (int32 ctrl_id = 0; ctrl_id < num_ctrls; ctrl_id++)
    {
      const int index = comp * stride_pdim + ctrl_id * stride_ctrl;
      ctrl_val_ptr[ctrl_id][0] = ctrl_vals[index];
    }
  }
}

// Assumes that values is allocated
template <int32 PhysDim, int32 RefDim>
void
import_indices(const mfem::GridFunction &mfem_gf,
               Array<int32> &indexs,
               mfem::Geometry::Type geom_type)
{
  // Access to degree of freedom mapping.
  const mfem::FiniteElementSpace *fespace = mfem_gf.FESpace ();

  mfem::Array<int> zeroth_dof_set;
  fespace->GetElementDofs (0, zeroth_dof_set);

  const int32 P = fespace->GetOrder (0);
  const int32 num_elements = fespace->GetNE ();
  const int32 dofs_per_element = zeroth_dof_set.Size ();

  // DRAY and MFEM may store degrees of freedom in different orderings.
  bool use_dof_map = fespace->Conforming ();

  mfem::Array<int> fe_dof_map;
  // figure out what kinds of elements these are
  std::string elem_type(fespace->FEColl()->Name());

  if(elem_type.find("H1Pos") != std::string::npos)
  {
    if(RefDim == 3)
    {
      if(geom_type == mfem::Geometry::Type::CUBE)
      {
        mfem::H1Pos_HexahedronElement h1_prototype (P);
        fe_dof_map = h1_prototype.GetDofMap();
      }
      else if(geom_type == mfem::Geometry::Type::TETRAHEDRON)
      {
        // simplex prototypes don't have dof maps
        // for some reason
        //mfem::H1Pos_TetrahedronElement h1_prototype (P);
        fe_dof_map = detail::tet_dof_map(P);
      }
      else
      {
        DRAY_ERROR("this should not happen");
      }

    }
    else
    {
      if(geom_type == mfem::Geometry::Type::SQUARE)
      {
        mfem::H1Pos_QuadrilateralElement h1_prototype (P);
        fe_dof_map = h1_prototype.GetDofMap();
      }
      else if(geom_type == mfem::Geometry::Type::TRIANGLE)
      {
        // simplex prototypes don't have dof maps
        // for some reason
        //mfem::H1Pos_TriangleElement h1_prototype (P);
        fe_dof_map = detail::tri_dof_map(P);;
      }
      else
      {
        DRAY_ERROR("this should not happen");
      }
    }
  }
  else
  {
    // The L2 prototype does not return anything, because
    // the ording is implicit. Like somehow I was just supposed
    // to know that and should have expected an empty array.
    // Going to make the assumption that this is just a linear ordering.
    //mfem::L2Pos_HexahedronElement l2_prototype(P);
    use_dof_map = false;
  }

  int32 *ctrl_idx_ptr = indexs.get_host_ptr ();
  for (int32 el_id = 0; el_id < num_elements; el_id++)
  {
    // TODO get internal representation of the mfem memory, so we can access in a device function.
    //
    mfem::Array<int> el_dof_set;
    fespace->GetElementDofs (el_id, el_dof_set);

    for (int32 dof_id = el_id * dofs_per_element, el_dof_id = 0;
         el_dof_id < dofs_per_element; dof_id++, el_dof_id++)
    {
      // Maintain same lexicographic order as MFEM (X-inner:Z-outer).
      const int32 el_dof_id_lex = el_dof_id;
      // Maybe there's a better practice than this inner conditional.
      const int32 mfem_el_dof_id = use_dof_map ? fe_dof_map[el_dof_id_lex] : el_dof_id_lex;
      ctrl_idx_ptr[dof_id] = el_dof_set[mfem_el_dof_id];
    }
  }
}

} // namespace detail



template <int32 PhysDim, int32 RefDim>
GridFunction<PhysDim>
import_grid_function2(const mfem::GridFunction &_mfem_gf,
                      int32 &space_P,
                      mfem::Geometry::Type geom_type,
                      int comp = -1)
{
  DRAY_LOG_OPEN("import_grid_function");
  Timer timer;
  bool is_gf_new;
  mfem::GridFunction *pos_gf = project_to_pos_basis (&_mfem_gf, is_gf_new);
  const mfem::GridFunction &mfem_gf = (is_gf_new ? *pos_gf : _mfem_gf);
  DRAY_LOG_ENTRY("project", timer.elapsed());

  constexpr int32 phys_dim = PhysDim;
  GridFunction<phys_dim> grid_func;

  // Access to degree of freedom mapping.
  const mfem::FiniteElementSpace *fespace = mfem_gf.FESpace ();

  const int32 P = fespace->GetOrder (0);

  mfem::Array<int> zeroth_dof_set;
  fespace->GetElementDofs (0, zeroth_dof_set);

  const int32 vdim = fespace->GetVDim();
  const int32 dofs_per_element = zeroth_dof_set.Size ();
  const int32 num_elements = fespace->GetNE ();
  const int32 num_ctrls = mfem_gf.Size () / vdim;

  grid_func.resize (num_elements, dofs_per_element, num_ctrls);

  timer.reset();
  detail::import_dofs<PhysDim,RefDim>(mfem_gf, grid_func.m_values, comp);
  DRAY_LOG_ENTRY("dofs", timer.elapsed());
  timer.reset();
  detail::import_indices<PhysDim,RefDim>(mfem_gf, grid_func.m_ctrl_idx, geom_type);
  DRAY_LOG_ENTRY("indices", timer.elapsed());

  if (is_gf_new)
  {
    delete pos_gf;
  }

  space_P = P;
  DRAY_LOG_CLOSE();
  return grid_func;
}

void import_field(DataSet &dataset,
                  const mfem::GridFunction &grid_function,
                  const mfem::Geometry::Type geom_type,
                  const std::string field_name,
                  const int32 comp) // single componet of vector (-1 all)
{
  if(geom_type != mfem::Geometry::CUBE &&
     geom_type != mfem::Geometry::SQUARE &&
     geom_type != mfem::Geometry::TRIANGLE &&
     geom_type != mfem::Geometry::TETRAHEDRON
     )
  {
    DRAY_ERROR("Only hex, quad, tet, and tri imports implemented");
  }

  int ref_dim = 3;
  if(geom_type == mfem::Geometry::SQUARE ||
     geom_type == mfem::Geometry::TRIANGLE)
  {
    ref_dim = 2;
  }

  if(ref_dim == 3)
  {
    if(geom_type == mfem::Geometry::CUBE)
    {
      int order;
      GridFunction<1> field_data
        = import_grid_function2<1,3> (grid_function, order, geom_type, comp);
      if (dray::prefer_native_order_field())
      {
        if (order == 0)
        {
          UnstructuredField<HexScalar_P0> field (field_data, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<HexScalar_P0>>(field));
        }
        else if (order == 1)
        {
          UnstructuredField<HexScalar_P1> field (field_data, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<HexScalar_P1>>(field));
        }
        else if (order == 2)
        {
          UnstructuredField<HexScalar_P2> field (field_data, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<HexScalar_P2>>(field));
        }
        else
        {
          std::stringstream msg;
          msg << "Can't obey policy use_fixed_field_order==true because field order ("
              << order << ") is too high. "
              << "Falling back on Order::General implementation.";
          DRAY_WARN(msg.str());

          UnstructuredField<HexScalar> field (field_data, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<HexScalar>>(field));
        }
      }
      else
      {
        UnstructuredField<HexScalar> field (field_data, order, field_name);
        dataset.add_field(std::make_shared<UnstructuredField<HexScalar>>(field));
      }
    }
    else if(geom_type == mfem::Geometry::TETRAHEDRON)
    {
      int order;
      GridFunction<1> field_data
        = import_grid_function2<1,3> (grid_function, order, geom_type, comp);
      if (dray::prefer_native_order_field())
      {
        if (order == 0)
        {
          UnstructuredField<TetScalar_P0> field (field_data, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<TetScalar_P0>>(field));
        }
        else if (order == 1)
        {
          UnstructuredField<TetScalar_P1> field (field_data, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<TetScalar_P1>>(field));
        }
        else if (order == 2)
        {
          UnstructuredField<TetScalar_P2> field (field_data, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<TetScalar_P2>>(field));
        }
        else
        {
          std::stringstream msg;
          msg << "Can't obey policy use_fixed_field_order==true because field order ("
              << order << ") is too high. "
              << "Falling back on Order::General implementation.";
          DRAY_WARN(msg.str());

          UnstructuredField<TetScalar> field (field_data, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<TetScalar>>(field));
        }
      }
      else
      {
        UnstructuredField<TetScalar> field (field_data, order, field_name);
        dataset.add_field(std::make_shared<UnstructuredField<TetScalar>>(field));
      }
    }
    else
    {
      DRAY_ERROR("This should not happen");
    }
  }
  else
  {
    if(geom_type == mfem::Geometry::SQUARE)
    {
      int order;
      GridFunction<1> field_data
        = import_grid_function2<1,2> (grid_function, order, geom_type, comp);
      if (dray::prefer_native_order_field())
      {
        if (order == 0)
        {
          UnstructuredField<QuadScalar_P0> field (field_data, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<QuadScalar_P0>>(field));
        }
        else if (order == 1)
        {
          UnstructuredField<QuadScalar_P1> field (field_data, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<QuadScalar_P1>>(field));
        }
        else if (order == 2)
        {
          UnstructuredField<QuadScalar_P2> field (field_data, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<QuadScalar_P2>>(field));
        }
        else
        {
          std::stringstream msg;
          msg << "Can't obey policy use_fixed_field_order==true because field order ("
              << order << ") is too high. "
              << "Falling back on Order::General implementation.";
          DRAY_WARN(msg.str());

          UnstructuredField<QuadScalar> field (field_data, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<QuadScalar>>(field));
        }
      }
      else
      {
        UnstructuredField<QuadScalar> field (field_data, order, field_name);
        dataset.add_field(std::make_shared<UnstructuredField<QuadScalar>>(field));
      }
    }
    else if(geom_type == mfem::Geometry::TRIANGLE)
    {
      int order;
      GridFunction<1> field_data
        = import_grid_function2<1,2> (grid_function, order, geom_type, comp);
      if (dray::prefer_native_order_field())
      {
        if (order == 0)
        {
          UnstructuredField<TriScalar_P0> field (field_data, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<TriScalar_P0>>(field));
        }
        else if (order == 1)
        {
          UnstructuredField<TriScalar_P1> field (field_data, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<TriScalar_P1>>(field));
        }
        else if (order == 2)
        {
          UnstructuredField<TriScalar_P2> field (field_data, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<TriScalar_P2>>(field));
        }
        else
        {
          std::stringstream msg;
          msg << "Can't obey policy use_fixed_field_order==true because field order ("
              << order << ") is too high. "
              << "Falling back on Order::General implementation.";
          DRAY_WARN(msg.str());

          UnstructuredField<TriScalar> field (field_data, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<TriScalar>>(field));
        }
      }
      else
      {
        UnstructuredField<TriScalar> field (field_data, order, field_name);
        dataset.add_field(std::make_shared<UnstructuredField<TriScalar>>(field));
      }
    }
    else
    {
      DRAY_ERROR("This should not happen");
    }
  }
}

void import_vector(DataSet &dataset,
                   const mfem::GridFunction &grid_function,
                   const mfem::Geometry::Type geom_type,
                   const std::string field_name)
{
  const int comp = -1; // import all three components
  if(geom_type != mfem::Geometry::CUBE &&
     geom_type != mfem::Geometry::SQUARE &&
     geom_type != mfem::Geometry::TRIANGLE &&
     geom_type != mfem::Geometry::TETRAHEDRON
     )
  {
    DRAY_ERROR("Only hex, quad, tet, and tri imports implemented");
  }

  int ref_dim = 3;
  if(geom_type == mfem::Geometry::SQUARE ||
     geom_type == mfem::Geometry::TRIANGLE)
  {
    ref_dim = 2;
  }

  if(ref_dim == 3)
  {
    if(geom_type == mfem::Geometry::CUBE)
    {
      int order;
      GridFunction<3> field_data
        = import_grid_function2<3,3> (grid_function, order, geom_type, comp);
      if (dray::prefer_native_order_field())
      {
        if (order == 0)
        {
          UnstructuredField<HexVector_P0> field (field_data, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<HexVector_P0>>(field));
        }
        else if (order == 1)
        {
          UnstructuredField<HexVector_P1> field (field_data, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<HexVector_P1>>(field));
        }
        else if (order == 2)
        {
          UnstructuredField<HexVector_P2> field (field_data, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<HexVector_P2>>(field));
        }
        else
        {
          std::stringstream msg;
          msg << "Can't obey policy use_fixed_field_order==true because field order ("
              << order << ") is too high. "
              << "Falling back on Order::General implementation.";
          DRAY_WARN(msg.str());

          UnstructuredField<HexVector> field (field_data, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<HexVector>>(field));
        }
      }
      else
      {
        UnstructuredField<HexVector> field (field_data, order, field_name);
        dataset.add_field(std::make_shared<UnstructuredField<HexVector>>(field));
      }
    }
    else if(geom_type == mfem::Geometry::TETRAHEDRON)
    {
      int order;
      GridFunction<3> field_data
        = import_grid_function2<3,3> (grid_function, order, geom_type, comp);
      if (dray::prefer_native_order_field())
      {
        if (order == 0)
        {
          UnstructuredField<TetVector_P0> field (field_data, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<TetVector_P0>>(field));
        }
        else if (order == 1)
        {
          UnstructuredField<TetVector_P1> field (field_data, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<TetVector_P1>>(field));
        }
        else if (order == 2)
        {
          UnstructuredField<TetVector_P2> field (field_data, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<TetVector_P2>>(field));
        }
        else
        {
          std::stringstream msg;
          msg << "Can't obey policy use_fixed_field_order==true because field order ("
              << order << ") is too high. "
              << "Falling back on Order::General implementation.";
          DRAY_WARN(msg.str());

          UnstructuredField<TetVector> field (field_data, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<TetVector>>(field));
        }
      }
      else
      {
        UnstructuredField<TetVector> field (field_data, order, field_name);
        dataset.add_field(std::make_shared<UnstructuredField<TetVector>>(field));
      }
    }
    else
    {
      DRAY_ERROR("This should not happen");
    }
  }
  else
  {
    if(geom_type == mfem::Geometry::SQUARE)
    {
      int order;
      GridFunction<3> field_data
        = import_grid_function2<3,2> (grid_function, order, geom_type, comp);
      if (dray::prefer_native_order_field())
      {
        if (order == 0)
        {
          UnstructuredField<QuadVector_P0> field (field_data, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<QuadVector_P0>>(field));
        }
        else if (order == 1)
        {
          UnstructuredField<QuadVector_P1> field (field_data, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<QuadVector_P1>>(field));
        }
        else if (order == 2)
        {
          UnstructuredField<QuadVector_P2> field (field_data, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<QuadVector_P2>>(field));
        }
        else
        {
          std::stringstream msg;
          msg << "Can't obey policy use_fixed_field_order==true because field order ("
              << order << ") is too high. "
              << "Falling back on Order::General implementation.";
          DRAY_WARN(msg.str());

          UnstructuredField<QuadVector> field (field_data, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<QuadVector>>(field));
        }
      }
      else
      {
        UnstructuredField<QuadVector> field (field_data, order, field_name);
        dataset.add_field(std::make_shared<UnstructuredField<QuadVector>>(field));
      }
    }
    else if(geom_type == mfem::Geometry::TRIANGLE)
    {
      int order;
      GridFunction<3> field_data
        = import_grid_function2<3,2> (grid_function, order, geom_type, comp);
      if (dray::prefer_native_order_field())
      {
        if (order == 0)
        {
          UnstructuredField<TriVector_P0> field (field_data, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<TriVector_P0>>(field));
        }
        else if (order == 1)
        {
          UnstructuredField<TriVector_P1> field (field_data, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<TriVector_P1>>(field));
        }
        else if (order == 2)
        {
          UnstructuredField<TriVector_P2> field (field_data, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<TriVector_P2>>(field));
        }
        else
        {
          std::stringstream msg;
          msg << "Can't obey policy use_fixed_field_order==true because field order ("
              << order << ") is too high. "
              << "Falling back on Order::General implementation.";
          DRAY_WARN(msg.str());

          UnstructuredField<TriVector> field (field_data, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<TriVector>>(field));
        }
      }
      else
      {
        UnstructuredField<TriVector> field (field_data, order, field_name);
        dataset.add_field(std::make_shared<UnstructuredField<TriVector>>(field));
      }
    }
    else
    {
      DRAY_ERROR("This should not happen");
    }
  }
}

DataSet import_mesh(const mfem::Mesh &mesh)
{
  DRAY_LOG_OPEN("import_mesh");
  mfem::Geometry::Type geom_type = mesh.GetElementBaseGeometry(0);

  if(geom_type != mfem::Geometry::CUBE &&
     geom_type != mfem::Geometry::SQUARE &&
     geom_type != mfem::Geometry::TRIANGLE &&
     geom_type != mfem::Geometry::TETRAHEDRON
     )
  {
    DRAY_ERROR("Only hex, quad, tri, and tets imports implemented");
  }

  int ref_dim = 3;
  if(geom_type == mfem::Geometry::SQUARE ||
     geom_type == mfem::Geometry::TRIANGLE)
  {
    ref_dim = 2;
  }

  mesh.GetNodes();

  if (mesh.Conforming())
  {
    DRAY_INFO("Conforming mesh");
  }
  else
  {
    DRAY_INFO("Non-Conforming mesh");
  }

  const mfem::GridFunction *nodes = mesh.GetNodes();;

  DataSet res;
  if (nodes != NULL)
  {
    if(ref_dim == 3)
    {
      if(geom_type == mfem::Geometry::CUBE)
      {
        int order;
        GridFunction<3> gf = import_grid_function2<3,3> (*nodes, order, geom_type);
        if (dray::prefer_native_order_mesh())
        {
          if (order == 1)
          {
            HexMesh_P1 mesh (gf, order);
            res = DataSet(std::make_shared<HexMesh_P1>(mesh));
          }
          else if (order == 2)
          {
            HexMesh_P2 mesh (gf, order);
            res = DataSet(std::make_shared<HexMesh_P2>(mesh));
          }
          else
          {
            std::stringstream msg;
            msg << "Can't obey policy use_fixed_mesh_order==true because mesh order ("
                << order << ") is too high. "
                << "Falling back on Order::General implementation.";
            DRAY_WARN(msg.str());

            HexMesh mesh (gf, order);
            res = DataSet(std::make_shared<HexMesh>(mesh));
          }
        }
        else
        {
          HexMesh mesh (gf, order);
          res = DataSet(std::make_shared<HexMesh>(mesh));
        }
      }
      else if(geom_type == mfem::Geometry::TETRAHEDRON)
      {
        int order;
        GridFunction<3> gf = import_grid_function2<3,3> (*nodes, order, geom_type);
        if (dray::prefer_native_order_field())
        {
          if (order == 1)
          {
            TetMesh_P1 mesh (gf, order);
            res = DataSet(std::make_shared<TetMesh_P1>(mesh));
          }
          else if (order == 2)
          {
            TetMesh_P2 mesh (gf, order);
            res = DataSet(std::make_shared<TetMesh_P2>(mesh));
          }
          else
          {
            std::stringstream msg;
            msg << "Can't obey policy use_fixed_mesh_order==true because mesh order ("
                << order << ") is too high. "
                << "Falling back on Order::General implementation.";
            DRAY_WARN(msg.str());

            using Tet = MeshElem<3u, Simplex, General>;
            UnstructuredMesh<Tet> mesh (gf, order);

            res = DataSet(std::make_shared<TetMesh>(mesh));
          }
        }
        else
        {
          TetMesh mesh (gf, order);
          DataSet dataset(std::make_shared<TetMesh>(mesh));
          res = dataset;
        }
      }
      else
      {
        DRAY_ERROR("this should not happen");
      }
    }
    else
    {
      if(geom_type == mfem::Geometry::SQUARE)
      {
        int order;
        GridFunction<3> gf = import_grid_function2<3,2> (*nodes, order, geom_type);
        if (dray::prefer_native_order_field())
        {
          if (order == 1)
          {
            QuadMesh_P1 mesh (gf, order);
            res = DataSet(std::make_shared<QuadMesh_P1>(mesh));
          }
          else if (order == 2)
          {
            QuadMesh_P2 mesh (gf, order);
            res = DataSet(std::make_shared<QuadMesh_P2>(mesh));
          }
          else
          {
            std::stringstream msg;
            msg << "Can't obey policy use_fixed_mesh_order==true because mesh order ("
                << order << ") is too high. "
                << "Falling back on Order::General implementation.";
            DRAY_WARN(msg.str());

            QuadMesh mesh (gf, order);
            res = DataSet(std::make_shared<QuadMesh>(mesh));
          }
        }
        else
        {
          QuadMesh mesh (gf, order);
          res = DataSet(std::make_shared<QuadMesh>(mesh));
        }
      }
      else if(geom_type == mfem::Geometry::TRIANGLE)
      {
        int order;
        GridFunction<3> gf = import_grid_function2<3,2> (*nodes, order, geom_type);
        TriMesh mesh (gf, order);
        if (dray::prefer_native_order_field())
        {
          if (order == 1)
          {
            TriMesh_P1 mesh (gf, order);
            res = DataSet(std::make_shared<TriMesh_P1>(mesh));
          }
          else if (order == 2)
          {
            TriMesh_P2 mesh (gf, order);
            res = DataSet(std::make_shared<TriMesh_P2>(mesh));
          }
          else
          {
            std::stringstream msg;
            msg << "Can't obey policy use_fixed_mesh_order==true because mesh order ("
                << order << ") is too high. "
                << "Falling back on Order::General implementation.";
            DRAY_WARN(msg.str());

            TriMesh mesh (gf, order);
            res = DataSet(std::make_shared<TriMesh>(mesh));
          }
        }
        else
        {
          TriMesh mesh (gf, order);
          res = DataSet(std::make_shared<TriMesh>(mesh));
        }
      }
      else
      {
        DRAY_ERROR("this should not happen");
      }
    }
  }
  else
  {
    DRAY_ERROR("Importing linear mesh not implemented");
    //space_P = 1;
    //return import_linear_mesh (mfem_mesh);
  }

  DRAY_LOG_CLOSE();
  return res;
}


void print_geom(mfem::Geometry::Type type)
{
  if(type == mfem::Geometry::POINT)
  {
    std::cout<<"point\n";
  }
  else if(type == mfem::Geometry::SEGMENT)
  {
    std::cout<<"segment\n";
  }
  else if(type == mfem::Geometry::TRIANGLE)
  {
    std::cout<<"triangle\n";
  }
  else if(type == mfem::Geometry::TETRAHEDRON)
  {
    std::cout<<"tet\n";
  }
  else if(type == mfem::Geometry::SQUARE)
  {
    std::cout<<"quad\n";
  }
  else if(type == mfem::Geometry::CUBE)
  {
    std::cout<<"hex\n";
  }
  else if(type == mfem::Geometry::PRISM)
  {
    std::cout<<"prism. no thanks\n";
  }
  else
  {
    std::cout<<"unknown\n";
  }
}

//
// project_to_pos_basis()
//
// If is_new was set to true, the caller is responsible for deleting the returned pointer.
// If is_new was set to false, then the returned value is null, and the caller should use gf.
mfem::GridFunction *project_to_pos_basis (const mfem::GridFunction *gf, bool &is_new)
{
  mfem::GridFunction *out_pos_gf = nullptr;
  is_new = false;

  /// bool is_high_order =
  ///    (gf != nullptr) && (mesh->GetNE() > 0);
  /// if(!is_high_order) std::cout<<"NOT High Order\n";

  // Sanity checks
  /// assert(is_high_order);
  assert (gf != nullptr);

  /// Generate (or access existing) positive (Bernstein) nodal grid function
  const mfem::FiniteElementSpace *nodal_fe_space = gf->FESpace ();
  if (nodal_fe_space == nullptr)
  {
    DRAY_ERROR("project_to_pos_basis(): nodal_fe_space is NULL!");
  }

  const mfem::FiniteElementCollection *nodal_fe_coll = nodal_fe_space->FEColl ();
  if (nodal_fe_coll == nullptr)
  {
    DRAY_ERROR("project_to_pos_basis(): nodal_fe_coll is NULL!");
  }

  // Check if grid function is positive, if not create positive grid function
  if (detail::is_positive_basis (nodal_fe_coll))
  {
    // std::cerr<<"Already positive.\n";
    is_new = false;
    out_pos_gf = nullptr;
  }
  else
  {
    // std::cerr<<"Attemping to convert to positive basis.\n";
    // Assume that all elements of the mesh have the same order and geom type
    mfem::Mesh *gf_mesh = nodal_fe_space->GetMesh ();
    if (gf_mesh == nullptr)
    {
      DRAY_ERROR("project_to_pos_basis(): gf_mesh is NULL!");
    }

    int order = nodal_fe_space->GetOrder (0);
    int dim = gf_mesh->Dimension ();
    mfem::Geometry::Type geom_type = gf_mesh->GetElementBaseGeometry (0);
    int map_type = (nodal_fe_coll != nullptr) ?
                   nodal_fe_coll->FiniteElementForGeometry (geom_type)->GetMapType () :
                   static_cast<int> (mfem::FiniteElement::VALUE);

    mfem::FiniteElementCollection *pos_fe_coll =
    detail::get_pos_fec (nodal_fe_coll, order, dim, map_type);

    if (pos_fe_coll != nullptr)
    {
      const int dims = nodal_fe_space->GetVDim ();
      // Create a positive (Bernstein) grid function for the nodes
      mfem::FiniteElementSpace *pos_fe_space =
      new mfem::FiniteElementSpace (gf_mesh, pos_fe_coll, dims);
      mfem::GridFunction *pos_nodes = new mfem::GridFunction (pos_fe_space);

      // m_pos_nodes takes ownership of pos_fe_coll's memory (and pos_fe_space's memory)
      pos_nodes->MakeOwner (pos_fe_coll);

      // Project the nodal grid function onto this
      pos_nodes->ProjectGridFunction (*gf);

      out_pos_gf = pos_nodes;
      is_new = true;
    }
    // DEBUG
    else
    {
      DRAY_ERROR("BAD... pos_fe_coll is NULL. Could not make FESpace or GridFunction.");
    }
    // DEBUG
    if (!out_pos_gf)
    {
      DRAY_ERROR("project_to_pos_basis(): Construction failed;  out_pos_gf is NULL!");
    }
  }

  return out_pos_gf;
}


} // namespace dray
