// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

#include <dray/data_model/unstructured_mesh.hpp>
#include <dray/data_model/unstructured_field.hpp>
#include <dray/filters/extract_slice.hpp>
#include <dray/filters/extract_three_slice.hpp>
#include <dray/io/blueprint_low_order.hpp>
#include <dray/io/blueprint_reader.hpp>
#include <dray/io/mfem_reader.hpp>
#include <dray/synthetic/tet_sphere_sample.hpp>
#include <dray/dispatcher.hpp>
//#include <dray/data_model/elem_ops.hpp>

#include <conduit/conduit.hpp>
#include <conduit/conduit_blueprint.hpp>

//-----------------------------------------------------------------------------
static void
dray_collection_to_blueprint(dray::Collection &c, conduit::Node &n)
{
  int i = 0;
  for(auto it = c.domains().begin();
      it != c.domains().end(); it++, i++)
  {
      std::stringstream s;
      s << "domain" << i;
      conduit::Node dnode;
      try
      {
          it->to_node(dnode);
          // Now, take the dray conduit node and convert to blueprint so
          // we can actually look at it in VisIt.
          std::string path(s.str());
          conduit::Node &bnode = n[path];
          dray::BlueprintLowOrder::to_blueprint(dnode, bnode);
      }
      catch(std::exception &e)
      {
          std::cerr << "EXCEPTION:" << e.what() << std::endl;
      }
  }
}

enum class TestCase {
  Tet,
  Hex
};

static conduit::Node
make_simple_field(int constant, int nx, int ny, int nz, int nc=1)
{
  std::vector<std::vector<float>> data;
  data.resize(nc);

  for(int i = 0; i < nz; i++)
  {
    for(int j = 0; j < ny*nx; j++)
    {
      for(int c = 0; c < nc; c++)
      {
        if(c % 2 == 0)
        {
          data[c].push_back(float(constant * (i)));
        }
        else
        {
          data[c].push_back(-float(constant * (i)));
        }
      }
    }
  }

#ifndef NDEBUG
  std::cout << "input:";
  for(int i = 0; i < data[0].size(); i++)
  {
    std::cout << " (";
    for(int c = 0; c < nc; c++)
    {
      std::cout << data[c][i] << (c < nc-1 ? "," : ")");
    }
  }
  std::cout << std::endl;
#endif

  conduit::Node n_field;
  if(nc == 1)
  {
    // Scalar
    n_field["association"].set("vertex");
    n_field["type"].set("scalar");
    n_field["topology"].set("mesh");
    n_field["values"].set(data[0]);
  }
  else
  {
    // Vector
    n_field["association"].set("vertex");
    n_field["type"].set("vector");
    n_field["topology"].set("mesh");
    for(int c = 0; c < nc; c++)
    {
      conduit::Node &n = n_field["values"].append();
      n.set(data[c]);
    }
  }
  return n_field;
}

static conduit::Node
make_simple_cell_field(int constant, int ncells, int nc=1)
{
  std::vector<std::vector<float>> data;
  data.resize(nc);

  for(int i = 0; i < ncells; i++)
  {
    for(int c = 0; c < nc; c++)
    {
      if(c % 2 == 0)
      {
        data[c].push_back(float(constant * (i)));
      }
      else
      {
        data[c].push_back(-float(constant * (i)));
      }
    }
  }

#ifndef NDEBUG
  std::cout << "input:";
  for(int i = 0; i < data[0].size(); i++)
  {
    std::cout << " (";
    for(int c = 0; c < nc; c++)
    {
      std::cout << data[c][i] << (c < nc-1 ? "," : ")");
    }
  }
  std::cout << std::endl;
#endif

  conduit::Node n_field;
  if(nc == 1)
  {
    // Scalar
    n_field["association"].set("element");
    n_field["type"].set("scalar");
    n_field["topology"].set("mesh");
    n_field["values"].set(data[0]);
  }
  else
  {
    // Vector
    n_field["association"].set("element");
    n_field["type"].set("vector");
    n_field["topology"].set("mesh");
    for(int c = 0; c < nc; c++)
    {
      conduit::Node &n = n_field["values"].append();
      n.set(data[c]);
    }
  }
  return n_field;
}

static dray::Collection
make_test_mesh(TestCase tc)
{
  using conduit::blueprint::mesh::examples::braid;
  const int DIM = 3;
  const bool is3D = tc == TestCase::Hex || tc == TestCase::Tet;
  int ncells = 0;
  conduit::Node n_collection;
  switch(tc)
  {
    case TestCase::Tet:
    {
      braid("tets",
            DIM,
            DIM,
            DIM,
            n_collection.add_child("domain0"));
      ncells = (DIM-1)*(DIM-1)*(DIM-1)*6;
      break;
    }
    case TestCase::Hex:
    {
      braid("hexs",
            DIM,
            DIM,
            DIM,
            n_collection.add_child("domain0"));
      ncells = (DIM-1)*(DIM-1)*(DIM-1);
      break;
    }
  }
  n_collection["domain0"].remove("fields");
  // For point fields, just pass 1 for nx if we are 2D. This will make the field increase with Y.
  n_collection["domain0/fields/simple"] = make_simple_field(2, is3D ? DIM : 1, DIM, DIM);
  n_collection["domain0/fields/simple_vec"] = make_simple_field(2, is3D ? DIM : 1, DIM, DIM, is3D ? 3 : 2);
  // Iso filter is returning everything as point fields
#if 0
  n_collection["domain0/fields/simple_cell"] = make_simple_cell_field(1, ncells);
  n_collection["domain0/fields/simple_cell_vec"] = make_simple_cell_field(1, ncells, is3D ? 3 : 2);
#endif

  // Convert it to a dray collection
  dray::DataSet domain0 = dray::BlueprintLowOrder::import(n_collection[0]);
  dray::Collection collection;
  collection.add_domain(domain0);
  return collection;
}

template<typename MeshElemType>
static void
test_mesh(dray::DataSet dset, const int axis, const float ans, const int ans_size)
{
  using namespace dray;
  auto *mesh = dynamic_cast<UnstructuredMesh<MeshElemType>*>(dset.mesh());
  ASSERT_TRUE(mesh);
  auto grid_func = mesh->get_dof_data();
  ASSERT_EQ(ans_size, grid_func.m_values.size());
  Vec<Float, MeshElemType::get_ncomp()> *data = mesh->get_dof_data().m_values.get_host_ptr();
  for(int i = 0; i < ans_size; i++)
  {
    EXPECT_FLOAT_EQ(ans, data[i][axis]);
  }
}

template<typename FieldElemType>
static void
test_result(dray::DataSet dset, const std::string &name, const float *ans, const int ans_size)
{
  using namespace dray;
  auto *field = dynamic_cast<UnstructuredField<FieldElemType>*>(dset.field(name));
  ASSERT_TRUE(field);
  auto grid_func = field->get_dof_data();
  ASSERT_EQ(ans_size, grid_func.m_values.size());
  Vec<Float, FieldElemType::get_ncomp()> *data = field->get_dof_data().m_values.get_host_ptr();
  for(int i = 0; i < ans_size; i++)
  {
    for(int c = 0; c < FieldElemType::get_ncomp(); c++)
    {
      EXPECT_FLOAT_EQ(c % 2 == 0 ? ans[i] : -ans[i], data[i][c]);
    }
  }
}

#if 0
TEST(dray_extract_slice, tets)
{
  using namespace dray;
  Collection collection = make_test_mesh(TestCase::Tet);
  conduit::Node n_tmp;
  collection.domain(0).to_node(n_tmp);
  n_tmp.print();

  ExtractSlice slicer;
  slicer.add_plane({-5., -5., -5.}, {0., 0., 1.});
  auto tris_quads = slicer.execute(collection);
  Collection tris = tris_quads.first;
  Collection quads = tris_quads.second;

  conduit::Node n_tris, n_quads;
  dray_collection_to_blueprint(tris, n_tris);
  dray_collection_to_blueprint(quads, n_quads);
  dray::BlueprintReader::save_blueprint("output_tet_tris", n_tris);
  dray::BlueprintReader::save_blueprint("output_tet_quads", n_quads);
}

TEST(dray_extract_slice, hexes)
{
  using namespace dray;
  Collection collection = make_test_mesh(TestCase::Hex);
  // conduit::Node n_tmp;
  // collection.domain(0).to_node(n_tmp);
  // n_tmp.print();

  // Test that we can make one slice
  ExtractSlice slicer;
  slicer.add_plane({-5., -5., -5.}, {0., 0., 1.});
  // Should only have quads
  Collection quads = slicer.execute(collection).second;
  conduit::Node n_quads;
  dray_collection_to_blueprint(quads, n_quads);
  dray::BlueprintReader::save_blueprint("output_hex_quads", n_quads);
  // n_quads.print();

  using MeshType = Element<2, 3, ElemType::Tensor, Order::Linear>;
  using ScalarFieldType = Element<2, 1, ElemType::Tensor, Order::Linear>;
  using VecFieldType = Element<2, 3, ElemType::Tensor, Order::Linear>;
  // There should only be 9 points but currently some points get
  // double / quadruple counted when included by multiple quads
  const float ans_z[16] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
                           1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
  test_mesh<MeshType>(quads.domain(0), 2, -5.f, 16);
  test_result<ScalarFieldType>(quads.domain(0), "simple", ans_z, 16);
  test_result<VecFieldType>(quads.domain(0), "simple_vec", ans_z, 16);

  // Test that we can make multiple slices, also test ExtractThreeSlice makes the same answer
  slicer.clear();
  slicer.add_plane({-5., -5., -5.}, {1., 0., 0.});
  slicer.add_plane({-5., -5., -5.}, {0., 1., 0.});
  slicer.add_plane({-5., -5., -5.}, {0., 0., 1.});
  ExtractThreeSlice three_slicer;
  three_slicer.set_point({-5., -5., -5.});
  quads = slicer.execute(collection).second;
  Collection quads_three = three_slicer.execute(collection).second;

  const float ans_x[16] = {0.f, 2.f, 0.f, 2.f, 0.f, 2.f, 0.f, 2.f,
                           2.f, 4.f, 2.f, 4.f, 2.f, 4.f, 2.f, 4.f};
  test_mesh<MeshType>(quads.domain(0), 0, -5.f, 16);
  test_result<ScalarFieldType>(quads.domain(0), "simple", ans_x, 16);
  test_result<ScalarFieldType>(quads_three.domain(0), "simple", ans_x, 16);
  test_result<VecFieldType>(quads.domain(0), "simple_vec", ans_x, 16);
  test_result<VecFieldType>(quads_three.domain(0), "simple_vec", ans_x, 16);

  const float ans_y[16] = {0.f, 0.f, 2.f, 2.f, 0.f, 0.f, 2.f, 2.f,
                           2.f, 2.f, 4.f, 4.f, 2.f, 2.f, 4.f, 4.f};
  test_mesh<MeshType>(quads.domain(1), 1, -5.f, 16);
  test_result<ScalarFieldType>(quads.domain(1), "simple", ans_y, 16);
  test_result<ScalarFieldType>(quads_three.domain(1), "simple", ans_y, 16);
  test_result<VecFieldType>(quads.domain(1), "simple_vec", ans_y, 16);
  test_result<VecFieldType>(quads_three.domain(1), "simple_vec", ans_y, 16);

  // Same answer as before
  test_mesh<MeshType>(quads.domain(2), 2, -5.f, 16);
  test_result<ScalarFieldType>(quads.domain(2), "simple", ans_z, 16);
  test_result<ScalarFieldType>(quads_three.domain(2), "simple", ans_z, 16);
  test_result<VecFieldType>(quads.domain(2), "simple_vec", ans_z, 16);
  test_result<VecFieldType>(quads_three.domain(2), "simple_vec", ans_z, 16);
}

TEST(dray_extract_slice, single_hex)
{
  using namespace dray;
  DataSet domain;
  {
    GridFunction<3> mesh_gf;
    mesh_gf.resize(1, 8, 8);
    int32 *conn = mesh_gf.m_ctrl_idx.get_host_ptr();
    for(int i = 0; i < 8; i++)
    {
      conn[i] = i;
    }
    Vec<Float, 3> *points = mesh_gf.m_values.get_host_ptr();
    points[0] = {0, 0, 0};
    points[1] = {1, 0, 0};
    points[2] = {0, 1, 0};
    points[3] = {1, 1, 0};
    points[4] = {0, 0, 1};
    points[5] = {1, 0, 1};
    points[6] = {0, 1, 1};
    points[7] = {1, 1, 1};

    using MeshElemType = Element<3, 3, ElemType::Tensor, Order::Linear>;
    std::shared_ptr<Mesh> mesh = std::make_shared<UnstructuredMesh<MeshElemType>>(mesh_gf, 1);
    domain.add_mesh(mesh);

    GridFunction<1> braid_gf;
    braid_gf.m_el_dofs = 8;
    braid_gf.m_size_el = 1;
    braid_gf.m_size_ctrl = 8;
    braid_gf.m_ctrl_idx = mesh_gf.m_ctrl_idx;
    braid_gf.m_values.resize(8);
    Vec<Float, 1> *braid = braid_gf.m_values.get_host_ptr();
    braid[0] = { 0.999936f};
    braid[1] = {-0.947823f};
    braid[2] = { 1.7136f};
    braid[3] = {-0.571105f};
    braid[4] = { 0.999795f};
    braid[5] = {-0.948017f};
    braid[6] = { 1.71352f};
    braid[7] = {-0.571308f};
    using FieldElemType = Element<3, 1, ElemType::Tensor, Order::Linear>;
    std::shared_ptr<Field> field = std::make_shared<UnstructuredField<FieldElemType>>(braid_gf, 1, "braid");
    domain.add_field(field);
  }

  Collection collection;
  collection.add_domain(domain);

  conduit::Node n_input;
  dray_collection_to_blueprint(collection, n_input);
  dray::BlueprintReader::save_blueprint("input_single_hex", n_input);

  // Extract the isosurface
  dray::Vec<dray::Float, 3> origin{0.5, 0.5, 0.5};
  dray::Vec<dray::Float, 3> normal{1., 0., 0.};
  dray::ExtractSlice slice;
  slice.add_plane(origin, normal);
  normal[0] = 0.; normal[1] = 1.; normal[2] = 0.;
  slice.add_plane(origin, normal);
  normal[0] = 0.; normal[1] = 0.; normal[2] = 1.;
  slice.add_plane(origin, normal);
  auto tris_quads = slice.execute(collection);
  conduit::Node n_tris;
  conduit::Node n_quads;
  dray_collection_to_blueprint(tris_quads.first, n_tris);
  dray_collection_to_blueprint(tris_quads.second, n_quads);
  dray::BlueprintReader::save_blueprint("output_single_hex_tris", n_tris);
  dray::BlueprintReader::save_blueprint("output_single_hex_quads", n_quads);
}
#endif

using namespace dray;

template<typename METype>
static void
read_connectivity(UnstructuredMesh<METype> &mesh)
{
  GridFunction<3> gf = mesh.get_dof_data();
  DeviceGridFunction<3> dgf(gf);
  auto dofs = dgf.get_rdp(0);
  std::cout << "dofs per elem: " << gf.m_el_dofs << "\n"
            << "mesh type: " << mesh.type_name() << std::endl;
  const auto nelem = gf.m_size_el;
  const auto ndofs = gf.m_el_dofs;
  int idx = 0;
  for(int e = 0; e < 1; ++e)
  {
    std::cout << e << ": ";
    for(int dof = 0; dof < ndofs; ++dof, ++idx)
    {
      std::cout << "(" << dofs[idx][0] << "," << dofs[idx][1]
                << "," << dofs[idx][2] << "), ";
    }
    std::cout << "\n";
  }
  std::cout << std::endl;
}

template <class RotatedIndexT>
static void
print_edge(const RotatedIndexT &wheel, int edge, int p)
{
  std::cout << "Edge " << edge << ":";
  for (int i = 0; i <= p; ++i)
  {
    std::cout << " " << wheel.linearize(i);
  }
  std::cout << std::endl;
}

TEST(dray_extract_slice, tets)
{
  using dray::detail::cartesian_to_tet_idx;
  for(int p = 1; p < 6; p++)
  {
    std::cout << "Cartesian to tet_p" << p << ":"
              << "\n(0,0,0): " << cartesian_to_tet_idx(0, 0, 0, p)
              << "\n(p,0,0): " << cartesian_to_tet_idx(p, 0, 0, p)
              << "\n(0,p,0): " << cartesian_to_tet_idx(0, p, 0, p)
              << "\n(0,0,p): " << cartesian_to_tet_idx(0, 0, p, p)
              << "\n(p,p,0): " << cartesian_to_tet_idx(p, p, 0, p)
              << "\n(p,0,p): " << cartesian_to_tet_idx(p, 0, p, p)
              << "\n(0,p,p): " << cartesian_to_tet_idx(0, p, p, p)
              << "\n(p,p,p): " << cartesian_to_tet_idx(p, p, p, p)
              << std::endl;
  }

  const int p = 5;
  const eops::TetFlat hlin{p};
  print_edge(eops::RotatedIdx3<0,1,2, eops::TetFlat>(0,0,0, hlin), 0, p);
  print_edge(eops::RotatedIdx3<1,0,2, eops::TetFlat>(p,0,0, hlin), 1, p);
  print_edge(eops::RotatedIdx3<1,0,2, eops::TetFlat>(0,0,0, hlin), 2, p);

  Collection c = make_test_mesh(TestCase::Tet);
  DataSet ds = c.domain(0);
  auto lambda = [](auto mesh) {
    read_connectivity(mesh);
  };
  dispatch(ds.mesh(), lambda);

  ExtractSlice slicer;
  slicer.add_plane({-0.5, -0.5, -0.5}, {0, 0, 1});
  auto output = slicer.execute(c);
}