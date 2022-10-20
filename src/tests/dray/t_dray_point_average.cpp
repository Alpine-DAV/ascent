#include "gtest/gtest.h"
#include "t_utils.hpp"

#include <conduit/conduit.hpp>
#include <dray/data_model/collection.hpp>
#include <dray/data_model/field.hpp>
#include <dray/data_model/unstructured_field.hpp>
#include <dray/data_model/unstructured_mesh.hpp>
#include <dray/filters/point_average.hpp>
#include <dray/io/blueprint_low_order.hpp>
#include <dray/io/blueprint_reader.hpp>
#include <dray/dray_node_to_dataset.hpp>

#include <iostream>
#include <vector>

static conduit::Node
make_simple_field(int constant, int nelem, int nc=1)
{
  std::vector<std::vector<float>> data;
  data.resize(nc);

  for(int i = 0; i < nelem; i++)
  {
    for(int c = 0; c < nc; c++)
    {
      if(c % 2 == 0)
      {
        data[c].push_back(float(constant * i));
      }
      else
      {
        data[c].push_back(-float(constant * i));
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

template<typename MeshElemType, typename FieldElemType>
static void
test_result(const float *ans, const int ans_size, dray::Collection &result,
            const std::string &field_name, const int nelem_dofs, const int nelem)
{
  using MeshType = dray::UnstructuredMesh<MeshElemType>;
  using FieldType = dray::UnstructuredField<FieldElemType>;
  constexpr auto ncomp = FieldElemType::get_ncomp();
  dray::DataSet result_dset = result.domain(0);
  ASSERT_TRUE(result_dset.has_field(field_name));
  dray::Field *field = result_dset.field(field_name);
#ifndef NDEBUG
  std::cout << "Field: " << field->type_name() << std::endl;
#endif
  FieldType *typed_field = dynamic_cast<FieldType*>(field);
  ASSERT_TRUE(typed_field);
  auto grid_func = typed_field->get_dof_data();
  ASSERT_EQ(grid_func.m_el_dofs, nelem_dofs);
  ASSERT_EQ(grid_func.m_size_el, nelem);
  ASSERT_EQ(ans_size, grid_func.m_values.size());
  dray::Array<dray::Vec<dray::Float, ncomp>> &values = grid_func.m_values;
  for(int i = 0; i < ans_size; i++)
  {
    for(int c = 0; c < ncomp; c++)
    {
      if(c % 2 == 0)
      {
        EXPECT_FLOAT_EQ(ans[i], values.get_value(i)[c])
          << "i=" << i << ", c=" << c;
      }
      else
      {
        EXPECT_FLOAT_EQ(-ans[i], values.get_value(i)[c])
          << "i=" << i << ", c=" << c;
      }
    }
  }

#ifndef NDEBUG
  std::cout << "Mesh: " << result_dset.mesh()->type_name() << std::endl;
#endif
  MeshType *typed_mesh = dynamic_cast<MeshType*>(result_dset.mesh());
  ASSERT_TRUE(typed_mesh);
  auto mesh_grid_func = typed_mesh->get_dof_data();
  ASSERT_EQ(mesh_grid_func.m_ctrl_idx.size(), grid_func.m_ctrl_idx.size());
  for(int i = 0; i < mesh_grid_func.m_ctrl_idx.size(); i++)
  {
    EXPECT_EQ(mesh_grid_func.m_ctrl_idx.get_value(i), grid_func.m_ctrl_idx.get_value(i))
      << "i=" << i;
  }
}

TEST(dray_point_average, set_output_field)
{
  // NOTE: Similar to quad test but an output name is set
  using MeshElemType = dray::Element<2, 3, dray::ElemType::Tensor, dray::Order::Linear>;
  const int DIM = 3;
  conduit::Node n_input;
  conduit::blueprint::mesh::examples::braid("quads",
                                             DIM,
                                             DIM,
                                             0,
                                             n_input.add_child("domain0"));
  n_input[0]["fields"]["simple"].set(make_simple_field(4, (DIM-1)*(DIM-1)));

  // Convert it to a dray collection
  dray::DataSet input_domain = dray::BlueprintLowOrder::import(n_input[0]);
  dray::Collection input;
  input.add_domain(input_domain);

  // Grab handles to the original field / grid function
  using OrigElemType = dray::Element<2, 1, dray::ElemType::Tensor, dray::Order::Constant>;
  ASSERT_TRUE(input_domain.has_field("simple"));
  dray::Field *simple_in_field = input_domain.field("simple");
  using OrigFieldType = dray::UnstructuredField<OrigElemType>;
  OrigFieldType *typed_in_simple = dynamic_cast<OrigFieldType*>(simple_in_field);
  ASSERT_TRUE(typed_in_simple);

  // *---*---*
  // | 8 | 12|
  // *---*---*
  // | 0 | 4 |
  // *---*---*
  const float ans[9] = {0.f, 2.f, 4.f, 4.f, 6.f, 8.f, 8.f, 10.f, 12.f};

  // Execute the filter over the collection, verify the result is stored in "simple_point_average"
  dray::PointAverage filter;
  filter.set_field("simple");
  filter.set_output_field("simple_point_average");
  dray::Collection result = filter.execute(input);
  using ElemTypeScalar = dray::Element<2, 1, dray::ElemType::Tensor, dray::Order::Linear>;
  test_result<MeshElemType, ElemTypeScalar>(ans, 9, result, "simple_point_average", 4, 4);

  // Ensure the output collection still has the original, unaltered "simple" field.
  dray::DataSet out_dset = result.domain(0);
  ASSERT_TRUE(out_dset.has_field("simple"));
  dray::Field *simple_out_field = out_dset.field("simple");
  OrigFieldType *typed_out_simple = dynamic_cast<OrigFieldType*>(simple_out_field);
  ASSERT_TRUE(typed_out_simple);
  EXPECT_EQ(typed_in_simple, typed_out_simple);
  const float *orig_data = (float*)n_input[0]["fields/simple/values"].element_ptr(0);
  auto in_gf = typed_in_simple->get_dof_data();
  auto &in_data = in_gf.m_values;
  ASSERT_EQ(1, in_gf.m_el_dofs);
  ASSERT_EQ(4, in_gf.m_size_el);
  ASSERT_EQ(4, in_data.size());
  for(int i = 0; i < 4; i++)
  {
    EXPECT_EQ(orig_data[i], in_data.get_value(i)[0])
      << "i=" << i;
  }

  // Quick check to ensure setting output_field to an empty string
  // restores the default behavior
  filter.set_output_field("");
  result = filter.execute(input);
  test_result<MeshElemType, ElemTypeScalar>(ans, 9, result, "simple", 4, 4);
}

TEST(dray_point_average, tris)
{

  using MeshElemType = dray::Element<2, 3, dray::ElemType::Simplex, dray::Order::Linear>;
  const int DIM = 3;
  conduit::Node n_input;
  conduit::blueprint::mesh::examples::braid("tris",
                                             DIM,
                                             DIM,
                                             0,
                                             n_input.add_child("domain0"));
  n_input[0]["fields"]["simple"].set(make_simple_field(6, 8));
#ifdef DISPATCH_2D_VECTOR_FIELDS
  n_input[0]["fields"]["simple_vec"].set(make_simple_field(6, 8, 2));
#endif

  // Convert it to a dray collection
  dray::DataSet input_domain = dray::BlueprintLowOrder::import(n_input[0]);
  dray::Collection input;
  input.add_domain(input_domain);

  // *-----*-----*
  // |24/30|36/42|
  // *-----*-----*
  // |0 / 6|12/18|
  // *-----*-----*
  const float ans[9] = {3.f, 12.f, 18.f, 18.f, 21.f, 24.f, 24.f, 30.f, 39.f};
  dray::PointAverage filter;
  filter.set_field("simple");
  dray::Collection result = filter.execute(input);
  using ElemTypeScalar = dray::Element<2, 1, dray::ElemType::Simplex, dray::Order::Linear>;
  test_result<MeshElemType, ElemTypeScalar>(ans, 9, result, "simple", 3, 8);

#ifdef DISPATCH_2D_VECTOR_FIELDS
  filter.set_field("simple_vec");
  result = filter.execute(input);
  using ElemTypeVector = dray::Element<2, 2, dray::ElemType::Simplex, dray::Order::Linear>;
  test_result<MeshElemType, ElemTypeVector>(ans, 9, result, "simple_vec", 3, 8);
#endif
}

TEST(dray_point_average, quads)
{

  using MeshElemType = dray::Element<2, 3, dray::ElemType::Tensor, dray::Order::Linear>;
  const int DIM = 3;
  conduit::Node n_input;
  conduit::blueprint::mesh::examples::braid("quads",
                                             DIM,
                                             DIM,
                                             0,
                                             n_input.add_child("domain0"));
  n_input[0]["fields"]["simple"].set(make_simple_field(4, 4));
  n_input[0]["fields"]["simple_vec"].set(make_simple_field(4, 4, 2));

  // Convert it to a dray collection
  dray::DataSet input_domain = dray::BlueprintLowOrder::import(n_input[0]);
  dray::Collection input;
  input.add_domain(input_domain);

  // *-----*-----*
  // |  8  |  12 |
  // *-----*-----*
  // |  0  |  4  |
  // *-----*-----*
  const float ans[9] = {0.f, 2.f, 4.f, 4.f, 6.f, 8.f, 8.f, 10.f, 12.f};
  dray::PointAverage filter;
  filter.set_field("simple");
  dray::Collection result = filter.execute(input);
  using ElemTypeScalar = dray::Element<2, 1, dray::ElemType::Tensor, dray::Order::Linear>;
  test_result<MeshElemType, ElemTypeScalar>(ans, 9, result, "simple", 4, 4);

  filter.set_field("simple_vec");
  result = filter.execute(input);
  using ElemTypeVector = dray::Element<2, 2, dray::ElemType::Tensor, dray::Order::Linear>;
  test_result<MeshElemType, ElemTypeVector>(ans, 9, result, "simple_vec", 4, 4);
}

TEST(dray_point_average, tets)
{

  using MeshElemType = dray::Element<3, 3, dray::ElemType::Simplex, dray::Order::Linear>;
  const int DIM = 2;
  conduit::Node n_input;
  conduit::blueprint::mesh::examples::braid("tets",
                                             DIM,
                                             DIM,
                                             DIM,
                                             n_input.add_child("domain0"));
  n_input[0]["fields"]["simple"].set(make_simple_field(6, 6));
  n_input[0]["fields"]["simple_vec"].set(make_simple_field(6, 6, 3));

  // Convert it to a dray collection
  dray::DataSet input_domain = dray::BlueprintLowOrder::import(n_input[0]);
  dray::Collection input;
  input.add_domain(input_domain);

  // This example is just one hex broken into 6 tets
  //   ID      CONN    Value
  // Elem 0 = 0 3 1 7 = 0
  // Elem 1 = 0 2 3 7 = 6
  // Elem 2 = 0 6 2 7 = 12
  // Elem 3 = 0 4 6 7 = 18
  // Elem 4 = 0 5 4 7 = 24
  // Elem 5 = 0 1 5 7 = 30
  //  Point Averages
  // Pt0 = Pt7 = (0 + 6 + 12 + 18 + 24 + 28) / 6 = 15
  // Pt1 = (0 + 30) / 2 = 15
  // Pt2 = (6 + 12) / 2 = 9
  // Pt3 = (0 + 6)  / 2 = 3
  // Pt4 = (18 + 24)/ 2 = 21
  // Pt5 = (24 + 30)/ 2 = 27
  // Pt6 = (12 + 18)/ 2 = 15
  const float ans[8] = {15.f, 15.f, 9.f, 3.f, 21.f, 27.f, 15.f, 15.f};
  dray::PointAverage filter;
  filter.set_field("simple");
  dray::Collection result = filter.execute(input);
  using ElemTypeScalar = dray::Element<3, 1, dray::ElemType::Simplex, dray::Order::Linear>;
  test_result<MeshElemType, ElemTypeScalar>(ans, 8, result, "simple", 4, 6);

  filter.set_field("simple_vec");
  result = filter.execute(input);
  using ElemTypeVector = dray::Element<3, 3, dray::ElemType::Simplex, dray::Order::Linear>;
  test_result<MeshElemType, ElemTypeVector>(ans, 8, result, "simple_vec", 4, 6);
}

TEST(dray_point_average, hexes)
{

  using MeshElemType = dray::Element<3, 3, dray::ElemType::Tensor, dray::Order::Linear>;
  const int DIM = 3;
  conduit::Node n_input;
  conduit::blueprint::mesh::examples::braid("hexs",
                                             DIM,
                                             DIM,
                                             3,
                                             n_input.add_child("domain0"));
  n_input[0]["fields"]["simple"].set(make_simple_field(8, 8));
  n_input[0]["fields"]["simple_vec"].set(make_simple_field(8, 8, 3));

  // Convert it to a dray collection
  dray::DataSet input_domain = dray::BlueprintLowOrder::import(n_input[0]);
  dray::Collection input;
  input.add_domain(input_domain);

  // z=0 cells
  // *-----*-----*
  // |  16 |  24 |
  // *-----*-----*
  // |  0  |  8  |
  // *-----*-----*
  // z=1 cells
  // *-----*-----*
  // |  48 |  56 |
  // *-----*-----*
  // |  32 |  40 |
  // *-----*-----*
  const float ans[27] = { 0.f,  4.f,  8.f,  8.f, 12.f, 16.f, 16.f, 20.f, 24.f,
                         16.f, 20.f, 24.f, 24.f, 28.f, 32.f, 32.f, 36.f, 40.f,
                         32.f, 36.f, 40.f, 40.f, 44.f, 48.f, 48.f, 52.f, 56.f};
  dray::PointAverage filter;
  filter.set_field("simple");
  dray::Collection result = filter.execute(input);
  using ElemTypeScalar = dray::Element<3, 1, dray::ElemType::Tensor, dray::Order::Linear>;
  test_result<MeshElemType, ElemTypeScalar>(ans, 27, result, "simple", 8, 8);

  filter.set_field("simple_vec");
  result = filter.execute(input);
  using ElemTypeVector = dray::Element<3, 3, dray::ElemType::Tensor, dray::Order::Linear>;
  test_result<MeshElemType, ElemTypeVector>(ans, 27, result, "simple_vec", 8, 8);
}