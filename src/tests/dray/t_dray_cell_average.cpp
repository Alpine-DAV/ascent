// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"
#include "t_utils.hpp"

#include <conduit/conduit.hpp>
#include <dray/data_model/collection.hpp>
#include <dray/data_model/field.hpp>
#include <dray/data_model/unstructured_field.hpp>
#include <dray/filters/cell_average.hpp>
#include <dray/io/blueprint_low_order.hpp>
#include <dray/io/blueprint_reader.hpp>
#include <dray/dray_node_to_dataset.hpp>

#include <iostream>
#include <vector>

const int EXAMPLE_MESH_SIDE_DIM = 3;

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
          data[c].push_back(float(constant * (i+1)));
        }
        else
        {
          data[c].push_back(-float(constant * (i+1)));
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

template<typename ElemType>
static void
test_result(const float *ans, const int ans_size, dray::Collection &result,
            const std::string &field_name)
{
  using FieldType = dray::UnstructuredField<ElemType>;
  constexpr auto ncomp = ElemType::get_ncomp();
  dray::DataSet result_dset = result.domain(0);
  ASSERT_TRUE(result_dset.has_field(field_name));
  dray::Field *field = result_dset.field(field_name);
#ifndef NDEBUG
  std::cout << "Field: " << field->type_name() << std::endl;
#endif
  FieldType *typed_field = dynamic_cast<FieldType*>(field);
  ASSERT_TRUE(typed_field);
  auto grid_func = typed_field->get_dof_data();
  ASSERT_EQ(grid_func.m_el_dofs, 1);
  ASSERT_EQ(grid_func.m_size_el, ans_size);
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
  ASSERT_EQ(grid_func.m_size_ctrl, ans_size);
  dray::Array<dray::int32> &ctrl = grid_func.m_ctrl_idx;
  for(int i = 0; i < ans_size; i++)
  {
    EXPECT_EQ(i, ctrl.get_value(i));
  }
}

TEST(dray_cell_average, set_output_field)
{
  // NOTE: Similar to quad test but an output name is set
  conduit::Node n_input;
  conduit::blueprint::mesh::examples::braid("quads",
                                             EXAMPLE_MESH_SIDE_DIM,
                                             EXAMPLE_MESH_SIDE_DIM,
                                             0,
                                             n_input.add_child("domain0"));
  n_input[0]["fields"]["simple"].set(make_simple_field(4, 1, EXAMPLE_MESH_SIDE_DIM, EXAMPLE_MESH_SIDE_DIM));

  // Convert it to a dray collection
  dray::DataSet input_domain = dray::BlueprintLowOrder::import(n_input[0]);
  dray::Collection input;
  input.add_domain(input_domain);

  // Grab handles to the original field / grid function
  using OrigElemType = dray::Element<2, 1, dray::ElemType::Tensor, dray::Order::Linear>;
  ASSERT_TRUE(input_domain.has_field("simple"));
  dray::Field *simple_in_field = input_domain.field("simple");
  using OrigFieldType = dray::UnstructuredField<OrigElemType>;
  OrigFieldType *typed_in_simple = dynamic_cast<OrigFieldType*>(simple_in_field);
  ASSERT_TRUE(typed_in_simple);

  // Stash a copy of the original connectivity
  const int orig_conn_size = 16;
  std::vector<int> orig_conn(orig_conn_size);
  for(int i = 0; i < orig_conn_size; i++)
  {
    orig_conn[i] = typed_in_simple->get_dof_data().m_ctrl_idx.get_value(i);
  }

  // *---*---*  <-- y=2, point values = 12
  // | 2 | 3 |
  // *---*---*  <-- y=1, point values = 8
  // | 0 | 1 |
  // *---*---*  <-- y=0, point values = 4
  //
  // Elem 0 = Elem 1 = (4 + 4 + 8 + 8) / 4 = 6
  // Elem 2 = Elem 3 = (8 + 8 + 12 + 12) / 4 = 10
  const float ans[4] = {6.f, 6.f, 10.f, 10.f};

  // Execute the filter over the collection, verify the result is stored in "simple_cell_average"
  dray::CellAverage filter;
  filter.set_field("simple");
  filter.set_output_field("simple_cell_average");
  dray::Collection result = filter.execute(input);
  using ElemTypeScalar = dray::Element<2, 1, dray::ElemType::Tensor, dray::Order::Constant>;
  test_result<ElemTypeScalar>(ans, 4, result, "simple_cell_average");

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
  ASSERT_EQ(4, in_gf.m_el_dofs);
  ASSERT_EQ(4, in_gf.m_size_el);
  ASSERT_EQ(9, in_data.size());
  for(int i = 0; i < 9; i++)
  {
    EXPECT_EQ(orig_data[i], in_data.get_value(i)[0])
      << "i=" << i;
  }
  // Ensure the connectivity is unchanged
  ASSERT_EQ(orig_conn_size, in_gf.m_size_ctrl);
  for(int i = 0; i < 16; i++)
  {
    EXPECT_EQ(orig_conn[i], in_gf.m_ctrl_idx.get_value(i));
  }

  // Quick check to ensure setting output_field to an empty string
  // restores the default behavior
  filter.set_output_field("");
  result = filter.execute(input);
  test_result<ElemTypeScalar>(ans, 4, result, "simple");
}

TEST(dray_cell_average, tris)
{
  // Load the example mesh
  conduit::Node n_input;
  conduit::blueprint::mesh::examples::braid("tris",
                                             EXAMPLE_MESH_SIDE_DIM,
                                             EXAMPLE_MESH_SIDE_DIM,
                                             0,
                                             n_input.add_child("domain0"));
  n_input[0]["fields"]["simple"].set(make_simple_field(3, 1, EXAMPLE_MESH_SIDE_DIM, EXAMPLE_MESH_SIDE_DIM));
  n_input[0]["fields"]["simple_vec"].set(make_simple_field(3, 1, EXAMPLE_MESH_SIDE_DIM, EXAMPLE_MESH_SIDE_DIM, 2));

  // Convert it to a dray collection
  dray::DataSet input_domain = dray::BlueprintLowOrder::import(n_input[0]);
  dray::Collection input;
  input.add_domain(input_domain);
  // n_input[0].save("tris_input.yaml");

  // *---*---*  <-- y=2, point values = 9
  // |4/5|6/7|
  // *---*---*  <-- y=1, point values = 6
  // |0/1|2/3|
  // *---*---*  <-- y=0, point values = 3
  //
  // Elem 0 = Elem 2 = (3 + 3 + 6) / 3 = 5
  // Elem 1 = Elem 3 = (4 + 6 + 6) / 3 = 4
  // Elem 4 = Elem 6 = (6 + 6 + 9) / 3 = 8
  // Elem 5 = Elem 7 = (6 + 6 + 9) / 3 = 7
  const float ans[8] = {5.f, 4.f, 5.f, 4.f, 8.f, 7.f, 8.f, 7.f};

  // Execute the filter over the collection
  dray::CellAverage filter;
  filter.set_field("simple");
  dray::Collection result = filter.execute(input);
  using ElemTypeScalar = dray::Element<2, 1, dray::ElemType::Simplex, dray::Order::Constant>;
  test_result<ElemTypeScalar>(ans, 8, result, "simple");

  // Now check the vector field
  filter.set_field("simple_vec");
  result = filter.execute(input);
  using ElemTypeVec = dray::Element<2, 2, dray::ElemType::Simplex, dray::Order::Constant>;
  test_result<ElemTypeVec>(ans, 8, result, "simple_vec");
}

TEST(dray_cell_average, quads)
{
  // Load the example mesh
  conduit::Node n_input;
  conduit::blueprint::mesh::examples::braid("quads",
                                             EXAMPLE_MESH_SIDE_DIM,
                                             EXAMPLE_MESH_SIDE_DIM,
                                             0,
                                             n_input.add_child("domain0"));
  n_input[0]["fields"]["simple"].set(make_simple_field(4, 1, EXAMPLE_MESH_SIDE_DIM, EXAMPLE_MESH_SIDE_DIM));
  n_input[0]["fields"]["simple_vec"].set(make_simple_field(4, 1, EXAMPLE_MESH_SIDE_DIM, EXAMPLE_MESH_SIDE_DIM, 2));

  // Convert it to a dray collection
  // n_input[0].save("quads_input.yaml");
  dray::DataSet input_domain = dray::BlueprintLowOrder::import(n_input[0]);
  dray::Collection input;
  input.add_domain(input_domain);

  // *---*---*  <-- y=2, point values = 12
  // | 2 | 3 |
  // *---*---*  <-- y=1, point values = 8
  // | 0 | 1 |
  // *---*---*  <-- y=0, point values = 4
  //
  // Elem 0 = Elem 1 = (4 + 4 + 8 + 8) / 4 = 6
  // Elem 2 = Elem 3 = (8 + 8 + 12 + 12) / 4 = 10
  const float ans[4] = {6.f, 6.f, 10.f, 10.f};

  // Execute the filter over the collection
  dray::CellAverage filter;
  filter.set_field("simple");
  dray::Collection result = filter.execute(input);
  using ElemTypeScalar = dray::Element<2, 1, dray::ElemType::Tensor, dray::Order::Constant>;
  test_result<ElemTypeScalar>(ans, 4, result, "simple");

  // Now check the vector field
  filter.set_field("simple_vec");
  result = filter.execute(input);
  using ElemTypeVec = dray::Element<2, 2, dray::ElemType::Tensor, dray::Order::Constant>;
  test_result<ElemTypeVec>(ans, 4, result, "simple_vec");
}

TEST(dray_cell_average, tets)
{
  // Load the example mesh
  conduit::Node n_input;
  conduit::blueprint::mesh::examples::braid("tets",
                                             EXAMPLE_MESH_SIDE_DIM,
                                             EXAMPLE_MESH_SIDE_DIM,
                                             EXAMPLE_MESH_SIDE_DIM,
                                             n_input.add_child("domain0"));
  n_input[0]["fields"]["simple"].set(make_simple_field(4, EXAMPLE_MESH_SIDE_DIM, EXAMPLE_MESH_SIDE_DIM, EXAMPLE_MESH_SIDE_DIM));
  n_input[0]["fields"]["simple_vec"].set(make_simple_field(4, EXAMPLE_MESH_SIDE_DIM, EXAMPLE_MESH_SIDE_DIM, EXAMPLE_MESH_SIDE_DIM, 3));

  // Convert it to a dray collection
  n_input[0].save("tets_input.yaml");
  dray::DataSet input_domain = dray::BlueprintLowOrder::import(n_input[0]);
  dray::Collection input;
  input.add_domain(input_domain);

  // For each hex in the hex mesh there are 6 tets
  // For the following, pts 0,1,2,3 are on z=0 and pts 4,5,6,7 are on z=1
  // Elem 0 = 0 3 1 7 = 4 + 4 + 4 + 8 = 20 / 4 = 5
  // Elem 1 = 0 2 3 7 = 4 + 4 + 4 + 8 = 20 / 4 = 5
  // Elem 2 = 0 6 2 7 = 4 + 8 + 4 + 8 = 24 / 4 = 6
  // Elem 3 = 0 4 6 7 = 4 + 8 + 8 + 8 = 28 / 4 = 7
  // Elem 4 = 0 5 4 7 = 4 + 8 + 8 + 8 = 28 / 4 = 7
  // Elem 5 = 0 1 5 7 = 4 + 4 + 8 + 8 = 24 / 4 = 6
  // Repeat this pattern for each hex in the hex mesh to get the following answer
  // z=0, point values = 4; z=1, point values = 8; z=2, point values = 12.
  const float ans[48] = { 5.f, 5.f,  6.f,  7.f,  7.f,  6.f, // Hex0
                          5.f, 5.f,  6.f,  7.f,  7.f,  6.f, // Hex1
                          5.f, 5.f,  6.f,  7.f,  7.f,  6.f, // Hex2
                          5.f, 5.f,  6.f,  7.f,  7.f,  6.f, // Hex3
                          9.f, 9.f, 10.f, 11.f, 11.f, 10.f, // Hex4
                          9.f, 9.f, 10.f, 11.f, 11.f, 10.f, // Hex5
                          9.f, 9.f, 10.f, 11.f, 11.f, 10.f, // Hex6
                          9.f, 9.f, 10.f, 11.f, 11.f, 10.f};// Hex7

  // Execute the filter over the collection
  dray::CellAverage filter;
  filter.set_field("simple");
  dray::Collection result = filter.execute(input);
  using ElemTypeScalar = dray::Element<3, 1, dray::ElemType::Simplex, dray::Order::Constant>;
  test_result<ElemTypeScalar>(ans, 48, result, "simple");

  // Now check the vector field
  filter.set_field("simple_vec");
  result = filter.execute(input);
  using ElemTypeVec = dray::Element<3, 3, dray::ElemType::Simplex, dray::Order::Constant>;
  test_result<ElemTypeVec>(ans, 48, result, "simple_vec");
}

TEST(dray_cell_average, hexes)
{
  // Load the example mesh
  conduit::Node n_input;
  conduit::blueprint::mesh::examples::braid("hexs",
                                             EXAMPLE_MESH_SIDE_DIM,
                                             EXAMPLE_MESH_SIDE_DIM,
                                             EXAMPLE_MESH_SIDE_DIM,
                                             n_input.add_child("domain0"));
  n_input[0]["fields"]["simple"].set(make_simple_field(8, EXAMPLE_MESH_SIDE_DIM, EXAMPLE_MESH_SIDE_DIM, EXAMPLE_MESH_SIDE_DIM));
  n_input[0]["fields"]["simple_vec"].set(make_simple_field(8, EXAMPLE_MESH_SIDE_DIM, EXAMPLE_MESH_SIDE_DIM, EXAMPLE_MESH_SIDE_DIM, 3));

  // Convert it to a dray collection
  n_input[0].save("hexes_input.yaml");
  dray::DataSet input_domain = dray::BlueprintLowOrder::import(n_input[0]);
  dray::Collection input;
  input.add_domain(input_domain);

  // z=0, point values = 8
  // z=1, point values = 16
  // z=2, point values = 24
  // Elems 0, 1, 2, 3 = (8*4 + 16*4) / 8 = 12
  // Elems 4, 5, 6, 7 = (16*4 + 24*4) / 8 = 20
  const float ans[8] = {12.f, 12.f, 12.f, 12.f, 20.f, 20.f, 20.f, 20.f};

  // Execute the filter over the collection
  dray::CellAverage filter;
  filter.set_field("simple");
  dray::Collection result = filter.execute(input);
  using ElemTypeScalar = dray::Element<3, 1, dray::ElemType::Tensor, dray::Order::Constant>;
  test_result<ElemTypeScalar>(ans, 8, result, "simple");

  // Now check the vector field
  filter.set_field("simple_vec");
  result = filter.execute(input);
  using ElemTypeVec = dray::Element<3, 3, dray::ElemType::Tensor, dray::Order::Constant>;
  test_result<ElemTypeVec>(ans, 8, result, "simple_vec");
}
