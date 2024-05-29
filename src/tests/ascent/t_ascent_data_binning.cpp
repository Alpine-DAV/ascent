//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_data_binning.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <ascent_expression_eval.hpp>
#include <expressions/ascent_blueprint_architect.hpp>
#include <runtimes/expressions/ascent_memory_manager.hpp>

#include <cmath>
#include <iostream>

#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>
#include <conduit_fmt/conduit_fmt.h>

#include "t_config.hpp"
#include "t_utils.hpp"

using namespace std;
using namespace conduit;
using namespace ascent;

index_t EXAMPLE_MESH_SIDE_DIM = 5;


// ----------------------------------------------------------------------------
// output helper
void print_result(const std::string &tag,
                  const std::string &expr, 
                  const conduit::Node &res)
{
  const std::string casebreak = "***************************";
  std::cout << casebreak << std::endl
            << tag << std::endl
            << expr << std::endl
            << res.to_yaml() << std::endl
            << casebreak << std::endl;
}

// ----------------------------------------------------------------------------
// helper to create 2D test meshes
void generate_2d_basic_test_mesh(const std::string &mtype,
                                 int nverts_x,
                                 int nverts_y,
                                 conduit::Node &data)
{
  data.reset();
  Node single_domain, multi_domain;
  Node verify_info;
  conduit::blueprint::mesh::examples::basic(mtype, nverts_x, nverts_y, 0, single_domain);

  int num_verts = nverts_x * nverts_y;
  int num_eles  = (nverts_x -1) * (nverts_y -1);

  // add extra fields
  single_domain["fields/ones/association"] = "element";
  single_domain["fields/ones/topology"] = single_domain["topologies"][0].name();
  single_domain["fields/ones/values"].set(DataType::float64(num_eles));

  float64_array ones_vals = single_domain["fields/ones/values"].value();
  ones_vals.fill(1.0);

  single_domain["fields/ones_vert/association"] = "vertex";
  single_domain["fields/ones_vert/topology"] = single_domain["topologies"][0].name();
  single_domain["fields/ones_vert/values"].set(DataType::float64(num_verts));

  float64_array ones_verts_val = single_domain["fields/ones_vert/values"].value();
  ones_verts_val.fill(1.0);

  single_domain["fields/field_vert/association"] = "vertex";
  single_domain["fields/field_vert/topology"] = single_domain["topologies"][0].name();
  single_domain["fields/field_vert/values"].set(DataType::float64(num_verts));

  float64_array field_verts_val = single_domain["fields/field_vert/values"].value();
  for(index_t i =0;i< field_verts_val.number_of_elements();i++)
  {
      field_verts_val[i] = i;
  }

  // ascent normally adds this but we are doing an end around
  single_domain["state/cycle"] = 100;
  single_domain["state/time"] = 1.3;
  single_domain["state/domain_id"] = 0;
  blueprint::mesh::to_multi_domain(single_domain, multi_domain);
  
  data.set(multi_domain);

  std::string ofname = conduit_fmt::format("tout_data_binning_input_mesh_basic_2d_{}_{}_{}.yaml",
                                           mtype,
                                           nverts_x,
                                           nverts_y);
  string output_path = prepare_output_dir();
  std::string output_file = conduit::utils::join_file_path(output_path,ofname);

  conduit::relay::io::blueprint::save_mesh(data,output_file,"hdf5");

}

// -----------------------------------------------------------------
// tests 2d meshes quad mesh cases
void test_binning_basic_mesh_2d_quads(const std::string &tag,Node &data)
{
  std::cout << " --------------------------- " << std::endl
            << tag << std::endl
            << " --------------------------- " << std::endl;

  conduit::Node res;
  std::string expr;

  runtime::expressions::register_builtin();
  runtime::expressions::ExpressionEval eval(&data);

  // single bin, should be the same as the sum of input field
  // -10 --  10 : =  0 + 1 + 2 + 3 
  expr = "binning('field', 'sum', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"6.0");
  print_result(tag + "_single_sum",expr,res);

  // single bin, should be the same as the min of input field
  // -10 --  10 : =  min (0  1  2  3 )
  expr = "binning('field', 'min', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"0.0");
  print_result(tag + "_single_min",expr,res);

  // single bin, should be the same as the max of input field
  // -10 --  10 : =  max (0  1  2  3 )
  expr = "binning('field', 'max', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"3.0");
  print_result(tag + "_single_max",expr,res);

  // single bin, should be the same as the avg of input field
  // -10 --  10 : =  (0 + 1 + 2 + 3 ) / 4.0
  expr = "binning('field', 'avg', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"1.5");
  print_result(tag + "_single_avg",expr,res);

  // single bin, std dev (population variant, normed by N)
  // sqrt( ( (0 - 1.5)^2  + (1 - 1.5)^2 + (2 - 1.5)^2 (3 - 1.5)^2 ) / 4.0 )
  expr = "binning('field', 'std', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"1.11803398874989");
  print_result(tag + "_single_std",expr,res);

  // single bin, var
  // ( (0 - 1.5)^2  + (1 - 1.5)^2 + (2 - 1.5)^2 (3 - 1.5)^2 ) / 4.0
  expr = "binning('field', 'var', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"1.25");
  print_result(tag + "_single_var",expr,res);

  // single bin, rms
  // sqrt( ( 0^2  + 1^2 + 2^2 + 3^2 ) / 4.0 )
  expr = "binning('field', 'rms', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"1.87082869338697");
  print_result(tag + "_single_rms",expr,res);

  // single bin, pdf should be 1.0 ?
  // TODO: count does not need a field!, but then assoc is ambiguous
  expr = "binning('field', 'pdf', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"1.0");
  print_result(tag + "_single_pdf",expr,res);

  // single bin, pdf should be 1.0 ?
  // TODO: count does not need a field!, but then assoc is ambiguous
  expr = "binning('field', 'count', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"4.0");
  print_result(tag + "_single_count",expr,res);

  // -10 --  0 : =  0 + 2
  //   0 -- 10 : =  1 + 3
  expr = "binning('field', 'sum', [axis('x', [-10, 0, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(), "[2.0, 4.0]");
  print_result(tag + "_two_bins_sum",expr,res);

  // -------------
  // clamp = True
  expr = "binning('field', 'max', [axis('x', [-4, 0, 4], clamp=True)])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(), "[2.0, 3.0]");
  print_result(tag + "_two_bins_max_clamp",expr,res);
  // -------------

  // -------------
  // clamp = False
  expr = "binning('field', 'max', [axis('x', [-4, 0, 4])])";
  res = eval.evaluate(expr);
  // default uncovered is 0.0
  EXPECT_EQ(res["attrs/value/value"].to_yaml(), "[0.0, 0.0]");
  print_result(tag + "_two_bins_max_no_clamp",expr,res);
  // -------------

  // -------------
  // clamp = False, totally out of range
  expr = "binning('field', 'min', [axis('x', [-100, -50, -25], clamp=False)],empty_bin_val=-42)";
  res = eval.evaluate(expr);
  std::cout << expr << std::endl << res.to_yaml() << std::endl;
  EXPECT_EQ(res["attrs/value/value"].to_yaml(), "[-42.0, -42.0]");
  print_result(tag + "_two_bins_min_no_clamp_custom_uncovered",expr,res);
  // -------------

  // -------------
  expr = "binning('field', 'max', [axis('x', [-5, 0, 1], clamp=True)])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(), "[2.0, 3.0]");
  print_result(tag + "_two_bins_max_offset_clamp",expr,res);
  // -------------

  // -------------
  expr = "binning('field', 'max', [axis('x', [-5, 0, 1], clamp=False)])";
  res = eval.evaluate(expr);
  // default uncovered is 0
  EXPECT_EQ(res["attrs/value/value"].to_yaml(), "[2.0, 0.0]");
  print_result(tag + "_two_bins_max_offset_clamp",expr,res);
  // -------------

  // -------------
  expr =
      "binning('field', 'max', [axis('x', num_bins=2), axis('y', num_bins=2)], "
      "empty_bin_val=100)";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"[0.0, 1.0, 2.0, 3.0]");
  print_result(tag + "_spatial_axis_max",expr,res);
  // -------------

  // -------------
  expr = "binning('field', 'sum', [axis('field', num_bins=4, clamp=True)])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),
            "[0.0, 1.0, 2.0, 3.0]");
  print_result(tag + "_field_axis_sum",expr,res);
  // -------------

  // -------------
  expr = "binning('ones', 'count', [axis('field', num_bins=4, clamp=True)])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),
            "[1.0, 1.0, 1.0, 1.0]");
  print_result(tag + "_field_axis_count",expr,res);
  // -------------

  // -------------
  expr = "binning('ones', 'pdf', [axis('field', num_bins=4, clamp=True)])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),
            "[0.25, 0.25, 0.25, 0.25]");
  print_result(tag + "_field_axis_pdf",expr,res);
  // -------------

  // ------------------------------------
  // ------------------------------------
  // ------------------------------------
  // vert centered
  // single bin, should be the same as the sum of input field
  // -10 --  10 : =  sum( 0 1  2  3  4  5  6  7  8 )
  expr = "binning('field_vert', 'sum', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"36.0");
  print_result(tag + "_vert_single_sum",expr,res);

  // single bin, should be the same as the min of input field
  // -10 --  10 : =  min (0  1  2  3  4  5  6  7  8)
  expr = "binning('field_vert', 'min', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"0.0");
  print_result(tag + "_vert_single_min",expr,res);

  // single bin, should be the same as the max of input field
  // -10 --  10 : =  max (0  1  2  3  4  5  6  7  8 )
  expr = "binning('field_vert', 'max', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"8.0");
  print_result(tag + "_vert_single_max",expr,res);
  //
  // single bin, should be the same as the avg of input field
  // -10 --  10 : =  (0  1  2  3  4  5  6  7  8 ) / 9.0
  expr = "binning('field_vert', 'avg', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"4.0");
  print_result(tag + "_vert_single_avg",expr,res);
  //
  // single bin, std dev (population variant, normed by N)
  // sqrt( ( (0 - 4)^2 + (1 - 4)^2 + (2 - 4)^2 + (3 - 4)^2 +
  //         (4 - 4)^2 + (5 - 4)^2 + (6 - 4)^2 + (7 - 4)^2 +
  //         (8 - 4)^2 ) / 9.0 )
  expr = "binning('field_vert', 'std', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"2.58198889747161");
  print_result(tag + "_vert__single_std",expr,res);

  // single bin, var
  // ( (0 - 4)^2 + (1 - 4)^2 + (2 - 4)^2 + (3 - 4)^2 +
  //   (4 - 4)^2 + (5 - 4)^2 + (6 - 4)^2 + (7 - 4)^2 +
  //   (8 - 4)^2 ) / 9.0
  expr = "binning('field_vert', 'var', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"6.66666666666667");
  print_result(tag + "_vert__single_var",expr,res);

  // single bin, rms
  // sqrt( ( 0^2  + 1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2 + 8^2 ) / 9.0 )
  expr = "binning('field_vert', 'rms', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"4.76095228569523");
  print_result(tag + "_vert__single_rms",expr,res);

  // single bin, pdf should be 1.0 ?
  // TODO: count does not need a field!, but then assoc is ambiguous
  expr = "binning('field_vert', 'pdf', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"1.0");
  print_result(tag + "_vert__single_pdf",expr,res);

  // single bin, pdf should be 1.0 ?
  // TODO: count does not need a field!, but then assoc is ambiguous
  expr = "binning('field_vert', 'count', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"9.0");
  print_result(tag + "_vert__single_count",expr,res);


}

// -----------------------------------------------------------------
// tests 2d meshes tri mesh cases
void test_binning_basic_mesh_2d_tris(const std::string &tag,
                                     Node &data)
{
  std::cout << " --------------------------- " << std::endl
            << tag << std::endl
            << " --------------------------- " << std::endl;

  conduit::Node res;
  std::string expr;

  runtime::expressions::register_builtin();
  runtime::expressions::ExpressionEval eval(&data);

  // single bin, should be the same as the sum of input field
  // -10 --  10 : =  0 + 1 + 2 + 3 + 4 + 6 + 6 + 7
  expr = "binning('field', 'sum', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"28.0");
  print_result(tag + "_single_sum",expr,res);

  // single bin, should be the same as the min of input field
  // -10 --  10 : =  min (0  1  2  3  4  5  6  7)
  expr = "binning('field', 'min', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"0.0");
  print_result(tag + "_single_min",expr,res);

  // single bin, should be the same as the max of input field
  // -10 --  10 : =  max (0  1  2  3  4  5  6  7)
  expr = "binning('field', 'max', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"7.0");
  print_result(tag + "_single_max",expr,res);

  // single bin, should be the same as the avg of input field
  // -10 --  10 : =  (0 + 1 + 2 + 3 ) / 4.0
  expr = "binning('field', 'avg', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"3.5");
  print_result(tag + "_single_avg",expr,res);

}

// -----------------------------------------------------------------
// tests 3d meshes tet mesh cases
void test_binning_basic_mesh_3d_tets(const std::string &tag,
                                     Node &data)
{
  std::cout << " --------------------------- " << std::endl
            << tag << std::endl
            << " --------------------------- " << std::endl;

  conduit::Node res;
  std::string expr;

  runtime::expressions::register_builtin();
  runtime::expressions::ExpressionEval eval(&data);

  // single bin, should be the same as the sum of input field
  // -10 --  10 : =  0 + 1 + 2 + 3 + 4 + 6 + 6 + 7
  expr = "binning('field', 'sum', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"28.0");
  print_result(tag + "_single_sum",expr,res);

  // single bin, should be the same as the min of input field
  // -10 --  10 : =  min (0  1  2  3  4  5  6  7)
  expr = "binning('field', 'min', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"0.0");
  print_result(tag + "_single_min",expr,res);

  // single bin, should be the same as the max of input field
  // -10 --  10 : =  max (0  1  2  3  4  5  6  7)
  expr = "binning('field', 'max', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"7.0");
  print_result(tag + "_single_max",expr,res);

  // single bin, should be the same as the avg of input field
  // -10 --  10 : =  (0 + 1 + 2 + 3 ) / 4.0
  expr = "binning('field', 'avg', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"3.5");
  print_result(tag + "_single_avg",expr,res);

}



// ----------------------------------------------------------------------------
// helper to create 2D test meshes
void generate_3d_basic_test_mesh(const std::string &mtype,
                                 int nverts_x,
                                 int nverts_y,
                                 int nverts_z,
                                 conduit::Node &data)
{
  data.reset();
  Node single_domain, multi_domain;
  Node verify_info;
  conduit::blueprint::mesh::examples::basic(mtype, nverts_x, nverts_y, nverts_z, single_domain);

  int num_eles = (nverts_x -1) * (nverts_y -1) * (nverts_y -1);

  // add an extra field
  single_domain["fields/ones/association"] = "element";
  single_domain["fields/ones/topology"] = single_domain["topologies"][0].name();
  single_domain["fields/ones/values"].set(DataType::float64(num_eles));

  float64_array ones_vals = single_domain["fields/ones/values"].value();
  ones_vals.fill(1.0);

  // ascent normally adds this but we are doing an end around
  single_domain["state/cycle"] = 100;
  single_domain["state/time"] = 1.3;
  single_domain["state/domain_id"] = 0;
  blueprint::mesh::to_multi_domain(single_domain, multi_domain);

  data.set(multi_domain);

  std::string ofname = conduit_fmt::format("tout_data_binning_input_mesh_basic_3d_{}_{}_{}_{}.yaml",
                                           mtype,
                                           nverts_x,
                                           nverts_y,
                                           nverts_z);

  string output_path = prepare_output_dir();
  std::string output_file = conduit::utils::join_file_path(output_path,ofname);

  conduit::relay::io::blueprint::save_mesh(data,output_file,"hdf5");

}

// -----------------------------------------------------------------
// tests 3d meshes quad mesh cases
void test_binning_basic_mesh_3d_quads(const std::string &tag,
                                      Node &data)
{
  std::cout << " --------------------------- " << std::endl
            << tag << std::endl
            << " --------------------------- " << std::endl;

  conduit::Node res;
  std::string expr;

  runtime::expressions::register_builtin();
  runtime::expressions::ExpressionEval eval(&data);

  // single bin, should be the same as the sum of input field
  // -10 --  10 : =  0 + 1 + 2 + 3 + 4 + 5 + 6 + 7
  expr = "binning('field', 'sum', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"28.0");
  print_result(tag + "_single_sum",expr,res);

  // single bin, should be the same as the min of input field
  // -10 --  10 : =  min (0  1  2  3  4  5  6  7)
  expr = "binning('field', 'min', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"0.0");
  print_result(tag + "_single_min",expr,res);

  // single bin, should be the same as the min of input field
  // -10 --  10 : =  max (0  1  2  3  4  5  6  7)
  expr = "binning('field', 'max', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"7.0");
  print_result(tag + "_single_max",expr,res);

  // single bin, should be the same as the avg of input field
  // -10 --  10 : =  (0 + 1 + 2 + 3 + 4 + 5 + 6 + 7) / 8.0
  expr = "binning('field', 'avg', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"3.5");
  print_result(tag + "_single_avg",expr,res);

  // single bin, std dev
  // sqrt( ( (0 - 3.5)^2  + (1 - 3.5)^2 + (2 - 3.5)^2
  //         (3 - 3.5)^2  + (4 - 3.5)^2 + (5 - 3.5)^2
  //         (6 - 3.5)^2  + (7 - 3.5)^2 ) ) / 8.0 )
  expr = "binning('field', 'std', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"2.29128784747792");
  print_result(tag + "_single_std",expr,res);

  // single bin, var
  //  ( (0 - 3.5)^2  + (1 - 3.5)^2 + (2 - 3.5)^2
  //    (3 - 3.5)^2  + (4 - 3.5)^2 + (5 - 3.5)^2
  //    (6 - 3.5)^2  + (7 - 3.5)^2 ) / 8.0
  expr = "binning('field', 'var', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"5.25");
  print_result(tag + "_single_var",expr,res);

  // single bin, rms
  // sqrt( ( 0^2  + 1^2 + 2^2
  //       ( 3^2  + 4^2 + 5^2
  //       ( 6^2  + 7^2 ) / 8.0)
  expr = "binning('field', 'rms', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"4.18330013267038");
  print_result(tag + "_single_std",expr,res);

  // single bin, pdf should be 1.0 ?
  // TODO: count does not need a field!, but then assoc is ambiguous
  expr = "binning('field', 'pdf', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"1.0");
  print_result(tag + "_single_pdf",expr,res);

  // single bin, pdf should be 1.0 ?
  // TODO: count does not need a field!, but then assoc is ambiguous
  expr = "binning('field', 'count', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  print_result(tag + "_single_count",expr,res);

  // -10 --  0 : =  0 + 2 + 4 + 6
  //   0 -- 10 : =  1 + 3 + 5 + 7
  expr = "binning('field', 'sum', [axis('x', [-10, 0, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(), "[12.0, 16.0]");
  print_result(tag + "_two_bin_sum",expr,res);

  // -------------
  // clamp = False
  expr = "binning('field', 'max', [axis('z', [-4, 0, 4], clamp=True)])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(), "[3.0, 7.0]");
  print_result(tag + "_spatial_max_clamp",expr,res);
  // -------------

  // -------------
  // clamp = False
  expr = "binning('field', 'max', [axis('z', [-4, 0, 4])])";
  res = eval.evaluate(expr);
  // default uncovered is 0.0
  EXPECT_EQ(res["attrs/value/value"].to_yaml(), "[0.0, 0.0]");
  print_result(tag + "_spatial_max_no_clamp",expr,res);
  // -------------

  // -------------
  // clamp = False, totally out of range
  expr = "binning('field', 'min', [axis('z', [-100, -50, -25], clamp=False)],empty_bin_val=-42)";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(), "[-42.0, -42.0]");
  print_result(tag + "_spatial_min_no_clamp_out_of_bounds",expr,res);
  // -------------

  // -------------
  expr = "binning('field', 'max', [axis('z', [-5, 0, 1], clamp=True)])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(), "[3.0, 7.0]");
  print_result(tag + "_spatial_max_clamp",expr,res);
  // -------------

  // -------------
  expr = "binning('field', 'max', [axis('z', [-5, 0, 1], clamp=False)])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(), "[3.0, 0.0]");
  print_result(tag + "_spatial_offset_bin_max_no_clamp",expr,res);
  // -------------

  // -------------
  expr =
      "binning('field', 'max', [axis('x', num_bins=2), axis('y', num_bins=2)], "
      "empty_bin_val=100)";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"[4.0, 5.0, 6.0, 7.0]");
  print_result(tag + "_spatial_offset_bin_max_no_clamp",expr,res);
  // -------------

  // -------------
  expr =
      "binning('field', 'sum', [axis('x', num_bins=2), axis('y', num_bins=2), "
      "axis('z', num_bins=2)])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),
            "[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]");
  print_result(tag + "_spatial_axis_sum",expr,res);
  // -------------

  // -------------
  expr = "binning('ones', 'count', [axis('field', num_bins=8, clamp=True)])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),
            "[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]");
  print_result(tag + "_field_axis_sum",expr,res);
  // -------------

  // -------------
  expr = "binning('ones', 'pdf', [axis('field', num_bins=8, clamp=True)])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),
            "[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]");
  print_result(tag + "_field_axis_pdf",expr,res);
  // -------------

  // -------------
  expr = "binning('ones', 'pdf', [axis('x', num_bins=2), axis('y', "
         "num_bins=2), axis('z', num_bins=2)])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),
             "[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]");
  print_result(tag + "_spatial_axis_pdf",expr,res);
  // -------------

  // -------------
  expr = "binning('ones', 'pdf', [axis('x', num_bins=2), axis('y', "
         "num_bins=2), axis('z',[-10, 10, 20], clamp=False)], empty_bin_val=0)";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),
             "[0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0]");
  print_result(tag + "_spatial_axis_offset_pdf",expr,res);
 // -------------

  // -------------
  expr = "binning('ones', 'count', [axis('x', num_bins=2), axis('y', "
         "num_bins=2), axis('z',[-10, 10, 20], clamp=False)], empty_bin_val=0)";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),
             "[2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0]");
  print_result(tag + "_spatial_axis_offset_count",expr,res);
  // -------------

  // -------------
  expr = "binning('ones', 'count', [axis('x', num_bins=2), axis('y', "
         "num_bins=2), axis('z',[-4, 10, 20], clamp=False)], empty_bin_val=0)";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),
             "[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]");
  print_result(tag + "_spatial_axis_offset_count",expr,res);
  // -------------

  // -------------
  expr = "binning('ones', 'count', [axis('x', num_bins=2), axis('y', "
         "num_bins=2), axis('z',[-25, -15, 20], clamp=False)], empty_bin_val=0)";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),
             "[0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0]");
  print_result(tag + "_spatial_axis_offset_count",expr,res);
  // -------------
}

//-----------------------------------------------------------------------------
void test_binning_pipline_filter(const std::string &tag,
                                 Node &data,
                                 const Node &binning_filter_def)
{
  // the vtkm runtime is currently our only rendering runtime
  Node n;
  ascent::about(n);
  // only run this test if ascent was built with vtkm support
  if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
  {
      ASCENT_INFO("Ascent vtkm support disabled, skipping test");
      return;
  }

  string output_path = prepare_output_dir();
  string output_file = conduit::utils::join_file_path(output_path,"tout_binning_" + tag + "_result_render");

  // remove old images before rendering
  remove_test_image(output_file);

  conduit::Node actions;

  // add the pipeline
  conduit::Node &add_pipelines= actions.append();
  add_pipelines["action"] = "add_pipelines";
  conduit::Node &pipelines = add_pipelines["pipelines"];

  // pipeline 1
  pipelines["pl1/f1"] = binning_filter_def;

  conduit::Node &add_scenes= actions.append();
  add_scenes["action"] = "add_scenes";
  conduit::Node &scenes = add_scenes["scenes"];
  scenes["s1/plots/p1/type"] = "pseudocolor";
  scenes["s1/plots/p1/field"] = "binning";
  scenes["s1/plots/p1/pipeline"] = "pl1";
  scenes["s1/plots/p2/type"] = "mesh";
  scenes["s1/plots/p2/pipeline"] = "pl1";


  // scenes["s1/renders/r1/image_name"] = output_file;
  scenes["s1/image_prefix"] = output_file;

  // add extract
  conduit::Node &add_extracts = actions.append();
  add_extracts["action"] = "add_extracts";
  conduit::Node &extracts = add_extracts["extracts"];

  extracts["e1/type"] = "relay";
  extracts["e1/pipeline"] = "pl1";
  extracts["e1/params/path"] = conduit::utils::join_file_path(output_path,
                                                              "tout_binning_" + tag + "_result_extract");
  extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

  actions.print();

  //
  // Run Ascent
  //
  Ascent ascent;
  ascent.open();
  ascent.publish(data);
  ascent.execute(actions);
  ascent.close();

  EXPECT_TRUE(check_test_image(output_file, 0.1));
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//
//
//
// //-----------------------------------------------------------------------------
// TEST(ascent_binning, binning_basic_meshes_2d)
// {
//   // the vtkm runtime is currently our only rendering runtime
//   Node n;
//   ascent::about(n);
//   // only run this test if ascent was built with vtkm support
//   if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
//   {
//     ASCENT_INFO("Ascent support disabled, skipping test");
//     return;
//   }
//
//   //
//   // Create an example mesh.
//   //
//
//   // for all cases
//   // extents of basic are -10, 10
//   // with 3x3 nodes, there are 4 elements
//   //
//
//
//   // "uniform", "rectilinear", "structured","tris", "quads", "polygons"
//
//   Node data;
//   generate_2d_basic_test_mesh("uniform", 3, 3,data);
//   test_binning_basic_mesh_2d_quads("2d_uniform_3_3_0",data);
//
//   generate_2d_basic_test_mesh("rectilinear", 3, 3,data);
//   test_binning_basic_mesh_2d_quads("2d_rectilinear_3_3_0",data);
//
//   // TODO: UNSUPPORTED
//   // generate_2d_basic_test_mesh("structured", 3, 3, data);
//   // test_binning_basic_mesh_2d(data,"2d_structured_3_3_0");
//
//   generate_2d_basic_test_mesh("quads", 3, 3,data);
//   test_binning_basic_mesh_2d_quads("2d_quads_3_3_0",data);
//   // TODO: UNSUPPORTED
//   // generate_2d_basic_test_mesh("polygons", 3, 3,data);
//   // test_binning_basic_mesh_2d_quads("2d_polygons_3_3_0",data);
//
//   // triangles produce different counts
//   generate_2d_basic_test_mesh("tris", 3, 3,data);
//   test_binning_basic_mesh_2d_tris("2d_tris_3_3_0",data);
//
//
// }
//
// //-----------------------------------------------------------------------------
// TEST(ascent_binning, binning_basic_meshes_3d)
// {
//   // the vtkm runtime is currently our only rendering runtime
//   Node n;
//   ascent::about(n);
//   // only run this test if ascent was built with vtkm support
//   if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
//   {
//     ASCENT_INFO("Ascent support disabled, skipping test");
//     return;
//   }
//
//   runtime::expressions::register_builtin();
//
//   // "uniform", "rectilinear", "structured","tets", "hexs", "polyhedra"
//   Node data;
//   generate_3d_basic_test_mesh("uniform", 3, 3, 3,data);
//   test_binning_basic_mesh_3d_quads("3d_uniform_3_3_3",data);
//
//   generate_3d_basic_test_mesh("rectilinear", 3, 3, 3,data);
//   test_binning_basic_mesh_3d_quads("3d_rectilinear_3_3_3",data);
//
//   // // TODO NOT SUPPORTED
//   // generate_3d_basic_test_mesh("structured", 3, 3, 3,data);
//   // test_binning_basic_mesh_3d_quads("3d_structured_3_3_3",data);
//
//   generate_3d_basic_test_mesh("hexs", 3, 3, 3,data);
//   test_binning_basic_mesh_3d_quads("3d_hexs_3_3_3",data);
//
//   generate_3d_basic_test_mesh("tets", 3, 3, 3,data);
//   test_binning_basic_mesh_3d_tets("3d_tets_3_3_3",data);
//
//   // // TODO NOT SUPPORTED
//   // generate_3d_basic_test_mesh("polyhedra", 3, 3, 3,data);
//   // test_binning_basic_mesh_3d_quads("3d_poly_3_3_3",data);
//
//
//
// }
//
//
// TEST(ascent_binning, binning_errors)
// {
//   return;
//   // the vtkm runtime is currently our only rendering runtime
//   Node n;
//   ascent::about(n);
//   // only run this test if ascent was built with vtkm support
//   if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
//   {
//     ASCENT_INFO("Ascent support disabled, skipping test");
//     return;
//   }
//
//   //
//   // Create an example mesh.
//   //
//   Node data, verify_info;
//   conduit::blueprint::mesh::examples::braid("hexs",
//                                             EXAMPLE_MESH_SIDE_DIM,
//                                             EXAMPLE_MESH_SIDE_DIM,
//                                             EXAMPLE_MESH_SIDE_DIM,
//                                             data);
//   // ascent normally adds this but we are doing an end around
//   data["state/domain_id"] = 0;
//   Node multi_dom;
//   blueprint::mesh::to_multi_domain(data, multi_dom);
//
//   runtime::expressions::register_builtin();
//   runtime::expressions::ExpressionEval eval(&multi_dom);
//
//   conduit::Node res;
//   std::string expr;
//
//   bool threw = false;
//   try
//   {
//     expr = "binning('', 'avg', [axis('x'), axis('y')])";
//     res = eval.evaluate(expr);
//   }
//   catch(...)
//   {
//     threw = true;
//   }
//   EXPECT_EQ(threw, true);
//
//   threw = false;
//   try
//   {
//     expr = "binning('braid', 'sum', [axis('x'), axis('vel')])";
//     res = eval.evaluate(expr);
//   }
//   catch(...)
//   {
//     threw = true;
//   }
//   EXPECT_EQ(threw, true);
//
//   threw = false;
//   try
//   {
//     expr = "binning('vel', 'sum', [axis('x'), axis('y')])";
//     res = eval.evaluate(expr);
//   }
//   catch(...)
//   {
//     threw = true;
//   }
//   EXPECT_EQ(threw, true);
//
//   threw = false;
//   try
//   {
//     expr = "binning('braid', 'sum', [axis('x', bins=[1,2], num_bins=1), "
//            "axis('y')])";
//     res = eval.evaluate(expr);
//   }
//   catch(...)
//   {
//     threw = true;
//   }
//   EXPECT_EQ(threw, true);
//
//   threw = false;
//   try
//   {
//     expr = "binning('braid', 'sum', [axis('x', bins=[1]), axis('y')])";
//     res = eval.evaluate(expr);
//   }
//   catch(...)
//   {
//     threw = true;
//   }
//   EXPECT_EQ(threw, true);
// }

//-----------------------------------------------------------------------------
TEST(ascent_binning, filter_braid_binning_mesh)
{
  // the vtkm runtime is currently our only rendering runtime
  Node n;
  ascent::about(n);
  // only run this test if ascent was built with vtkm support
  if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
  {
    ASCENT_INFO("Ascent support disabled, skipping test");
    return;
  }

  string output_path = prepare_output_dir();
  std::string output_file =
      conduit::utils::join_file_path(output_path, "tout_binning_filter");

  remove_test_image(output_file);
  //
  // Create an example mesh.
  //
  Node data, verify_info;
  conduit::blueprint::mesh::examples::braid("hexs", 10, 10, 10, data);

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] = "binning";
  // filter knobs
  conduit::Node &params = pipelines["pl1/f1/params"];
  params["reduction_op"] = "sum";
  params["reduction_field"] = "braid";
  params["output_field"] = "binning";
  // paint the field onto the original mesh
  params["output_type"] = "mesh";

  conduit::Node &axis0 = params["axes"].append();
  axis0["field"] = "x";
  axis0["num_bins"] = 10;
  axis0["min_val"] = -10.0;
  axis0["max_val"] = 10.0;
  axis0["clamp"] = 1;

  conduit::Node &axis1 = params["axes"].append();
  axis1["field"] = "y";
  axis1["num_bins"] = 10;

  conduit::Node scenes;
  scenes["s1/plots/p1/type"] = "pseudocolor";
  scenes["s1/plots/p1/field"] = "binning";
  scenes["s1/plots/p1/pipeline"] = "pl1";
  scenes["s1/plots/p2/type"] = "mesh";
  scenes["s1/plots/p2/pipeline"] = "pl1";
  scenes["s1/renders/r1/camera/zoom"] = 0.85;
  scenes["s1/renders/r1/image_name"] = output_file;

  conduit::Node actions;
  // add the pipeline
  conduit::Node &add_pipelines= actions.append();
  add_pipelines["action"] = "add_pipelines";
  add_pipelines["pipelines"] = pipelines;
  // add the scenes
  conduit::Node &add_scenes= actions.append();
  add_scenes["action"] = "add_scenes";
  add_scenes["scenes"] = scenes;

  //
  // Run Ascent
  //

  Ascent ascent;
  ascent.open();
  ascent.publish(data);
  ascent.execute(actions);
  ascent.close();

  EXPECT_TRUE(check_test_image(output_file, 0.1,""));
}

//-----------------------------------------------------------------------------
TEST(ascent_binning, filter_braid_binning_bins)
{
  // the vtkm runtime is currently our only rendering runtime
  Node n;
  ascent::about(n);
  // only run this test if ascent was built with vtkm support
  if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
  {
    ASCENT_INFO("Ascent support disabled, skipping test");
    return;
  }

  string output_path = prepare_output_dir();
  std::string output_file =
      conduit::utils::join_file_path(output_path, "tout_binning_filter_bins");

  remove_test_image(output_file);
  //
  // Create an example mesh.
  //
  Node data, verify_info;
  conduit::blueprint::mesh::examples::braid("hexs", 20, 20, 20, data);

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] = "binning";
  // filter knobs
  conduit::Node &params = pipelines["pl1/f1/params"];
  params["reduction_op"] = "sum";
  params["reduction_field"] = "braid";
  params["output_field"] = "binning";
  // reduced dataset of only the bins
  params["output_type"] = "bins";

  conduit::Node &axis0 = params["axes"].append();
  axis0["field"] = "x";
  axis0["num_bins"] = 10;
  axis0["min_val"] = -10.0;
  axis0["max_val"] = 10.0;
  axis0["clamp"] = 1;

  conduit::Node &axis1 = params["axes"].append();
  axis1["field"] = "y";
  axis1["num_bins"] = 10;
  axis1["clamp"] = 0;

  conduit::Node &axis2 = params["axes"].append();
  axis2["field"] = "z";
  axis2["num_bins"] = 10;
  axis2["clamp"] = 10;

  conduit::Node scenes;
  scenes["s1/plots/p1/type"] = "pseudocolor";
  scenes["s1/plots/p1/field"] = "binning";
  scenes["s1/plots/p1/pipeline"] = "pl1";
  scenes["s1/image_prefix"] = output_file;

  conduit::Node actions;
  // add the pipeline
  conduit::Node &add_pipelines= actions.append();
  add_pipelines["action"] = "add_pipelines";
  add_pipelines["pipelines"] = pipelines;
  // add the scenes
  conduit::Node &add_scenes= actions.append();
  add_scenes["action"] = "add_scenes";
  add_scenes["scenes"] = scenes;

  //
  // Run Ascent
  //

  Ascent ascent;
  ascent.open();
  ascent.publish(data);
  ascent.execute(actions);
  ascent.close();

  EXPECT_TRUE(check_test_image(output_file, 0.1));
  std::string msg = "An example of data binning, binning spatially and summing a field.";
  ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}


//-----------------------------------------------------------------------------
// this is here b/c there was a bug with using int64 for num_bins
// that caused a conduit access error b/c we expected int32 only
//-----------------------------------------------------------------------------
TEST(ascent_binning, filter_braid_binning_bins_int64_params)
{
  // the vtkm runtime is currently our only rendering runtime
  Node n;
  ascent::about(n);
  // only run this test if ascent was built with vtkm support
  if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
  {
    ASCENT_INFO("Ascent support disabled, skipping test");
    return;
  }

  string output_path = prepare_output_dir();
  std::string output_file =
      conduit::utils::join_file_path(output_path, "tout_binning_filter_bins_int64");

  remove_test_image(output_file);
  //
  // Create an example mesh.
  //
  Node data, verify_info;
  conduit::blueprint::mesh::examples::braid("hexs", 20, 20, 20, data);

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] = "binning";
  // filter knobs
  conduit::Node &params = pipelines["pl1/f1/params"];
  params["reduction_op"] = "sum";
  params["reduction_field"] = "braid";
  params["output_field"] = "binning";
  // reduced dataset of only the bins
  params["output_type"] = "bins";

  conduit::Node &axis0 = params["axes"].append();
  axis0["field"] = "x";
  axis0["num_bins"] = (int64)10;
  axis0["min_val"] = -10.0;
  axis0["max_val"] = 10.0;
  axis0["clamp"] =  (int64)1;

  conduit::Node &axis1 = params["axes"].append();
  axis1["field"] = "y";
  axis1["num_bins"] = (int64)10;
  axis1["clamp"] = (int64)0;

  conduit::Node &axis2 = params["axes"].append();
  axis2["field"] = "z";
  axis2["num_bins"] = (int64)10;
  axis2["clamp"] = 1; // <--?

  conduit::Node scenes;
  scenes["s1/plots/p1/type"] = "pseudocolor";
  scenes["s1/plots/p1/field"] = "binning";
  scenes["s1/plots/p1/pipeline"] = "pl1";
  scenes["s1/image_prefix"] = output_file;

  conduit::Node actions;
  // add the pipeline
  conduit::Node &add_pipelines= actions.append();
  add_pipelines["action"] = "add_pipelines";
  add_pipelines["pipelines"] = pipelines;
  // add the scenes
  conduit::Node &add_scenes= actions.append();
  add_scenes["action"] = "add_scenes";
  add_scenes["scenes"] = scenes;

  //
  // Run Ascent
  //

  Ascent ascent;
  ascent.open();
  ascent.publish(data);
  ascent.execute(actions);
  ascent.close();

  EXPECT_TRUE(check_test_image(output_file, 0.1));
}


//-----------------------------------------------------------------------------
TEST(ascent_binning, expr_braid_non_spatial_bins)
{
  //
  // Create an example mesh.
  //
  Node data, verify_info;
  conduit::blueprint::mesh::examples::braid("hexs", 50, 50, 50, data);

  conduit::Node pipelines;

  // braid is  vertex-assoced
  // radial is element-assoced

  // recenter braid to be element-assoced
  // so we can same assoc for binning


  // pipeline 1
  pipelines["pl1/f1/type"] = "recenter";
  pipelines["pl1/f1/params/field"] = "braid";
  pipelines["pl1/f1/params/association"] = "element";


  conduit::Node actions;
  // add the pipeline
  conduit::Node &add_pipelines= actions.append();
  add_pipelines["action"] = "add_pipelines";
  add_pipelines["pipelines"] = pipelines;

  Node &add_act = actions.append();
  add_act["action"] = "add_queries";

  // declare a queries to ask some questions
  Node &queries = add_act["queries"];

  // Create a 2D binning projected onto the x-y plane
  queries["q2/params/expression"] = "binning('radial','max', [axis('radial',num_bins=10), axis( 'braid' ,num_bins=10)])";
  queries["q2/params/name"] = "my_binning";
  queries["q2/pipeline"] = "pl1";

  // print our full actions tree
  std::cout << actions.to_yaml() << std::endl;

  //
  // Run Ascent
  //

  Ascent ascent;
  ascent.open();
  ascent.publish(data);
  ascent.execute(actions);
  Node ascent_info;
  ascent.info(ascent_info);
  ascent_info["expressions/my_binning"].print();

  ascent.close();

}

//-----------------------------------------------------------------------------
TEST(ascent_binning, binning_render_basic_mesh_cases)
{
  // the vtkm runtime is currently our only rendering runtime
  Node n;
  ascent::about(n);
  // only run this test if ascent was built with vtkm support
  if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
  {
      ASCENT_INFO("Ascent vtkm support disabled, skipping test");
      return;
  }

  Node data;


  std::vector<std::string> mesh_types_2d = {"uniform",
                                            "rectilinear",
                                            // "structured",
                                            // "tris", // skip tris until umr issue is fixed
                                            "quads"};

  for( auto mesh_type : mesh_types_2d)
  {
    data.reset();
    generate_2d_basic_test_mesh(mesh_type,3,3,data);

    {
      Node filter;
      filter["type"] = "binning";
      conduit::Node &params = filter["params"];
      params["reduction_op"] = "sum";
      params["var"] = "ones";
      params["output_field"] = "binning";
      params["output_type"] = "bins";

      conduit::Node &axis0 = params["axes"].append();
      axis0["var"] = "x";
      axis0["num_bins"] = 2;

      conduit::Node &axis1 = params["axes"].append();
      axis1["var"] = "y";
      axis1["num_bins"] = 2;

      test_binning_pipline_filter( mesh_type + "_2d_full_ones",data,filter);
    }

    {
      Node filter;
      filter["type"] = "binning";
      conduit::Node &params = filter["params"];
      params["reduction_op"] = "sum";
      params["var"] = "field";
      params["output_field"] = "binning";
      params["output_type"] = "bins";

      conduit::Node &axis0 = params["axes"].append();
      axis0["var"] = "x";
      axis0["num_bins"] = 2;

      conduit::Node &axis1 = params["axes"].append();
      axis1["var"] = "y";
      axis1["num_bins"] = 2;

      test_binning_pipline_filter( mesh_type + "_2d_full_field",data,filter);
    }

    {
      Node filter;
      filter["type"] = "binning";
      conduit::Node &params =  filter["params"];
      params["reduction_op"] = "sum";
      params["var"] = "ones";
      params["output_field"] = "binning";
      params["output_type"] = "bins";

      conduit::Node &axis0 = params["axes"].append();
      axis0["var"] = "x";
      axis0["num_bins"] = 3;
      axis0["min_val"] = 1.0;
      axis0["max_val"] = 3.0;
      axis0["clamp"] = 1;

      conduit::Node &axis1 = params["axes"].append();
      axis1["var"] = "y";
      axis1["num_bins"] = 3;
      axis1["min_val"] = 1.0;
      axis1["max_val"] = 3.0;
      axis1["clamp"] = 1;
      test_binning_pipline_filter(  mesh_type + "_2d_clamp",data,filter);
    }

    {
      Node filter;
      filter["type"] = "binning";
      conduit::Node &params =  filter["params"];
      params["reduction_op"] = "sum";
      params["var"] = "ones";
      params["output_field"] = "binning";
      params["output_type"] = "bins";

      conduit::Node &axis0 = params["axes"].append();
      axis0["var"] = "x";
      axis0["num_bins"] = 3;
      axis0["min_val"] = 1.0;
      axis0["max_val"] = 3.0;
      axis0["clamp"] = 0;

      conduit::Node &axis1 = params["axes"].append();
      axis1["var"] = "y";
      axis1["num_bins"] = 3;
      axis1["min_val"] = 1.0;
      axis1["max_val"] = 3.0;
      axis1["clamp"] = 0;
      test_binning_pipline_filter(  mesh_type + "_2d_no_clamp",data,filter);
    }
  }

  std::vector<std::string> mesh_types_3d = {"uniform",
                                            "rectilinear",
                                            // "structured",
                                            "tets",
                                            "hexs"};


  for( auto mesh_type : mesh_types_3d)
  {
    data.reset();
    generate_3d_basic_test_mesh(mesh_type,3,3,3,data);

    {
      Node filter;
      filter["type"] = "binning";
      conduit::Node &params = filter["params"];
      params["reduction_op"] = "sum";
      params["var"] = "ones";
      params["output_field"] = "binning";
      params["output_type"] = "bins";

      conduit::Node &axis0 = params["axes"].append();
      axis0["var"] = "x";
      axis0["num_bins"] = 2;

      conduit::Node &axis1 = params["axes"].append();
      axis1["var"] = "y";
      axis1["num_bins"] = 2;

      conduit::Node &axis2 = params["axes"].append();
      axis2["var"] = "z";
      axis2["num_bins"] = 2;

      test_binning_pipline_filter( mesh_type + "_3d_full_ones",data,filter);
    }

    {
      Node filter;
      filter["type"] = "binning";
      conduit::Node &params = filter["params"];
      params["reduction_op"] = "sum";
      params["var"] = "field";
      params["output_field"] = "binning";
      params["output_type"] = "bins";

      conduit::Node &axis0 = params["axes"].append();
      axis0["var"] = "x";
      axis0["num_bins"] = 2;

      conduit::Node &axis1 = params["axes"].append();
      axis1["var"] = "y";
      axis1["num_bins"] = 2;

      conduit::Node &axis2 = params["axes"].append();
      axis2["var"] = "z";
      axis2["num_bins"] = 2;

      test_binning_pipline_filter( mesh_type + "_3d_full_field",data,filter);
    }

    {
      Node filter;
      filter["type"] = "binning";
      conduit::Node &params =  filter["params"];
      params["reduction_op"] = "sum";
      params["var"] = "ones";
      params["output_field"] = "binning";
      params["output_type"] = "bins";

      conduit::Node &axis0 = params["axes"].append();
      axis0["var"] = "x";
      axis0["num_bins"] = 3;
      axis0["min_val"] = 1.0;
      axis0["max_val"] = 3.0;
      axis0["clamp"] = 1;

      conduit::Node &axis1 = params["axes"].append();
      axis1["var"] = "y";
      axis1["num_bins"] = 3;
      axis1["min_val"] = 1.0;
      axis1["max_val"] = 3.0;
      axis1["clamp"] = 1;

      conduit::Node &axis2 = params["axes"].append();
      axis2["var"] = "z";
      axis2["num_bins"] = 3;
      axis2["min_val"] = 1.0;
      axis2["max_val"] = 3.0;
      axis2["clamp"] = 1;

      test_binning_pipline_filter(  mesh_type + "_3d_clamp",data,filter);
    }

    {
      Node filter;
      filter["type"] = "binning";
      conduit::Node &params =  filter["params"];
      params["reduction_op"] = "sum";
      params["var"] = "ones";
      params["output_field"] = "binning";
      params["output_type"] = "bins";

      conduit::Node &axis0 = params["axes"].append();
      axis0["var"] = "x";
      axis0["num_bins"] = 3;
      axis0["min_val"] = 1.0;
      axis0["max_val"] = 3.0;
      axis0["clamp"] = 0;

      conduit::Node &axis1 = params["axes"].append();
      axis1["var"] = "y";
      axis1["num_bins"] = 3;
      axis1["min_val"] = 1.0;
      axis1["max_val"] = 3.0;
      axis1["clamp"] = 0;

      conduit::Node &axis2 = params["axes"].append();
      axis2["var"] = "z";
      axis2["num_bins"] = 3;
      axis2["min_val"] = 1.0;
      axis2["max_val"] = 3.0;
      axis2["clamp"] = 0;

      test_binning_pipline_filter(  mesh_type + "_3d_no_clamp",data,filter);
    }
  }

}

//-----------------------------------------------------------------------------
int
main(int argc, char *argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  // this is normally set in ascent::Initialize, but we
  // have to set it here so that we do the right thing with
  // device pointers
  AllocationManager::set_conduit_mem_handlers();

  // allow override of the data size via the command line
  if(argc == 2)
  {
    EXAMPLE_MESH_SIDE_DIM = atoi(argv[1]);
  }

  result = RUN_ALL_TESTS();
  return result;
}
