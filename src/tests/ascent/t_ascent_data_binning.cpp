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

#include "t_config.hpp"
#include "t_utils.hpp"

using namespace std;
using namespace conduit;
using namespace ascent;

index_t EXAMPLE_MESH_SIDE_DIM = 5;

//-----------------------------------------------------------------------------
TEST(ascent_binning, binning_basic_meshes)
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

  //
  // Create an example mesh.
  //
  Node data, verify_info;

  conduit::blueprint::mesh::examples::basic("hexs", 3, 3, 3, data);

  conduit::relay::io::blueprint::save_mesh(data,"tout_data_binning_basic_3_3_3","hdf5");
  // extents of basic are -10, 10
  // with 3x3x3 nodes, there are 8 elements
  //
  
  // add an extra field
  data["fields/ones/association"] = "element";
  data["fields/ones/topology"] = data["topologies"][0].name();
  data["fields/ones/values"].set(DataType::float64(8));

  float64_array ones_vals = data["fields/ones/values"].value();
  ones_vals.fill(1.0);

  // ascent normally adds this but we are doing an end around
  data["state/cycle"] = 100;
  data["state/time"] = 1.3;
  data["state/domain_id"] = 0;
  Node multi_dom;
  blueprint::mesh::to_multi_domain(data, multi_dom);

  runtime::expressions::register_builtin();
  runtime::expressions::ExpressionEval eval(&multi_dom);

  conduit::Node res;
  std::string expr;
  std::string casebreak = "***************************";

  // single bin, should be the same as the sum of input field
  // -10 --  10 : =  0 + 1 + 2 + 3 + 4 + 5 + 6 + 7
  expr = "binning('field', 'sum', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  std::cout << expr << std::endl << res.to_yaml() << std::endl;
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"28.0");

  // single bin, should be the same as the min of input field
  // -10 --  10 : =  min (0  1  2  3  4  5  6  7)
  expr = "binning('field', 'min', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  std::cout << expr << std::endl << res.to_yaml() << std::endl;
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"0.0");

  // single bin, should be the same as the min of input field
  // -10 --  10 : =  max (0  1  2  3  4  5  6  7)
  expr = "binning('field', 'max', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  std::cout << expr << std::endl << res.to_yaml() << std::endl;
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"7.0");

  // single bin, should be the same as the avg of input field
  // -10 --  10 : =  (0 + 1 + 2 + 3 + 4 + 5 + 6 + 7) / 8.0
  expr = "binning('field', 'avg', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  std::cout << expr << std::endl << res.to_yaml() << std::endl;
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"3.5");

  // single bin, std dev
  // sqrt( ( (0 - 3.5)^2  + (1 - 3.5)^2 + (2 - 3.5)^2
  //       ( )(3 - 3.5)^2  + (4 - 3.5)^2 + (5 - 3.5)^2
  //       ( )(6 - 3.5)^2  + (7 - 3.5)^2 ) / 8.0 )
  expr = "binning('field', 'std', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  std::cout << expr << std::endl << res.to_yaml() << std::endl;
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"5.25");

  // single bin, rms
  // sqrt( ( 0^2  + 1^2 + 2^2
  //       ( 3^2  + 4^2 + 5^2
  //       ( 6^2  + 7^2 ) / 8.0)
  expr = "binning('field', 'rms', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  std::cout << expr << std::endl << res.to_yaml() << std::endl;
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"4.18330013267038");

  // single bin, pdf should be 1.0 ?
  // TODO: count does not need a field!, but then assoc is ambiguous
  expr = "binning('field', 'pdf', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  std::cout << expr << std::endl << res.to_yaml() << std::endl;
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"1.0");

  // single bin, pdf should be 1.0 ?
  // TODO: count does not need a field!, but then assoc is ambiguous
  expr = "binning('field', 'count', [axis('x', [-10, 10])])";
  res = eval.evaluate(expr);
  std::cout << expr << std::endl << res.to_yaml() << std::endl;
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"8.0");

  // -10 --  0 : =  0 + 2 + 4 + 6
  //   0 -- 10 : =  1 + 3 + 5 + 7
  expr = "binning('field', 'sum', [axis('x', [-10, 0, 10])])";
  res = eval.evaluate(expr);
  std::cout << expr << std::endl << res.to_yaml() << std::endl;
  EXPECT_EQ(res["attrs/value/value"].to_yaml(), "[12.0, 16.0]");
  std::cout << casebreak << std::endl;

  // -------------
  // clamp = False
  expr = "binning('field', 'max', [axis('z', [-4, 0, 4], clamp=True)])";
  res = eval.evaluate(expr);
  std::cout << expr << std::endl << res.to_yaml() << std::endl;
  // default uncovered is 0.0
  EXPECT_EQ(res["attrs/value/value"].to_yaml(), "[3.0, 7.0]");
  std::cout << casebreak << std::endl;
  // -------------

  // -------------
  // clamp = False
  expr = "binning('field', 'max', [axis('z', [-4, 0, 4])])";
  res = eval.evaluate(expr);
  std::cout << expr << std::endl << res.to_yaml() << std::endl;
  // default uncovered is 0.0
  EXPECT_EQ(res["attrs/value/value"].to_yaml(), "[0.0, 0.0]");
  std::cout << casebreak << std::endl;
  // -------------

  // -------------
  // clamp = False, totally out of range
  expr = "binning('field', 'min', [axis('z', [-100, -50, -25], clamp=False)],empty_bin_val=-42)";
  res = eval.evaluate(expr);
  std::cout << expr << std::endl << res.to_yaml() << std::endl;
  EXPECT_EQ(res["attrs/value/value"].to_yaml(), "[-42.0, -42.0]");
  std::cout << casebreak << std::endl;
  // -------------

  // -------------
  expr = "binning('field', 'max', [axis('z', [-5, 0, 1], clamp=True)])";
  res = eval.evaluate(expr);
  std::cout << expr << std::endl << res.to_yaml() << std::endl;
  EXPECT_EQ(res["attrs/value/value"].to_yaml(), "[3.0, 7.0]");
  std::cout << casebreak << std::endl;
  // -------------

  // -------------
  expr = "binning('field', 'max', [axis('z', [-5, 0, 1], clamp=False)])";
  res = eval.evaluate(expr);
  std::cout << expr << std::endl << res.to_yaml() << std::endl;
  EXPECT_EQ(res["attrs/value/value"].to_yaml(), "[3.0, 0.0]");
  std::cout << casebreak << std::endl;
  // -------------

  // -------------
  expr =
      "binning('field', 'max', [axis('x', num_bins=2), axis('y', num_bins=2)], "
      "empty_bin_val=100)";
  res = eval.evaluate(expr);
  std::cout << expr << std::endl << res.to_yaml() << std::endl;

  EXPECT_EQ(res["attrs/value/value"].to_yaml(),"[4.0, 5.0, 6.0, 7.0]");
  std::cout << casebreak << std::endl;
  // -------------

  // -------------
  expr =
      "binning('field', 'sum', [axis('x', num_bins=2), axis('y', num_bins=2), "
      "axis('z', num_bins=2)])";
  res = eval.evaluate(expr);
  std::cout << expr << std::endl << res.to_yaml() << std::endl;
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),
            "[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]");
  std::cout << casebreak << std::endl;
  // -------------

  // -------------
  expr = "binning('ones', 'count', [axis('field', num_bins=8, clamp=True)])";
  res = eval.evaluate(expr);
  std::cout << expr << std::endl << res.to_yaml() << std::endl;
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),
            "[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]");
  std::cout << casebreak << std::endl;
  // -------------

  // -------------
  expr = "binning('ones', 'pdf', [axis('field', num_bins=8, clamp=True)])";
  res = eval.evaluate(expr);
  std::cout << expr << std::endl << res.to_yaml() << std::endl;
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),
            "[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]");
  std::cout << casebreak << std::endl;
  // -------------

  // -------------
  expr = "binning('ones', 'pdf', [axis('x', num_bins=2), axis('y', "
         "num_bins=2), axis('z', num_bins=2)])";
  res = eval.evaluate(expr);
  std::cout << expr << std::endl << res.to_yaml() << std::endl;
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),
             "[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]");
  std::cout << casebreak << std::endl;
  // -------------

  // -------------
  expr = "binning('ones', 'pdf', [axis('x', num_bins=2), axis('y', "
         "num_bins=2), axis('z',[-10, 10, 20], clamp=False)], empty_bin_val=0)";
  res = eval.evaluate(expr);
  std::cout << expr << std::endl << res.to_yaml() << std::endl;
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),
             "[0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0]");
  std::cout << casebreak << std::endl;
  // -------------

  // -------------
  expr = "binning('ones', 'count', [axis('x', num_bins=2), axis('y', "
         "num_bins=2), axis('z',[-10, 10, 20], clamp=False)], empty_bin_val=0)";
  res = eval.evaluate(expr);
  std::cout << expr << std::endl << res.to_yaml() << std::endl;
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),
             "[2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0]");
  std::cout << casebreak << std::endl;
  // -------------


  // -------------
  expr = "binning('ones', 'count', [axis('x', num_bins=2), axis('y', "
         "num_bins=2), axis('z',[-4, 10, 20], clamp=False)], empty_bin_val=0)";
  res = eval.evaluate(expr);
  std::cout << expr << std::endl << res.to_yaml() << std::endl;
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),
             "[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]");
  std::cout << casebreak << std::endl;
  // -------------

  // -------------
  expr = "binning('ones', 'count', [axis('x', num_bins=2), axis('y', "
         "num_bins=2), axis('z',[-25, -15, 20], clamp=False)], empty_bin_val=0)";
  res = eval.evaluate(expr);
  std::cout << expr << std::endl << res.to_yaml() << std::endl;
  EXPECT_EQ(res["attrs/value/value"].to_yaml(),
             "[0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0]");
  std::cout << casebreak << std::endl;
  // -------------

}

TEST(ascent_binning, binning_errors)
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

  //
  // Create an example mesh.
  //
  Node data, verify_info;
  conduit::blueprint::mesh::examples::braid("hexs",
                                            EXAMPLE_MESH_SIDE_DIM,
                                            EXAMPLE_MESH_SIDE_DIM,
                                            EXAMPLE_MESH_SIDE_DIM,
                                            data);
  // ascent normally adds this but we are doing an end around
  data["state/domain_id"] = 0;
  Node multi_dom;
  blueprint::mesh::to_multi_domain(data, multi_dom);

  runtime::expressions::register_builtin();
  runtime::expressions::ExpressionEval eval(&multi_dom);

  conduit::Node res;
  std::string expr;

  bool threw = false;
  try
  {
    expr = "binning('', 'avg', [axis('x'), axis('y')])";
    res = eval.evaluate(expr);
  }
  catch(...)
  {
    threw = true;
  }
  EXPECT_EQ(threw, true);

  threw = false;
  try
  {
    expr = "binning('braid', 'sum', [axis('x'), axis('vel')])";
    res = eval.evaluate(expr);
  }
  catch(...)
  {
    threw = true;
  }
  EXPECT_EQ(threw, true);

  threw = false;
  try
  {
    expr = "binning('vel', 'sum', [axis('x'), axis('y')])";
    res = eval.evaluate(expr);
  }
  catch(...)
  {
    threw = true;
  }
  EXPECT_EQ(threw, true);

  threw = false;
  try
  {
    expr = "binning('braid', 'sum', [axis('x', bins=[1,2], num_bins=1), "
           "axis('y')])";
    res = eval.evaluate(expr);
  }
  catch(...)
  {
    threw = true;
  }
  EXPECT_EQ(threw, true);

  threw = false;
  try
  {
    expr = "binning('braid', 'sum', [axis('x', bins=[1]), axis('y')])";
    res = eval.evaluate(expr);
  }
  catch(...)
  {
    threw = true;
  }
  EXPECT_EQ(threw, true);
}

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
  conduit::blueprint::mesh::examples::braid("hexs", 20, 20, 20, data);

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
TEST(ascent_binning, filter_braid_non_spatial_bins)
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
      conduit::utils::join_file_path(output_path, "tout_binning_braid_non_spatial");

  remove_test_image(output_file);
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

  // now add binning
  // pipeline 2
  pipelines["pl1/f2/type"] = "binning";
  // filter knobs
  conduit::Node &params = pipelines["pl1/f2/params"];
  params["reduction_op"] = "sum";
  params["reduction_field"] = "braid";
  params["output_field"] = "binning";
  // reduced dataset of only the bins
  params["output_type"] = "bins";

  conduit::Node &axis0 = params["axes"].append();
  axis0["field"] = "radial";
  axis0["num_bins"] = 10;
  axis0["clamp"] = 0;

  conduit::Node &axis1 = params["axes"].append();
  axis1["field"] = "braid";
  axis1["num_bins"] = 10;
  axis1["clamp"] = 0;

  conduit::Node scenes;
  scenes["s1/plots/p1/type"] = "pseudocolor";
  scenes["s1/plots/p1/field"] = "binning";
  scenes["s1/plots/p1/pipeline"] = "pl1";
  scenes["s1/image_prefix"] = output_file;

  conduit::Node extracts;
  extracts["e1/type"]  = "relay";
  extracts["e1/pipeline"] = "pl1";
  extracts["e1/params/protocol"] = "hdf5";
  extracts["e1/params/path"] = output_file + "_extract";

  conduit::Node actions;
  // add the pipeline
  conduit::Node &add_pipelines= actions.append();
  add_pipelines["action"] = "add_pipelines";
  add_pipelines["pipelines"] = pipelines;
  // add the scenes
  conduit::Node &add_scenes= actions.append();
  add_scenes["action"] = "add_scenes";
  add_scenes["scenes"] = scenes;

  conduit::Node &add_extracts= actions.append();
  add_extracts["action"] = "add_extracts";
  add_extracts["extracts"] = extracts;

  std::cout << actions.to_yaml() << std::endl;

  //
  // Run Ascent
  //

  Ascent ascent;
  ascent.open();
  ascent.publish(data);
  ascent.execute(actions);
  ascent.close();

  EXPECT_TRUE(check_test_image(output_file, 0.1));

  std::string msg = "An example of data binning, non-spatial binning and summing a field.";

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
