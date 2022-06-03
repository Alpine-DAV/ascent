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

  expr = "binning('field', 'sum', [axis('x', [0, 2.5, 5, 7.5, 10])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_json(), "[0.0, 0.0, 16.0, 0.0]");

  expr = "binning('field', 'max', [axis('z', [-5, 0, 5])])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_json(), "[3.0, 7.0]");

  expr = "binning('field', 'max', [axis('z', [-5, 0, 1], clamp=True)])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_json(), "[3.0, 7.0]");

  expr =
      "binning('field', 'max', [axis('x', num_bins=4), axis('y', num_bins=4)], "
      "empty_bin_val=100)";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_json(),
            "[4.0, 100.0, 5.0, 100.0, 100.0, 100.0, 100.0, 100.0, 6.0, 100.0, "
            "7.0, 100.0, 100.0, 100.0, 100.0, 100.0]");

  expr =
      "binning('field', 'sum', [axis('x', num_bins=2), axis('y', num_bins=2), "
      "axis('z', num_bins=2)])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_json(),
            "[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]");

  expr = "binning('', 'pdf', [axis('field', num_bins=8)])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_json(),
            "[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]");

  expr = "binning('field', 'pdf', [axis('x', num_bins=2), axis('y', "
         "num_bins=2), axis('z', num_bins=2)])";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].to_json(),
            "[0.0, 0.0357142857142857, 0.0714285714285714, 0.107142857142857, "
            "0.142857142857143, 0.178571428571429, 0.214285714285714, 0.25]");
}

TEST(ascent_binning, binning_errors)
{
  
  /// <<<<<<<<<< TODO FIX!
  return;
  
  
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
  params["var"] = "braid";
  params["output_field"] = "binning";
  // paint the field onto the original mesh
  params["output_type"] = "mesh";

  conduit::Node &axis0 = params["axes"].append();
  axis0["var"] = "x";
  axis0["num_bins"] = 10;
  axis0["min_val"] = -10.0;
  axis0["max_val"] = 10.0;
  axis0["clamp"] = 1;

  conduit::Node &axis1 = params["axes"].append();
  axis1["var"] = "y";
  axis1["num_bins"] = 10;
  axis1["clamp"] = 0;

  conduit::Node &axis2 = params["axes"].append();
  axis2["var"] = "z";
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

  Node ascent_opts;
  ascent_opts["runtime/type"] = "ascent";
  ascent.open(ascent_opts);
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
  params["var"] = "braid";
  params["output_field"] = "binning";
  // reduced dataset of only the bins
  params["output_type"] = "bins";

  conduit::Node &axis0 = params["axes"].append();
  axis0["var"] = "x";
  axis0["num_bins"] = 10;
  axis0["min_val"] = -10.0;
  axis0["max_val"] = 10.0;
  axis0["clamp"] = 1;

  conduit::Node &axis1 = params["axes"].append();
  axis1["var"] = "y";
  axis1["num_bins"] = 10;
  axis1["clamp"] = 0;

  conduit::Node &axis2 = params["axes"].append();
  axis2["var"] = "z";
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

  Node ascent_opts;
  ascent_opts["runtime/type"] = "ascent";
  ascent.open(ascent_opts);
  ascent.publish(data);
  ascent.execute(actions);
  ascent.close();

  EXPECT_TRUE(check_test_image(output_file, 0.1));
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
