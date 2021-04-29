//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-716457
//
// All rights reserved.
//
// This file is part of Ascent.
//
// For details, see: http://ascent.readthedocs.io/.
//
// Please also read ascent/LICENSE
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_expressions.cpp
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
  //if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
  //{
  //  ASCENT_INFO("Ascent support disabled, skipping test");
  //  return;
  //}

  //
  // Create an example mesh.
  //
  Node data, verify_info;

  conduit::blueprint::mesh::examples::basic("hexs", 3, 3, 3, data);
  std::cout<<data.to_summary_string()<<"\n";
  data.print();

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

  //expr = "binning('field', 'sum', [axis('x', [0, 2.5, 5, 7.5, 10])])";
  //expr = "binning('field', 'sum', [axis('x', num_bins=5)])";
  expr = "binning('field', 'sum', [axis('field', num_bins=5), axis('x', num_bins=5)])";
  res = eval.evaluate(expr);
  res.print();
  //EXPECT_EQ(res["attrs/value/value"].to_json(), "[0.0, 0.0, 16.0, 0.0]");

  //expr = "binning('field', 'max', [axis('z', [-5, 0, 5])])";
  //res = eval.evaluate(expr);
  //EXPECT_EQ(res["attrs/value/value"].to_json(), "[3.0, 0.0]");

  //expr = "binning('field', 'max', [axis('z', [-5, 0, 5], clamp=True)])";
  //res = eval.evaluate(expr);
  //EXPECT_EQ(res["attrs/value/value"].to_json(), "[3.0, 7.0]");

  //expr =
  //    "binning('field', 'max', [axis('x', num_bins=4), axis('y', num_bins=4)], "
  //    "empty_bin_val=100)";
  //res = eval.evaluate(expr);
  //EXPECT_EQ(res["attrs/value/value"].to_json(),
  //          "[4.0, 100.0, 5.0, 100.0, 100.0, 100.0, 100.0, 100.0, 6.0, 100.0, "
  //          "7.0, 100.0, 100.0, 100.0, 100.0, 100.0]");

  //expr =
  //    "binning('field', 'sum', [axis('x', num_bins=2), axis('y', num_bins=2), "
  //    "axis('z', num_bins=2)])";
  //res = eval.evaluate(expr);
  //EXPECT_EQ(res["attrs/value/value"].to_json(),
  //          "[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]");

  //expr = "binning('', 'pdf', [axis('field', num_bins=8)])";
  //res = eval.evaluate(expr);
  //EXPECT_EQ(res["attrs/value/value"].to_json(),
  //          "[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]");

  //expr = "binning('field', 'pdf', [axis('x', num_bins=2), axis('y', "
  //       "num_bins=2), axis('z', num_bins=2)])";
  //res = eval.evaluate(expr);
  //EXPECT_EQ(res["attrs/value/value"].to_json(),
  //          "[0.0, 0.0357142857142857, 0.0714285714285714, 0.107142857142857, "
  //          "0.142857142857143, 0.178571428571429, 0.214285714285714, 0.25]");
}
#if 0
//-----------------------------------------------------------------------------
void
output_mesh(const conduit::Node &mesh, const std::string &output_file)
{
  // remove old images before rendering
  remove_test_image(output_file);

  conduit::Node extracts;
  extracts["e1/type"] = "relay";

  extracts["e1/params/path"] = output_file;
  extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

  conduit::Node actions;
  // add the extracts
  conduit::Node &add_extracts = actions.append();
  add_extracts["action"] = "add_extracts";
  add_extracts["extracts"] = extracts;

  conduit::Node &execute = actions.append();
  execute["action"] = "execute";

  //
  // Run Ascent
  //

  Ascent ascent;

  Node ascent_opts;
  ascent_opts["runtime"] = "ascent";
  ascent.open(ascent_opts);
  ascent.publish(mesh);
  ascent.execute(actions);
  ascent.close();
}

void
output_pseudocolor(const conduit::Node &mesh,
                   const std::string &field,
                   const std::string &output_file)
{
  // remove old images before rendering
  remove_test_image(output_file);

  //
  // Create the actions.
  //

  conduit::Node scenes;
  scenes["s1/plots/p1/type"] = "pseudocolor";
  scenes["s1/plots/p1/field"] = field;
  scenes["s1/renders/r1/image_prefix"] = output_file;

  conduit::Node actions;
  conduit::Node &add_plots = actions.append();
  add_plots["action"] = "add_scenes";
  add_plots["scenes"] = scenes;

  //
  // Run Ascent
  //

  Ascent ascent;

  Node ascent_opts;
  // ascent_opts["ascent_info"] = "verbose";
  ascent_opts["timings"] = "enabled";
  ascent_opts["runtime/type"] = "ascent";
  ascent.open(ascent_opts);
  ascent.publish(mesh);
  ascent.execute(actions);
  ascent.close();
}

TEST(ascent_binning, braid_binning)
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
  conduit::blueprint::mesh::examples::braid("hexs", 20, 20, 20, data);
  // ascent normally adds this but we are doing an end around
  data["state/domain_id"] = 0;
  Node multi_dom;
  blueprint::mesh::to_multi_domain(data, multi_dom);

  runtime::expressions::register_builtin();
  runtime::expressions::ExpressionEval eval(&multi_dom);

  std::string expr;

  string output_path = prepare_output_dir();

  expr = "binning('braid', 'sum', [axis('x', num_bins=21), axis('y', "
         "num_bins=21)], output='bins')";
  eval.evaluate(expr);
  expr = "binning('braid', 'std', [axis('x', num_bins=10), axis('y', "
         "num_bins=10)], output='mesh')";
  eval.evaluate(expr);

  std::string output_file =
      conduit::utils::join_file_path(output_path, "tout_binning_braid_xysum");
  output_pseudocolor(multi_dom, "braid_sum", output_file);
  EXPECT_TRUE(check_test_image(output_file, 0.1));

  output_file = conduit::utils::join_file_path(
      output_path, "tout_binning_painted_braid_xystd");
  output_pseudocolor(multi_dom, "painted_braid_std", output_file);
  EXPECT_TRUE(check_test_image(output_file, 0.1));
}

TEST(ascent_binning, multi_dom_binning)
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
  conduit::blueprint::mesh::examples::spiral(5, data);
  Node multi_dom;
  blueprint::mesh::to_multi_domain(data, multi_dom);
  // ascent normally adds this but we are doing an end around
  for(int i = 0; i < multi_dom.number_of_children(); ++i)
  {
    multi_dom.child(i)["state/cycle"] = 100;
    multi_dom.child(i)["state/time"] = 1.2;
    multi_dom.child(i)["state/domain_id"] = 0;
  }

  runtime::expressions::register_builtin();
  runtime::expressions::ExpressionEval eval(&multi_dom);

  std::string expr;
  expr = "binning('dist', 'std', [axis('x', num_bins=6), axis('y', "
         "num_bins=9)], output='bins')";
  eval.evaluate(expr);
  expr = "binning('dist', 'std', [axis('x', num_bins=6), axis('y', "
         "num_bins=9)], output='mesh')";
  eval.evaluate(expr);

  string output_path = prepare_output_dir();

  std::string output_file =
      conduit::utils::join_file_path(output_path, "tout_binning_dist_xystd");
  output_pseudocolor(multi_dom, "dist_std", output_file);
  EXPECT_TRUE(check_test_image(output_file, 0.01));

  output_file = conduit::utils::join_file_path(
      output_path, "tout_binning_painted_dist_xystd");
  output_pseudocolor(multi_dom, "painted_dist_std", output_file);
  EXPECT_TRUE(check_test_image(output_file, 0.01));
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
#endif
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
