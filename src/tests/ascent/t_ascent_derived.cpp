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
/// file: t_ascent_render_3d.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <ascent_expression_eval.hpp>
#include <ascent_hola.hpp>

#include <cmath>
#include <iostream>

#include <conduit_blueprint.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"

using namespace std;
using namespace conduit;
using namespace ascent;

index_t EXAMPLE_MESH_SIDE_DIM = 20;
#if 0
TEST(ascent_jit_expressions, derived_support_test)
{

  Node data;
  conduit::blueprint::mesh::examples::braid("uniform",
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

  expr = "builtin_avg = avg(sin(field('radial')))\n"
         "num_elements = sum(derived_field(1.0, 'mesh', 'element'))\n"
         "manual_avg = sum(sin(field('radial'))) / num_elements\n"
         "builtin_avg == manual_avg";

  bool threw = false;
  try
  {
    res = eval.evaluate(expr);
  }
  catch(...)
  {
    threw = true;
  }

  Node n;
  ascent::about(n);
  if(n["runtimes/ascent/jit/status"].as_string() == "disabled")
  {
    EXPECT_TRUE(threw);
  }
  else
  {
    EXPECT_FALSE(threw);
  }
}

TEST(ascent_expressions, derived_simple)
{
  Node n;
  ascent::about(n);

  // only run this test if ascent was built with jit support
  if(n["runtimes/ascent/jit/status"].as_string() == "disabled")
  {
      ASCENT_INFO("Ascent JIT support disabled, skipping test\n");
      return;
  }

  Node data;
  conduit::blueprint::mesh::examples::braid("uniform",
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

  //expr = "avg(topo('mesh').cell.x)";
  //res = eval.evaluate(expr);
  //const double tiny = 1e-10;
  //std::cout<<std::abs(res["value"].to_float64())<<"\n";
  //EXPECT_EQ(std::abs(res["value"].to_float64()) < tiny, true);
  //EXPECT_EQ(res["type"].as_string(), "double");

  //expr = "min_val = min(field('braid')).value\n"
  //       "max_val = max(field('braid')).value\n"
  //       "norm_field = (field('braid') - min_val) / (max_val - min_val)\n"
  //       "not_between_0_1 = not (norm_field >= 0 and norm_field <= 1)\n"
  //       "sum(not_between_0_1)";


  //res = eval.evaluate(expr, "bananas");
  //res = eval.evaluate(expr);
  //EXPECT_EQ(res["value"].to_float64(), 0);
  //EXPECT_EQ(res["type"].as_string(), "double");

  //expr = "bananas = field('braid') + 1\n";
  //       "bananas + 1";
  //res = eval.evaluate(expr);

  expr = "builtin_avg = avg(sin(field('radial')))\n"
         "num_elements = sum(derived_field(1.0, 'mesh', 'element'))\n"
         "manual_avg = sum(sin(field('radial'))) / num_elements\n"
         "builtin_avg == manual_avg";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["type"].as_string(), "bool");

  Node last;
  runtime::expressions::ExpressionEval::get_last(last);
  double manual = last["manual_avg/100/value"].to_float64();
  double builtin = last["builtin_avg/100/value"].to_float64();
  EXPECT_NEAR(manual, builtin, 1e-8);
}

//-----------------------------------------------------------------------------
TEST(ascent_expressions, derived_expressions)
{
  Node n;
  ascent::about(n);
  // only run this test if ascent was built with jit support
  if(n["runtimes/ascent/jit/status"].as_string() == "disabled")
  {
      ASCENT_INFO("Ascent JIT support disabled, skipping test\n");
      return;
  }

  //
  // Create an example mesh.
  //
  Node data;
  conduit::blueprint::mesh::examples::braid("uniform",
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

  // expr = "1 + field('braid') + 1";
  //    +
  //  /   \
  //  1   + return_type: jitable
  //     / \
  // field  1
  //
  // field + field + 1
  //
  //       +
  //     /   \
  // field   + return_type: jitable
  //        / \
  //    field  1
  //
  // max(field + 1)
  //
  //     max
  //      |
  //      + jitable -> field
  //     / \
  // field  1
  //

  expr = "f1 = max((2.0 + 1) / 0.5 + field('braid'), 0.0)\n"
         "sum(f1 < 0)";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].to_float64(), 0);
  EXPECT_EQ(res["type"].as_string(), "double");

  expr = "method1 = field('braid') + 1 + field('braid') + 1\n"
         "method2 = 2 * field('braid') + 2\n"
         "bool_field = method1 != method2\n"
         "sum(bool_field) == 0";

  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].to_uint8(), 1);
  EXPECT_EQ(res["type"].as_string(), "bool");

  // errors
  // scalar * vector fields
  bool threw = false;
  try
  {
    expr = "field('braid') * field('vel')";
    eval.evaluate(expr);
  }
  catch(...)
  {
    threw = true;
  }
  EXPECT_EQ(threw, true);

  // incompatible number of entries
  threw = false;
  try
  {
    expr = "field('braid') * field('radial')";
    eval.evaluate(expr);
  }
  catch(...)
  {
    threw = true;
  }
  EXPECT_EQ(threw, true);

  // Run on multiple meshes
  const std::vector<std::string> mesh_types = {
      "uniform", "rectilinear", "structured", "tris", "quads", "hexs", "tets"};
  const std::vector<std::vector<long long>> mesh_dims = {
      {EXAMPLE_MESH_SIDE_DIM, EXAMPLE_MESH_SIDE_DIM, EXAMPLE_MESH_SIDE_DIM},
      {EXAMPLE_MESH_SIDE_DIM, EXAMPLE_MESH_SIDE_DIM, 0}};

  for(const std::string &mesh_type : mesh_types)
  {
    for(const auto &dims : mesh_dims)
    {
      if(((mesh_type == "hexs" || mesh_type == "tets") && dims[2] == 0) ||
         (mesh_type == "quads" && dims[2] != 0))
      {
        continue;
      }
      Node data;
      conduit::blueprint::mesh::examples::braid(
          mesh_type, dims[0], dims[1], dims[2], data);

      // ascent normally adds this but we are doing an end around
      data["state/domain_id"] = 0;
      Node multi_dom;
      blueprint::mesh::to_multi_domain(data, multi_dom);

      runtime::expressions::register_builtin();
      runtime::expressions::ExpressionEval eval(&multi_dom);

      conduit::Node res;
      std::string expr;

      expr = "min_val = min(field('braid')).value\n"
             "max_val = max(field('braid')).value\n"
             "norm_field = (field('braid') - min_val) / (max_val - min_val)\n"
             "not_between_0_1 = not (norm_field >= 0 and norm_field <= 1)\n"
             "sum(not_between_0_1)";
      res = eval.evaluate(expr);
      EXPECT_EQ(res["value"].to_float64(), 0);
      EXPECT_EQ(res["type"].as_string(), "double");

      expr = "avg(topo('mesh').cell.x)";
      res = eval.evaluate(expr);
      const double tiny = 1e-10;
      EXPECT_EQ(std::abs(res["value"].to_float64()) < tiny, true);
      EXPECT_EQ(res["type"].as_string(), "double");

      expr = "builtin_avg = avg(sin(field('radial')))\n"
             "num_elements = sum(derived_field(1.0, 'mesh', 'element'))\n"
             "manual_avg = sum(sin(field('radial'))) / num_elements\n"
             "builtin_avg == manual_avg";
      res = eval.evaluate(expr);

      EXPECT_EQ(res["type"].as_string(), "bool");

      Node last;
      runtime::expressions::ExpressionEval::get_last(last);
      double manual = last["manual_avg/100/value"].to_float64();
      double builtin = last["builtin_avg/100/value"].to_float64();
      EXPECT_NEAR(manual, builtin, 1e-8);

      // apparently vel is element assoc
      if(dims[2] != 0 && (mesh_type == "uniform" || mesh_type == "rectilinear"))
      {
        expr = "builtin_vort = curl(field('vel'))\n"
               "du = gradient(field('vel', 'u'))\n"
               "dv = gradient(field('vel', 'v'))\n"
               "dw = gradient(field('vel', 'w'))\n"
               "w_x = dw.y - dv.z\n"
               "w_y = du.z - dw.x\n"
               "w_z = dv.x - du.y\n"
               "not_eq = not (builtin_vort.x == w_x and\n"
               "              builtin_vort.y == w_y and\n"
               "              builtin_vort.z == w_z)\n"
               "sum(not_eq)";
        res = eval.evaluate(expr);
        EXPECT_EQ(res["value"].to_float64(), 0);
        EXPECT_EQ(res["type"].as_string(), "double");
      }

      if(dims[2] == 0)
      {
        expr = "topo('mesh').cell.area";
        eval.evaluate(expr);

        expr = "topo('mesh').cell.volume";
        EXPECT_ANY_THROW(eval.evaluate(expr));
      }
      else
      {
        // expr = "topo('mesh').cell.area";
        // EXPECT_ANY_THROW(eval.evaluate(expr));

        // expr = "topo('mesh').cell.volume";
        // eval.evaluate(expr);
      }

      if(mesh_type == "uniform" || mesh_type == "rectilinear" ||
         mesh_type == "structured")
      {
        // element to vertex
        expr = "recenter(field('radial') + 1)";
        eval.evaluate(expr);
      }

      // vertex to element
      expr = "recenter(field('braid'))";
      eval.evaluate(expr);
    }
  }
}
//-----------------------------------------------------------------------------
#endif

TEST(ascent_expressions, braid_sample)
{
  conduit::Node data, multi_dom;
  conduit::blueprint::mesh::examples::braid("structured",
                                            EXAMPLE_MESH_SIDE_DIM,
                                            EXAMPLE_MESH_SIDE_DIM,
                                            EXAMPLE_MESH_SIDE_DIM,
                                            data);
  data["state/domain_id"] = 0;
  blueprint::mesh::to_multi_domain(data, multi_dom);


  const std::string output_path = prepare_output_dir();

  std::string output_image =
      conduit::utils::join_file_path(output_path, "tout_half_braid_");

  conduit::Node actions;

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] = "expression";
  // filter knobs
  conduit::Node &expr_params = pipelines["pl1/f1/params"];
  expr_params["expression"] = "field('braid') / 2.0";
  expr_params["name"] = "half";

  conduit::Node &add_pipelines = actions.append();
  add_pipelines["action"] = "add_pipelines";
  add_pipelines["pipelines"] = pipelines;

  conduit::Node scenes;
  scenes["s1/plots/p1/type"] = "pseudocolor";
  scenes["s1/plots/p1/field"] = "half";
  scenes["s1/plots/p1/pipeline"] = "pl1";
  scenes["s1/renders/r1/image_prefix"] = output_image;

  conduit::Node &add_plots = actions.append();
  add_plots["action"] = "add_scenes";
  add_plots["scenes"] = scenes;

  //
  // Run Ascent
  //

  Ascent ascent;

  Node ascent_opts;
  ascent_opts["ascent_info"] = "verbose";
  ascent_opts["timings"] = "enabled";
  ascent_opts["runtime/type"] = "ascent";

  ascent.open(ascent_opts);
  ascent.publish(multi_dom);
  ascent.execute(actions);
  ascent.close();

  EXPECT_TRUE(check_test_image(output_image, 0.1));
}

//-----------------------------------------------------------------------------

int
main(int argc, char *argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  // allow override of the data size via the command line
  if(argc == 2)
  {
    EXAMPLE_MESH_SIDE_DIM = atoi(argv[1]);
  }

  result = RUN_ALL_TESTS();
  return result;
}
