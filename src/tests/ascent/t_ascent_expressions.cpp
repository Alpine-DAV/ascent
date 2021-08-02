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
TEST(ascent_expressions, basic_expressions)
{
  Node n;
  ascent::about(n);

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

  expr = "(2.0 + 1) / 0.5";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].to_float64(), 6.0);
  EXPECT_EQ(res["type"].as_string(), "double");

  expr = "(2.0 * 2) / 2";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].to_float64(), 2.0);
  EXPECT_EQ(res["type"].as_string(), "double");

  expr = "2.0 + 1 / 0.5";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].to_float64(), 4.0);
  EXPECT_EQ(res["type"].as_string(), "double");

  expr = "5e-2 / .5";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].to_float64(), 0.1);
  EXPECT_EQ(res["type"].as_string(), "double");

  expr = "8 % 3";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].to_int32(), 2);
  EXPECT_EQ(res["type"].as_string(), "int");

  expr = "abs(-1)";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].to_float64(), 1);
  EXPECT_EQ(res["type"].as_string(), "int");

  expr = "abs(-1.0)";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].to_float64(), 1);
  EXPECT_EQ(res["type"].as_string(), "double");

  expr = "exp(1.0)";
  res = eval.evaluate(expr);
  EXPECT_NEAR(res["value"].to_float64(), 2.71828, 0.0001);
  EXPECT_EQ(res["type"].as_string(), "double");

  expr = "pow(2.0,2)";
  res = eval.evaluate(expr);
  EXPECT_NEAR(res["value"].to_float64(), 4.0, 0.0001);
  EXPECT_EQ(res["type"].as_string(), "double");

  expr = "log(3)";
  res = eval.evaluate(expr);
  EXPECT_NEAR(res["value"].to_float64(), 1.0986122886681098, 0.0001);
  EXPECT_EQ(res["type"].as_string(), "double");

  bool threw = false;
  try
  {
    expr = "4 % 2.5";
    res = eval.evaluate(expr);
  }
  catch(...)
  {
    threw = true;
  }
  EXPECT_EQ(threw, true);

  expr = "True and not False";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].to_uint8(), 1);
  EXPECT_EQ(res["type"].as_string(), "bool");

  expr = "2.5 >= 2";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].to_uint8(), 1);
  EXPECT_EQ(res["type"].as_string(), "bool");

  expr = "(1 == 1) and (3 <= 3)";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].to_uint8(), 1);
  EXPECT_EQ(res["type"].as_string(), "bool");

  expr = "(2.3 != 2.3) or (3 > 3)";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].to_uint8(), 0);
  EXPECT_EQ(res["type"].as_string(), "bool");

  expr = "not (55 < 59)";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].to_uint8(), 0);
  EXPECT_EQ(res["type"].as_string(), "bool");

  expr = "max(1, 2)";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].to_float64(), 2.0);
  EXPECT_EQ(res["type"].as_string(), "int");

  expr = "max(1, 2.0)";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].to_float64(), 2.0);
  EXPECT_EQ(res["type"].as_string(), "double");

  expr = "min(1, 2)";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].to_float64(), 1.0);
  EXPECT_EQ(res["type"].as_string(), "int");

  threw = false;
  try
  {
    expr = "(2.0 + 1 / 0.5";
    res = eval.evaluate(expr);
  }
  catch(...)
  {
    threw = true;
  }
  EXPECT_EQ(threw, true);

  expr = "vector(1.5,0,0)";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].to_float64(), 1.5); // test the first val
  EXPECT_EQ(res["type"].as_string(), "vector");

  expr = "vector(1.5,0,0) - vector(1,0,0)";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].to_float64(), 0.5); // test the first val
  EXPECT_EQ(res["type"].as_string(), "vector");

  expr = "magnitude(vector(2,0,0) - vector(0,0,0))";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].to_float64(), 2);
  EXPECT_EQ(res["type"].as_string(), "double");

  // currently unsupported vector ops
  threw = false;
  try
  {
    expr = "vector(1.5,0,0) * vector(1,0,0)";
    res = eval.evaluate(expr);
    EXPECT_EQ(res["value"].to_float64(), 0.5); // test the first val
    EXPECT_EQ(res["type"].as_string(), "vector");
  }
  catch(...)
  {
    threw = true;
  }
  EXPECT_EQ(threw, true);

  threw = false;
  try
  {
    expr = "vector(1.5,0,0) / vector(1,0,0)";
    res = eval.evaluate(expr);
    EXPECT_EQ(res["value"].to_float64(), 0.5); // test the first val
    EXPECT_EQ(res["type"].as_string(), "vector");
  }
  catch(...)
  {
    threw = true;
  }
  EXPECT_EQ(threw, true);

  expr = "cycle()";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].to_float64(), 100);
  EXPECT_EQ(res["type"].as_string(), "int");

  expr = "magnitude(max(field('braid')).position) > 0";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].to_uint8(), 1);
  EXPECT_EQ(res["type"].as_string(), "bool");
}

//-----------------------------------------------------------------------------
TEST(ascent_expressions, functional_correctness)
{
  Node n;
  ascent::about(n);

  //
  // Create an example mesh.
  //
  Node data;

  data["state/time"] = 1.0;
  // create the coordinate set
  data["coordsets/coords/type"] = "uniform";
  data["coordsets/coords/dims/i"] = 5;
  data["coordsets/coords/dims/j"] = 5;
  // add origin and spacing to the coordset (optional)
  data["coordsets/coords/origin/x"] = -10.0;
  data["coordsets/coords/origin/y"] = -10.0;
  data["coordsets/coords/spacing/dx"] = 10.0;
  data["coordsets/coords/spacing/dy"] = 10.0;

  // add the topology
  // this case is simple b/c it's implicitly derived from the coordinate set
  data["topologies/topo/type"] = "uniform";
  // reference the coordinate set by name
  data["topologies/topo/coordset"] = "coords";

  // add a simple element-associated field
  data["fields/ele_example/association"] = "element";
  // reference the topology this field is defined on by name
  data["fields/ele_example/topology"] = "topo";
  // set the field values, for this case we have 16 elements
  data["fields/ele_example/values"].set(DataType::float64(16));

  float64 *ele_vals_ptr = data["fields/ele_example/values"].value();

  for(int i = 0; i < 16; i++)
  {
    ele_vals_ptr[i] = float64(i);
  }

  // add a element-associated field  with nans
  data["fields/ele_nan_example/association"] = "element";
  // reference the topology this field is defined on by name
  data["fields/ele_nan_example/topology"] = "topo";
  // set the field values, for this case we have 9 elements
  data["fields/ele_nan_example/values"].set(DataType::float64(16));

  float64 *ele_nan_vals_ptr = data["fields/ele_nan_example/values"].value();

  for(int i = 0; i < 16; i++)
  {
    if(i == 0)
    {
      ele_nan_vals_ptr[i] = std::nan("");
    }
    else if(i == 1)
    {
      ele_nan_vals_ptr[i] = -1. / 0.;
    }
    else
    {
      ele_nan_vals_ptr[i] = float64(i);
    }
  }

  data["state/cycle"] = 100;
  data["state/time"] = 2.1;

  // make sure we conform:
  Node verify_info;
  if(!blueprint::mesh::verify(data, verify_info))
  {
    std::cout << "Verify failed!" << std::endl;
    verify_info.print();
  }

  // ascent normally adds this but we are doing an end around
  data["state/domain_id"] = 0;
  Node multi_dom;
  blueprint::mesh::to_multi_domain(data, multi_dom);

  runtime::expressions::register_builtin();
  runtime::expressions::ExpressionEval eval(&multi_dom);

  conduit::Node res;
  std::string expr;

  expr = "max(field('ele_example')).position";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].as_float64(), 25);
  EXPECT_EQ(res["type"].as_string(), "vector");

  expr = "entropy(histogram(field('ele_example')))";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].as_float64(), -std::log(1.0 / 16.0));
  EXPECT_EQ(res["type"].as_string(), "double");

  expr = "bin(cdf(histogram(field('ele_example'))), val=5.0)";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].as_float64(), .375);
  EXPECT_EQ(res["type"].as_string(), "double");

  expr = "bin(pdf(histogram(field('ele_example'))), val=5)";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].as_float64(), 1.0 / 16.0);
  EXPECT_EQ(res["type"].as_string(), "double");

  expr = "bin(pdf(histogram(field('ele_example'))), val=4.5)";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].as_float64(), 0);
  EXPECT_EQ(res["type"].as_string(), "double");

  expr = "bin(pdf(histogram(field('ele_example'))), bin=0) == "
         "pdf(histogram(field('ele_example'))).value[0]";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].to_uint8(), 1);
  EXPECT_EQ(res["type"].as_string(), "bool");

  expr =
      "quantile(cdf(histogram(field('ele_example'), num_bins=240)), 3.0/16.0)";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].to_float64(), 2);
  EXPECT_EQ(res["type"].as_string(), "double");

  expr = "16.0/256 == avg(histogram(field('ele_example')).value)";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].to_uint8(), 1);
  EXPECT_EQ(res["type"].as_string(), "bool");

  expr = "16 == sum(histogram(field('ele_example')).value)";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].to_uint8(), 1);
  EXPECT_EQ(res["type"].as_string(), "bool");

  expr = "1 == field_nan_count(field('ele_nan_example'))";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].to_uint8(), 1);
  EXPECT_EQ(res["type"].as_string(), "bool");

  expr = "1 == field_inf_count(field('ele_nan_example'))";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].to_uint8(), 1);
  EXPECT_EQ(res["type"].as_string(), "bool");
}

//-----------------------------------------------------------------------------
TEST(ascent_expressions, expressions_named_params)
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

  // test named parameters

  std::string expr;
  conduit::Node res;

  expr = "histogram(field('braid'), num_bins=10)";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].dtype().number_of_elements(), 10);
  EXPECT_EQ(res["type"].as_string(), "histogram");

  expr = "histogram(field('braid'),min_val=0,num_bins=10,max_val=1)";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/value/value"].dtype().number_of_elements(), 10);
  EXPECT_EQ(res["attrs/min_val/value"].to_float64(), 0);
  EXPECT_EQ(res["attrs/max_val/value"].to_float64(), 1);
  EXPECT_EQ(res["type"].as_string(), "histogram");

  bool threw = false;
  try
  {
    expr = "histogram(field('braid'),field('braid'))";
    res = eval.evaluate(expr);
  }
  catch(...)
  {
    threw = true;
  }
  EXPECT_EQ(threw, true);

  expr = "histogram(field('braid'),max_val=2)";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["attrs/max_val/value"].to_float64(), 2);
  EXPECT_EQ(res["type"].as_string(), "histogram");

  threw = false;
  try
  {
    expr = "histogram(field('braid'),min_val=field('braid'))";
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
    expr = "histogram(field('braid'),min_val=0,num_bins=10,1)";
    res = eval.evaluate(expr);
  }
  catch(...)
  {
    threw = true;
  }
  EXPECT_EQ(threw, true);
}
//-----------------------------------------------------------------------------
TEST(ascent_expressions, test_identifier)
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
  std::string expr;
  conduit::Node res1, res2;

  // test retrieving named cached value
  expr = "max(field('braid'))";
  const std::string cache_name = "mx_b";
  res1 = eval.evaluate(expr, cache_name);
  res2 = eval.evaluate("mx_b");
  EXPECT_EQ(res1["value"].to_float64(), res2["value"].to_float64());

  // grab attribute from cached value
  res2 = eval.evaluate("mx_b.position");
  EXPECT_EQ(res2["type"].as_string(), "vector");
}

//-----------------------------------------------------------------------------
TEST(ascent_expressions, test_history)
{
  Node n;
  ascent::about(n);

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
  runtime::expressions::ExpressionEval::reset_cache();

  conduit::Node res;
  std::string expr;

  // we can't change the input object so keep
  // giving eval new ones
  {
    runtime::expressions::ExpressionEval eval(&multi_dom);
    res = eval.evaluate("1", "val");
    res = eval.evaluate("vector(1,2,3)", "vec");
  }

  {
    multi_dom.child(0)["state/cycle"] = 200;
    runtime::expressions::ExpressionEval eval(&multi_dom);
    res = eval.evaluate("2", "val");
    res = eval.evaluate("vector(9,3,4)", "vec");
  }

  {
    multi_dom.child(0)["state/cycle"] = 300;
    runtime::expressions::ExpressionEval eval(&multi_dom);
    res = eval.evaluate("3", "val");
    res = eval.evaluate("vector(3,4,0)", "vec");
  }

  {
    multi_dom.child(0)["state/cycle"] = 400;
    runtime::expressions::ExpressionEval eval(&multi_dom);
    res = eval.evaluate("4", "val");
    res = eval.evaluate("vector(6,4,8)", "vec");
  }

  expr = "history(val, absolute_index=2)";
  runtime::expressions::ExpressionEval eval(&multi_dom);
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].to_int32(), 3);
  EXPECT_EQ(res["type"].as_string(), "int");

  expr = "history(val, 3)";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].to_int32(), 1);
  EXPECT_EQ(res["type"].as_string(), "int");

  bool threw = false;
  try
  {
    expr = "history(val, absolute_index = 10)";
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
    expr = "history(vval, 1)";
    res = eval.evaluate(expr);
  }
  catch(...)
  {
    threw = true;
  }
  EXPECT_EQ(threw, true);

  expr = "history(vec, 2) - vector(1,1,1)";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].to_float64(), 8); // test the first val
  EXPECT_EQ(res["type"].as_string(), "vector");

  expr = "magnitude(history(vec, 1))";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].to_float64(), 5); // test the first val
  EXPECT_EQ(res["type"].as_string(), "double");
}

//-----------------------------------------------------------------------------
TEST(ascent_expressions, test_gradient_scalar)
{
  Node n;
  ascent::about(n);

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
  runtime::expressions::ExpressionEval::reset_cache();

  conduit::Node res;
  std::string expr;

  // we can't change the input object so keep
  // giving eval new ones
  {
    multi_dom.child(0)["state/cycle"] = 100;
    multi_dom.child(0)["state/time"] = 2.0;
    runtime::expressions::ExpressionEval eval(&multi_dom);
    res = eval.evaluate("1", "val");
    res = eval.evaluate("vector(1,2,3)", "vec");
    res = eval.evaluate("gradient(val)", "gradient_val");
  }

  {
    multi_dom.child(0)["state/cycle"] = 200;
    multi_dom.child(0)["state/time"] = 4.0;
    runtime::expressions::ExpressionEval eval(&multi_dom);
    res = eval.evaluate("2", "val");
    res = eval.evaluate("vector(9,3,4)", "vec");
    res = eval.evaluate("gradient(val)", "gradient_val");
  }

  {
    multi_dom.child(0)["state/cycle"] = 300;
    multi_dom.child(0)["state/time"] = 6.0;
    runtime::expressions::ExpressionEval eval(&multi_dom);
    res = eval.evaluate("3", "val");
    res = eval.evaluate("vector(3,4,0)", "vec");
    res = eval.evaluate("gradient(val)", "gradient_val");
  }

  {
    multi_dom.child(0)["state/cycle"] = 400;
    multi_dom.child(0)["state/time"] = 8.0;
    runtime::expressions::ExpressionEval eval(&multi_dom);
    res = eval.evaluate("4", "val");
    res = eval.evaluate("vector(6,4,8)", "vec");
    res = eval.evaluate("gradient(val)", "gradient_val");
  }
  runtime::expressions::ExpressionEval eval(&multi_dom);

  for(const string &expression : {
      "gradient(val)", 
      "gradient(val, window_length=1)",
      "gradient(val, window_length=1, window_length_unit='index')",
      "gradient(val, window_length=2)",
      "gradient(val, window_length=5)",
    }) {
    res = eval.evaluate(expression);
    EXPECT_EQ(res["value"].to_float64(), 1);
  }

  for(const string &expression : {
      "gradient(val, window_length=2, window_length_unit='time')",
      "gradient(val, window_length=3, window_length_unit='time')",
      "gradient(val, window_length=4, window_length_unit='time')",
      "gradient(val, window_length=10, window_length_unit='time')",
    }) {
    res = eval.evaluate(expression);
    EXPECT_EQ(res["value"].to_float64(), .5);
  }

  for(const string &expression : {
      "gradient(val, window_length=100, window_length_unit='cycle')",
      "gradient(val, window_length=150, window_length_unit='cycle')",
    }) {
    res = eval.evaluate(expression);
    EXPECT_EQ(res["value"].to_float64(), .01);
  }

  //add one more data point so we can evaluate the gradient of the gradient
  {
    multi_dom.child(0)["state/cycle"] = 500;
    multi_dom.child(0)["state/time"] = 12.0;
    runtime::expressions::ExpressionEval eval(&multi_dom);
    res = eval.evaluate("8", "val");
    res = eval.evaluate("gradient(val)", "gradient_val");
  }

  //test the gradient of gradient
  for(const string &expression : {
      "gradient(gradient_val)", 
    }) {
    res = eval.evaluate(expression);
    EXPECT_EQ(res["value"].to_float64(), 3);
  }
}

//-----------------------------------------------------------------------------
TEST(ascent_expressions, test_gradient_array)
{
  Node n;
  ascent::about(n);

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
  runtime::expressions::ExpressionEval::reset_cache();

  conduit::Node res;
  std::string expr;

  // we can't change the input object so keep
  // giving eval new ones
  {
    multi_dom.child(0)["state/cycle"] = 100;
    multi_dom.child(0)["state/time"] = 2.0;
    runtime::expressions::ExpressionEval eval(&multi_dom);
    res = eval.evaluate("1", "val");
    res = eval.evaluate("vector(1,2,3)", "vec");
    res = eval.evaluate("gradient(val, window_length=100, window_length_unit='cycle')", "gradient_val");   
  }

  {
    multi_dom.child(0)["state/cycle"] = 200;
    multi_dom.child(0)["state/time"] = 4.0;
    runtime::expressions::ExpressionEval eval(&multi_dom);
    res = eval.evaluate("2", "val");
    res = eval.evaluate("vector(9,3,4)", "vec");
    res = eval.evaluate("gradient(val, window_length=100, window_length_unit='cycle')", "gradient_val");   
  }

  {
    multi_dom.child(0)["state/cycle"] = 300;
    multi_dom.child(0)["state/time"] = 6.0;
    runtime::expressions::ExpressionEval eval(&multi_dom);
    res = eval.evaluate("3", "val");
    res = eval.evaluate("vector(3,4,0)", "vec");
    res = eval.evaluate("gradient(val, window_length=100, window_length_unit='cycle')", "gradient_val");   
  }

  {
    multi_dom.child(0)["state/cycle"] = 500;
    multi_dom.child(0)["state/time"] = 10.0;
    runtime::expressions::ExpressionEval eval(&multi_dom);
    res = eval.evaluate("4", "val");
    res = eval.evaluate("vector(6,4,8)", "vec");
    res = eval.evaluate("gradient(val, window_length=200, window_length_unit='cycle')", "gradient_val");   
  }

  runtime::expressions::ExpressionEval eval(&multi_dom);
  conduit::float64_array result;

  // confirm it works properly
  for(const string &expression : {
      "gradient_range(val, first_absolute_index=0, last_absolute_index=2)", 
      "gradient_range(val, first_relative_index=1, last_relative_index=3)",
    }) {
    res = eval.evaluate(expression);
    EXPECT_EQ(res["type"].as_string(), "array");
    result = res["value"].as_float64_array();
    EXPECT_EQ(result.to_json(), "[1.0, 1.0]");
  }

  for(const string &expression : {
      "gradient_range(val, first_absolute_time=2.0, last_absolute_time=6.0)",
      "gradient_range(val, first_absolute_time=1.0, last_absolute_time=7.0)",
    }) {
    res = eval.evaluate(expression);
    EXPECT_EQ(res["type"].as_string(), "array");
    result = res["value"].as_float64_array();
    EXPECT_EQ(result.to_json(), "[0.5, 0.5]");
  }

  for(const string &expression : {
      "gradient_range(val, first_absolute_cycle=100, last_absolute_cycle=300)",
    }) {
    res = eval.evaluate(expression);
    EXPECT_EQ(res["type"].as_string(), "array");
    result = res["value"].as_float64_array();
    EXPECT_EQ(result.to_json(), "[0.01, 0.01]");
  }

  for(const string &expression : {
      "gradient_range(val, first_absolute_time=4.0, last_absolute_time=10.0)",
    }) {
    res = eval.evaluate(expression);
    EXPECT_EQ(res["type"].as_string(), "array");
    result = res["value"].as_float64_array();
    EXPECT_EQ(result.to_json(), "[0.5, 0.25]");
  }

  // confirm it works properly if a single element is returned
  for(const string &expression : {
      "gradient_range(val, first_absolute_index=1, last_absolute_index=2)", 
      "gradient_range(val, first_relative_index=1, last_relative_index=2)",
    }) {
    res = eval.evaluate(expression);
    EXPECT_EQ(res["type"].as_string(), "array");
    result = res["value"].as_float64_array();
    EXPECT_EQ(result.to_json(), "1.0");
  }

  for(const string &expression : {
      "gradient_range(val, first_absolute_time=1.0, last_absolute_time=4.0)",
    }) {
    res = eval.evaluate(expression);
    EXPECT_EQ(res["type"].as_string(), "array");
    result = res["value"].as_float64_array();
    EXPECT_EQ(result.to_json(), "0.5");
  }

  for(const string &expression : {
      "gradient_range(val, first_absolute_cycle=200, last_absolute_cycle=300)",
    }) {
    res = eval.evaluate(expression);
    EXPECT_EQ(res["type"].as_string(), "array");
    result = res["value"].as_float64_array();
    EXPECT_EQ(result.to_json(), "0.01");
  }

  //confirm that it clamps to the end as expected
  for(const string &expression : {
      "gradient_range(val, first_absolute_index=1, last_absolute_index=5)",
    }) {
    res = eval.evaluate(expression);
    EXPECT_EQ(res["type"].as_string(), "array");
    result = res["value"].as_float64_array();
    EXPECT_EQ(result.to_json(), "[1.0, 1.0]");
  }

  //confirm that it clamps to the beginning as expected
  for(const string &expression : {
    "gradient_range(val, first_relative_index=1, last_relative_index=6)",
  }) {
    res = eval.evaluate(expression);
    EXPECT_EQ(res["type"].as_string(), "array");
    result = res["value"].as_float64_array();
    EXPECT_EQ(result.to_json(), "[1.0, 1.0]");
  }

  //confirm that it clamps to the beginning and end as expected
  for(const string &expression : {
      "gradient_range(val, first_absolute_time=0.0, last_absolute_time=20.0)",
    }) {
    res = eval.evaluate(expression);
    EXPECT_EQ(res["type"].as_string(), "array");
    result = res["value"].as_float64_array();
    EXPECT_EQ(result.to_json(), "[0.5, 0.5, 0.25]");
  }

  for(const string &expression : {
      "gradient_range(val, first_absolute_cycle=0, last_absolute_cycle=500)",
    }) {
    res = eval.evaluate(expression);
    EXPECT_EQ(res["type"].as_string(), "array");
    result = res["value"].as_float64_array();
    EXPECT_EQ(result.to_json(), "[0.01, 0.01, 0.005]");
  }

  // confirm it behaves properly with other operators that take an array as input
  for(const string &expression : {
      "max(gradient_range(val, first_absolute_index=1, last_absolute_index=3))", 
    }) {
    res = eval.evaluate(expression);
    EXPECT_EQ(res["type"].as_string(), "double");
    EXPECT_EQ(res["value"].to_float64(), 1);
  }

  //confirm it returns an empty gradient if there is only a single value
  for(const string &expression : {
      "gradient_range(val, first_absolute_index=0, last_absolute_index=0)", 
      "gradient_range(val, first_relative_index=0, last_relative_index=0)",
      "gradient_range(val, first_absolute_time=1.0, last_absolute_time=3.0)",
      "gradient_range(val, first_absolute_time=2.0, last_absolute_time=2.0)",
    }) {
    res = eval.evaluate(expression);
    EXPECT_EQ(res["type"].as_string(), "double");
    EXPECT_EQ(res["value"].to_string(), "\"-inf\"");
  }

  //test the gradient of gradient
  for(const string &expression : {
      "gradient_range(gradient_val, first_absolute_cycle=200, last_absolute_cycle=500)",
    }) {
    res = eval.evaluate(expression);
    EXPECT_EQ(res["type"].as_string(), "array");
    result = res["value"].as_float64_array();
    EXPECT_EQ(result.to_json(), "[0.0, -2.5e-05]");
  }


}

//-----------------------------------------------------------------------------
TEST(ascent_expressions, test_history_range)
{
  Node n;
  ascent::about(n);

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
  runtime::expressions::ExpressionEval::reset_cache();

  conduit::Node res;
  std::string expr;

  // we can't change the input object so keep
  // giving eval new ones
  {
    multi_dom.child(0)["state/cycle"] = 100;
    multi_dom.child(0)["state/time"] = 1.0;
    runtime::expressions::ExpressionEval eval(&multi_dom);
    res = eval.evaluate("-1.0", "val");
    res = eval.evaluate("vector(1,2,3)", "vec");
  }

  {
    multi_dom.child(0)["state/cycle"] = 200;
    multi_dom.child(0)["state/time"] = 2.0;
    runtime::expressions::ExpressionEval eval(&multi_dom);
    res = eval.evaluate("-2.0", "val");
    res = eval.evaluate("vector(9,3,4)", "vec");
  }

  {
    multi_dom.child(0)["state/cycle"] = 300;
    multi_dom.child(0)["state/time"] = 3.0;
    runtime::expressions::ExpressionEval eval(&multi_dom);
    res = eval.evaluate("-3.0", "val");
    res = eval.evaluate("vector(3,4,0)", "vec");
  }

  {
    multi_dom.child(0)["state/cycle"] = 400;
    multi_dom.child(0)["state/time"] = 4.0;
    runtime::expressions::ExpressionEval eval(&multi_dom);
    res = eval.evaluate("-4.0", "val");
    res = eval.evaluate("vector(6,4,8)", "vec");
  }

  conduit::float64_array result;
  runtime::expressions::ExpressionEval eval(&multi_dom);

  for(const string &expression : {
      "history_range(val, first_absolute_index=0, last_absolute_index=2)", 
    }) {
    res = eval.evaluate(expression);
    EXPECT_EQ(res["type"].as_string(), "array");
    result = res["value"].as_float64_array();
    EXPECT_EQ(result.to_json(), "[-1.0, -2.0, -3.0]");
  }


  for(const string &expression : {
      "max(history_range(val, first_absolute_index=0, last_absolute_index=2))", 
    }) {
    res = eval.evaluate(expression);
    EXPECT_EQ(res["type"].as_string(), "double");
    EXPECT_EQ(res["value"].as_double(), -1);
  }

  for(const string &expression : {
      "max(history_range(val, first_absolute_index=0, last_absolute_index=2)) == -1.0", 
    }) {
    res = eval.evaluate(expression);
    EXPECT_EQ(res["type"].as_string(), "bool");
    EXPECT_EQ(res["value"].to_int8(), 1);
  }

  for(const string &expression : {
      "history_range(val, first_absolute_index=0, last_absolute_index=2)", 
      "history_range(val, first_relative_index=1, last_relative_index=3)",
      "history_range(val, first_absolute_time=1.0, last_absolute_time=3.0)",
      "history_range(val, first_absolute_cycle=100, last_absolute_cycle=300)",  
    }) {
    res = eval.evaluate(expression);
    EXPECT_EQ(res["type"].as_string(), "array");
    result = res["value"].as_float64_array();
    EXPECT_EQ(result.to_json(), "[-1.0, -2.0, -3.0]");
  }

  for(const string &expression : {
      "history_range(val, first_absolute_index=1, last_absolute_index=2)", 
      "history_range(val, first_relative_index=1, last_relative_index=2)",
      "history_range(val, first_absolute_time=2.0, last_absolute_time=3.0)",
      "history_range(val, first_absolute_cycle=200, last_absolute_cycle=300)",
    }) {
    res = eval.evaluate(expression);
    EXPECT_EQ(res["type"].as_string(), "array");
    result = res["value"].as_float64_array();
    EXPECT_EQ(result.to_json(), "[-2.0, -3.0]");
  }

  //confirm that it clamps to the end as expected
  for(const string &expression : {
      "history_range(val, first_absolute_index=1, last_absolute_index=5)",
      "history_range(val, first_absolute_time=2.0, last_absolute_time=5.0)",
      "history_range(val, first_absolute_cycle=200, last_absolute_cycle=500)",
    }) {
    res = eval.evaluate(expression);
    EXPECT_EQ(res["type"].as_string(), "array");
    result = res["value"].as_float64_array();
    EXPECT_EQ(result.to_json(), "[-2.0, -3.0, -4.0]");
  }

  //confirm that it clamps to the beginning as expected
  for(const string &expression : {
    "history_range(val, first_relative_index=1, last_relative_index=6)",
    "history_range(val, first_absolute_time=0.0, last_absolute_time=3.0)",
    "history_range(val, first_absolute_cycle=0, last_absolute_cycle=300)",
  }) {
    res = eval.evaluate(expression);
    EXPECT_EQ(res["type"].as_string(), "array");
    result = res["value"].as_float64_array();
    EXPECT_EQ(result.to_json(), "[-1.0, -2.0, -3.0]");
  }

  for(const string &expression : {
      "max(history_range(val, first_absolute_index=1, last_absolute_index=3))", 
    }) {
    res = eval.evaluate(expression);
    EXPECT_EQ(res["type"].as_string(), "double");
    EXPECT_EQ(res["value"].to_float64(), -2);
  }

}

//-----------------------------------------------------------------------------
TEST(ascent_expressions, if_expressions)
{
  Node n;
  ascent::about(n);

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

  expr = "if 5 == 5 then 1 else 2";
  res = eval.evaluate(expr);
  EXPECT_EQ(res["value"].to_int32(), 1);
  EXPECT_EQ(res["type"].as_string(), "int");

  bool threw = false;
  try
  {
    expr = "if max(3,7) > min(2,6) then 1 else vector(1,1,1)";
    res = eval.evaluate(expr);
  }
  catch(...)
  {
    threw = true;
  }
  EXPECT_EQ(threw, true);
}

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
TEST(ascent_expressions, lineout)
{
  Node n;
  ascent::about(n);

  // only run this test if ascent was built with dray support
  if(n["runtimes/ascent/dray/status"].as_string() == "disabled")
  {
      ASCENT_INFO("Ascent Devil Ray support disabled, skipping test");
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

  expr = "lineout(10,vector(0,1,1),vector(5,5,5), fields=['braid'], empty_val=-1.0)";
  res = eval.evaluate(expr);
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

