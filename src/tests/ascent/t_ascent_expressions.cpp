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

#include <iostream>
#include <math.h>

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
    EXPECT_EQ(res["type"].as_string(), "scalar");

    expr = "(2.0 * 2) / 2";
    res = eval.evaluate(expr);
    EXPECT_EQ(res["value"].to_float64(), 2.0);
    EXPECT_EQ(res["type"].as_string(), "scalar");

    expr = "2.0 + 1 / 0.5";
    res = eval.evaluate(expr);
    EXPECT_EQ(res["value"].to_float64(), 4.0);
    EXPECT_EQ(res["type"].as_string(), "scalar");

    expr = "max(1, 2)";
    res = eval.evaluate(expr);
    EXPECT_EQ(res["value"].to_float64(), 2.0);
    EXPECT_EQ(res["type"].as_string(), "scalar");

    expr = "max(1, 2.0)";
    res = eval.evaluate(expr);
    EXPECT_EQ(res["value"].to_float64(), 2.0);
    EXPECT_EQ(res["type"].as_string(), "scalar");

    expr = "min(1, 2)";
    res = eval.evaluate(expr);
    EXPECT_EQ(res["value"].to_float64(), 1.0);
    EXPECT_EQ(res["type"].as_string(), "scalar");

    bool threw = false;
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
    EXPECT_EQ(res["type"].as_string(), "scalar");

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
    EXPECT_EQ(res["type"].as_string(), "scalar");

    expr = "position(max(\"braid\"))";
    res = eval.evaluate(expr);
    EXPECT_EQ(res["type"].as_string(), "vector");

    expr = "magnitude(position(max(\"braid\"))) > 0";
    res = eval.evaluate(expr);
    EXPECT_EQ(res["value"].to_float64(), 1);
    EXPECT_EQ(res["type"].as_string(), "boolean");
}

//-----------------------------------------------------------------------------
TEST(ascent_expressions, expressions_optional_params)
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
    // test optional parameters
    std::string expr;
    conduit::Node res;
    expr = "histogram(\"braid\")";
    res = eval.evaluate(expr);
    EXPECT_EQ(res["value"].dtype().number_of_elements(), 256);
    EXPECT_EQ(res["type"].as_string(), "histogram");

    expr = "histogram(\"braid\", 10)";
    res = eval.evaluate(expr);
    EXPECT_EQ(res["value"].dtype().number_of_elements(), 10);
    EXPECT_EQ(res["type"].as_string(), "histogram");

    expr = "histogram(\"braid\",10,0,1)";
    res = eval.evaluate(expr);
    EXPECT_EQ(res["value"].dtype().number_of_elements(), 10);
    EXPECT_EQ(res["min_val"].to_float64(), 0);
    EXPECT_EQ(res["max_val"].to_float64(), 1);
    EXPECT_EQ(res["type"].as_string(), "histogram");
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

    expr = "histogram(\"braid\", num_bins=10)";
    res = eval.evaluate(expr);
    EXPECT_EQ(res["value"].dtype().number_of_elements(), 10);
    EXPECT_EQ(res["type"].as_string(), "histogram");

    expr = "histogram(\"braid\",min_val=0,num_bins=10,max_val=1)";
    res = eval.evaluate(expr);
    EXPECT_EQ(res["value"].dtype().number_of_elements(), 10);
    EXPECT_EQ(res["min_val"].to_float64(), 0);
    EXPECT_EQ(res["max_val"].to_float64(), 1);
    EXPECT_EQ(res["type"].as_string(), "histogram");

    bool threw = false;
    try
    {
      expr = "histogram(\"braid\",\"braid\")";
      res = eval.evaluate(expr);
    }
    catch(...)
    {
      threw = true;
    }
    EXPECT_EQ(threw, true);


    expr = "histogram(\"braid\",max_val=2)";
    res = eval.evaluate(expr);
    EXPECT_EQ(res["max_val"].to_float64(), 2);
    EXPECT_EQ(res["type"].as_string(), "histogram");

    threw = false;
    try
    {
      expr = "histogram(\"braid\",min_val=\"braid\")";
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
      expr = "histogram(\"braid\",min_val=0,num_bins=10,1)";
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
    expr = "max(\"braid\")";
    const std::string cache_name = "mx_b";
    res1 = eval.evaluate(expr, cache_name);
    res2 = eval.evaluate("mx_b");
    EXPECT_EQ(res1["value"].to_float64(), res2["value"].to_float64());

    // grab attribute from cached value
    res2 = eval.evaluate("position(mx_b)");
    EXPECT_EQ(res2["type"].as_string(), "vector");

}

//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
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


