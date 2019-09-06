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
#include <cmath>

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

    expr = "8 % 3";
    res = eval.evaluate(expr);
    EXPECT_EQ(res["value"].to_int32(), 2);
    EXPECT_EQ(res["type"].as_string(), "int");

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

    expr = "magnitude(max(field(\"braid\"))[\"position\"]) > 0";
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
    data["fields/ele_example/association"] =  "element";
    // reference the topology this field is defined on by name
    data["fields/ele_example/topology"] =  "topo";
    // set the field values, for this case we have 9 elements
    data["fields/ele_example/values"].set(DataType::float64(16));

    float64 *ele_vals_ptr = data["fields/ele_example/values"].value();

    for(int i = 0; i < 16; i++)
    {
        ele_vals_ptr[i] = float64(i);
    }

    data["state/cycle"] = 100;

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

    expr = "max(field(\"ele_example\"))[\"position\"]";
    res = eval.evaluate(expr);
    EXPECT_EQ(res["value"].as_float64(), 25);
    EXPECT_EQ(res["type"].as_string(), "vector");

    expr = "entropy(histogram(field(\"ele_example\")))";
    res = eval.evaluate(expr);
    EXPECT_EQ(res["value"].as_float64(), -std::log(1.0/16.0));
    EXPECT_EQ(res["type"].as_string(), "double");

    expr = "bin(cdf(histogram(field(\"ele_example\"))), val=5.0)";
    res = eval.evaluate(expr);
    EXPECT_EQ(res["value"].as_float64(), .375);
    EXPECT_EQ(res["type"].as_string(), "double");

    expr = "bin(pdf(histogram(field(\"ele_example\"))), val=5)";
    res = eval.evaluate(expr);
    EXPECT_EQ(res["value"].as_float64(), 1.0/16.0);
    EXPECT_EQ(res["type"].as_string(), "double");

    expr = "bin(pdf(histogram(field(\"ele_example\"))), val=4.5)";
    res = eval.evaluate(expr);
    EXPECT_EQ(res["value"].as_float64(), 0);
    EXPECT_EQ(res["type"].as_string(), "double");

    expr = "bin(pdf(histogram(field(\"ele_example\"))), bin=4) == pdf(histogram(field(\"ele_example\")))[\"value\"][4]";
    res = eval.evaluate(expr);
    EXPECT_EQ(res["value"].to_uint8(), 1);
    EXPECT_EQ(res["type"].as_string(), "bool");

    expr = "quantile(cdf(histogram(field(\"ele_example\"))), 1.0)";
    res = eval.evaluate(expr);
    EXPECT_EQ(res["value"].to_float64(), 15);
    EXPECT_EQ(res["type"].as_string(), "double");
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

    expr = "histogram(field(\"braid\"), num_bins=10)";
    res = eval.evaluate(expr);
    EXPECT_EQ(res["attrs/value/value"].dtype().number_of_elements(), 10);
    EXPECT_EQ(res["type"].as_string(), "histogram");

    expr = "histogram(field(\"braid\"),min_val=0,num_bins=10,max_val=1)";
    res = eval.evaluate(expr);
    EXPECT_EQ(res["attrs/value/value"].dtype().number_of_elements(), 10);
    EXPECT_EQ(res["attrs/min_val/value"].to_float64(), 0);
    EXPECT_EQ(res["attrs/max_val/value"].to_float64(), 1);
    EXPECT_EQ(res["type"].as_string(), "histogram");

    bool threw = false;
    try
    {
      expr = "histogram(field(\"braid\"),field(\"braid\"))";
      res = eval.evaluate(expr);
    }
    catch(...)
    {
      threw = true;
    }
    EXPECT_EQ(threw, true);


    expr = "histogram(field(\"braid\"),max_val=2)";
    res = eval.evaluate(expr);
    EXPECT_EQ(res["attrs/max_val/value"].to_float64(), 2);
    EXPECT_EQ(res["type"].as_string(), "histogram");

    threw = false;
    try
    {
      expr = "histogram(field(\"braid\"),min_val=field(\"braid\"))";
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
      expr = "histogram(field(\"braid\"),min_val=0,num_bins=10,1)";
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
    expr = "max(field(\"braid\"))";
    const std::string cache_name = "mx_b";
    res1 = eval.evaluate(expr, cache_name);
    res2 = eval.evaluate("mx_b");
    EXPECT_EQ(res1["value"].to_float64(), res2["value"].to_float64());

    // grab attribute from cached value
    res2 = eval.evaluate("mx_b[\"position\"]");
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
    runtime::expressions::ExpressionEval eval(&multi_dom);

    conduit::Node res;
    std::string expr;


    res = eval.evaluate("1", "val");
    res = eval.evaluate("vector(1,2,3)", "vec");
    data["state/cycle"] = 200;
    res = eval.evaluate("2", "val");
    res = eval.evaluate("vector(9,3,4)", "vec");
    data["state/cycle"] = 300;
    res = eval.evaluate("3", "val");
    res = eval.evaluate("vector(3,4,0)", "vec");
    data["state/cycle"] = 400;
    res = eval.evaluate("4", "val");
    res = eval.evaluate("vector(6,4,8)", "vec");


    expr = "history(val, 2)";
    res = eval.evaluate(expr);
    EXPECT_EQ(res["value"].to_int32(), 2);
    EXPECT_EQ(res["type"].as_string(), "int");

    expr = "history(val, 3)";
    res = eval.evaluate(expr);
    EXPECT_EQ(res["value"].to_int32(), 1);
    EXPECT_EQ(res["type"].as_string(), "int");

    bool threw = false;
    try
    {
      expr = "history(val, 10)";
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


