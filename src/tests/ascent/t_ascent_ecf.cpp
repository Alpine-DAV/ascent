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
/// file: t_ascent_ecf.cpp
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


TEST(ascent_expressions, custom_ecf)
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
    // set the field values, for this case we have 16 elements
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

    Node bin_axes;

    Node &x_axes = bin_axes.append();
    x_axes["field_name"] = "x";
    x_axes["num_bins"] = 4;
    x_axes["min_val"] = 0;
    x_axes["max_val"] = 3;

    Node &y_axes = bin_axes.append();
    y_axes["field_name"] = "y";
    y_axes["num_bins"] = 4;
    y_axes["min_val"] = 0;
    y_axes["max_val"] = 3;

    Node res = runtime::expressions::ecf(&multi_dom, bin_axes, "ele_example", "sum");
    res.print();
}

//-----------------------------------------------------------------------------
TEST(ascent_ecf, braid_ecf)
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


