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

#include <ascent.hpp>

#include <iostream>
#include <math.h>

#include <conduit_blueprint.hpp>
#include <conduit_relay_io_blueprint.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"


using namespace std;
using namespace conduit;
using namespace ascent;


index_t EXAMPLE_MESH_SIDE_DIM = 20;

//-----------------------------------------------------------------------------
TEST(ascent_amr, test_amr_render_simple)
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
    blueprint::mesh::examples::julia_nestsets_simple(-2.0,  2.0, // x range
                                                     -2.0,  2.0, // y range
                                                     0.285, 0.01, // c value
                                                     data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing 3D Rendering with Default Pipeline");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,
                                                        "tout_render_amr_simple");

    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "iters";
    //scenes["s1/plots/p2/type"] = "mesh";
    scenes["s1/image_prefix"] = output_file;

    conduit::Node actions;
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

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file,0.01,"0"));
    std::string msg = "An example of rendering amr data";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_amr, test_amr_render_complex)
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
    blueprint::mesh::examples::julia_nestsets_complex(EXAMPLE_MESH_SIDE_DIM,
                                                      EXAMPLE_MESH_SIDE_DIM,
                                                      -2.0,  2.0, // x range
                                                      -2.0,  2.0, // y range
                                                      0.285, 0.01, // c value
                                                      2, // amr levels
                                                      data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing 3D Rendering with Default Pipeline");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,
                                                        "tout_render_amr_complex");

    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "iters";
    //scenes["s1/plots/p1/field"] = "topo_ghosts";
    //scenes["s1/plots/p2/type"] = "mesh";
    scenes["s1/image_prefix"] = output_file;

    conduit::Node actions;
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

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file,0.01,"0"));
    std::string msg = "An example of rendering amr data";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_amr, test_amr_render_complex_ghosts)
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
    blueprint::mesh::examples::julia_nestsets_complex(EXAMPLE_MESH_SIDE_DIM,
                                                      EXAMPLE_MESH_SIDE_DIM,
                                                      -2.0,  2.0, // x range
                                                      -2.0,  2.0, // y range
                                                      0.285, 0.01, // c value
                                                      2, // amr levels
                                                      data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));
    ASCENT_INFO("Testing 3D Rendering with Default Pipeline");



    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,
                                                        "tout_render_amr_complex_ghost");

    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "iters";
    scenes["s1/image_prefix"] = output_file;

    conduit::Node actions;
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

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file,0.01,"0"));
    std::string msg = "An example of rendering amr data";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
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


