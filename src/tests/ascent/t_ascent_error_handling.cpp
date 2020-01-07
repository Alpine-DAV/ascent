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

#include "t_config.hpp"
#include "t_utils.hpp"




using namespace std;
using namespace conduit;
using namespace ascent;

void
throw_handler(const std::string &,
              const std::string &,
              int )
{
  ASCENT_ERROR("");
}

index_t EXAMPLE_MESH_SIDE_DIM = 20;

//-----------------------------------------------------------------------------
TEST(ascent_error_handling, test_bad_filter_field)
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

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing bad field name for filter");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_never_written");

    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "contour";
    // filter knobs
    conduit::Node &contour_params = pipelines["pl1/f1/params"];
    contour_params["field"] = "bananas";
    contour_params["iso_values"] = 0.;

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/image_prefix"] = output_file;

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
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
    ascent_opts["exceptions"] = "forward";
    ascent.open(ascent_opts);
    ascent.publish(data);

    bool error = false;
    try
    {
      ascent.execute(actions);
    }
    catch(conduit::Error &e)
    {
      error = true;
    }

    ASSERT_TRUE(error);
    ascent.close();
}
//-----------------------------------------------------------------------------
TEST(ascent_error_handling, test_bad_plot_var_name)
{

    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

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

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));
    //verify_info.print();

    ASCENT_INFO("Testing bad plot variable name");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_empty");

    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "clip";
    // filter knobs
    conduit::Node &clip_params = pipelines["pl1/f1/params"];
    clip_params["topology"] = "mesh";
    clip_params["box/min/x"] = 0.1;
    clip_params["box/min/y"] = 0.1;
    clip_params["box/min/z"] = 0.1;
    clip_params["box/max/x"] = 10.01; // <=
    clip_params["box/max/y"] = 10.01;
    clip_params["box/max/z"] = 10.01;

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "bananas";
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
    ascent_opts["exceptions"] = "forward";
    ascent.open(ascent_opts);
    ascent.publish(data);

    conduit::utils::set_info_handler(throw_handler);
    bool error = false;
    try
    {
      ascent.execute(actions);
    }
    catch(conduit::Error &e)
    {
      error = true;
    }

    ASSERT_TRUE(error);

    ascent.close();
    conduit::utils::set_info_handler(conduit::utils::default_info_handler);
}
//-----------------------------------------------------------------------------
TEST(ascent_error_handling, test_bad_color_table)
{

    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

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

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));
    //verify_info.print();

    ASCENT_INFO("Testing 3D warning for no bad color table name");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_bananas");

    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "radial";
    scenes["s1/plots/p1/color_table/name"] = "bananas";
    scenes["s1/image_prefix"] = output_file;

    conduit::Node actions;
    // add the pipeline
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
    ascent_opts["exceptions"] = "forward";
    ascent.open(ascent_opts);
    ascent.publish(data);

    conduit::utils::set_info_handler(throw_handler);
    bool error = false;
    try
    {
      ascent.execute(actions);
    }
    catch(conduit::Error &e)
    {
      error = true;
    }

    ASSERT_TRUE(error);

    ascent.close();
    conduit::utils::set_info_handler(conduit::utils::default_info_handler);
}
//-----------------------------------------------------------------------------
TEST(ascent_error_handling, test_emtpy)
{

    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

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

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));
    //verify_info.print();

    ASCENT_INFO("Testing 3D warning for no data to plot");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_empty");

    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "clip";
    // filter knobs
    conduit::Node &clip_params = pipelines["pl1/f1/params"];
    clip_params["topology"] = "mesh";
    clip_params["box/min/x"] = -10.1;
    clip_params["box/min/y"] = -10.1;
    clip_params["box/min/z"] = -10.1;
    clip_params["box/max/x"] = 10.01; // <=
    clip_params["box/max/y"] = 10.01;
    clip_params["box/max/z"] = 10.01;

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "radial";
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
    ascent_opts["exceptions"] = "forward";
    ascent.open(ascent_opts);
    ascent.publish(data);

    conduit::utils::set_info_handler(throw_handler);
    bool error = false;
    try
    {
      ascent.execute(actions);
    }
    catch(conduit::Error &e)
    {
      error = true;
    }

    ASSERT_TRUE(error);

    ascent.close();
    conduit::utils::set_info_handler(conduit::utils::default_info_handler);
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


