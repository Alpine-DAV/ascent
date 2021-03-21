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
/// file: t_ascent_slice.cpp
///
//-----------------------------------------------------------------------------


#include "gtest/gtest.h"

#include <ascent.hpp>
#include <ascent_hola.hpp>

#include <iostream>
#include <math.h>

#include <conduit_blueprint.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"


using namespace std;
using namespace conduit;
using namespace ascent;

//-----------------------------------------------------------------------------
TEST(ascent_devil_ray, test_pseudocolor)
{   Node n;
    ascent::about(n);

    //
    // Create an example mesh.
    //
    Node data, hola_opts, verify_info;
    hola_opts["root_file"] = test_data_file("taylor_green.cycle_001860.root");
    ascent::hola("relay/blueprint/mesh", hola_opts, data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing Devil Ray");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_dray_surface");

    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //

    conduit::Node extracts;
    extracts["e1/type"] = "dray_pseudocolor";
    // filter knobs
    conduit::Node &params = extracts["e1/params/"];
    params["field"] = "density";
    params["min_value"] = 0.99;
    params["max_value"] = 1.0;
    params["log_scale"] = "false";
    params["image_prefix"] = output_file;
    params["camera/azimuth"] = -30;
    params["camera/elevation"] = 35;

    params["draw_mesh"] = "true";
    params["line_thickness"] = 0.1;
    float line_color[4] = {0.f, 0.f, 0.f, 1.0f};
    params["line_color"].set(line_color, 4);;

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

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
    EXPECT_TRUE(check_test_image(output_file, 0.1, "1860"));
    std::string msg = "An example of using devil ray for pseudocolor plot.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_devil_ray, test_pseudocolor_noannots)
{   Node n;
    ascent::about(n);

    //
    // Create an example mesh.
    //
    Node data, hola_opts, verify_info;
    hola_opts["root_file"] = test_data_file("taylor_green.cycle_001860.root");
    ascent::hola("relay/blueprint/mesh", hola_opts, data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing Devil Ray");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_dray_noannots");

    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //

    conduit::Node extracts;
    extracts["e1/type"] = "dray_pseudocolor";

    // filter knobs
    conduit::Node &params = extracts["e1/params/"];
    params["field"] = "density";
    params["min_value"] = 0.99;
    params["max_value"] = 1.0;
    params["annotations"] = "false";
    params["log_scale"] = "false";
    params["image_prefix"] = output_file;
    params["camera/azimuth"] = -30;
    params["camera/elevation"] = 35;

    params["draw_mesh"] = "true";
    params["line_thickness"] = 0.1;
    float line_color[4] = {0.f, 0.f, 0.f, 1.0f};
    params["line_color"].set(line_color, 4);;

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

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
    EXPECT_TRUE(check_test_image(output_file, 0.1, "1860"));
    std::string msg = "An example of using devil ray for pseudocolor plot.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_devil_ray, test_3slice)
{
    Node n;
    ascent::about(n);

    //
    // Create an example mesh.
    //
    Node data, hola_opts, verify_info;
    hola_opts["root_file"] = test_data_file("taylor_green.cycle_001860.root");
    ascent::hola("relay/blueprint/mesh", hola_opts, data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing Devil Ray");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_dray_3slice");

    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //

    conduit::Node extracts;
    extracts["e1/type"] = "dray_3slice";
    // filter knobs
    conduit::Node &params = extracts["e1/params/"];
    params["field"] = "density";
    params["min_value"] = 0.99;
    params["max_value"] = 1.0;
    params["log_scale"] = "false";
    params["image_prefix"] = output_file;
    params["camera/azimuth"] = -30;
    params["camera/elevation"] = 35;

    params["x_offset"] = 0.;
    params["y_offset"] = 0.;
    params["z_offset"] = 0.;

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

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
    EXPECT_TRUE(check_test_image(output_file, 0.1, "1860"));
    std::string msg = "An example of using devil ray for pseudocolor plot.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_devil_ray, test_volume)
{
    Node n;
    ascent::about(n);

    //
    // Create an example mesh.
    //
    Node data, hola_opts, verify_info;
    hola_opts["root_file"] = test_data_file("taylor_green.cycle_001860.root");
    ascent::hola("relay/blueprint/mesh", hola_opts, data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing Devil Ray");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_dray_volume");

    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //

    conduit::Node extracts;
    extracts["e1/type"] = "dray_volume";
    // filter knobs
    conduit::Node &params = extracts["e1/params/"];
    params["field"] = "density";
    //params["min_value"] = 0.955;
    params["min_value"] = 0.98;
    params["max_value"] = 1.04;
    params["log_scale"] = "false";
    params["image_prefix"] = output_file;
    params["camera/azimuth"] = -30;
    params["camera/elevation"] = 35;

    params["samples"] = 100;

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

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
    EXPECT_TRUE(check_test_image(output_file, 0.1, "1860"));
    std::string msg = "An example of using devil ray for pseudocolor plot.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}
//-----------------------------------------------------------------------------
TEST(ascent_devil_ray, test_reflect)
{
    Node n;
    ascent::about(n);

    //
    // Create an example mesh.
    //
    Node data, hola_opts, verify_info;
    hola_opts["root_file"] = test_data_file("taylor_green.cycle_001860.root");
    ascent::hola("relay/blueprint/mesh", hola_opts, data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing Devil Ray");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_reflect");

    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //

    conduit::Node pipelines;
    pipelines["p1/f1/type"] = "dray_reflect";
    // filter knobs
    conduit::Node &rparams = pipelines["p1/f1/params/"];
    rparams["point/x"] = 0.;
    rparams["point/y"] = 0.;
    rparams["point/z"] = 0.;
    rparams["normal/x"] = 0.;
    rparams["normal/y"] = 1.;
    rparams["normal/z"] = 0.;

    conduit::Node extracts;
    extracts["e1/type"] = "dray_pseudocolor";
    extracts["e1/pipeline"] = "p1";
    // filter knobs
    conduit::Node &params = extracts["e1/params/"];
    params["field"] = "density";
    //params["min_value"] = 0.955;
    params["min_value"] = 0.98;
    params["max_value"] = 1.04;
    params["log_scale"] = "false";
    params["image_prefix"] = output_file;
    params["camera/azimuth"] = -30;
    params["camera/elevation"] = 35;

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

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
    EXPECT_TRUE(check_test_image(output_file, 0.1, "1860"));
    std::string msg = "An example of using devil ray to reflect a data set.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_devil_ray, test_scalar_rendering)
{
    Node n;
    ascent::about(n);

    //
    // Create an example mesh.
    //
    Node data, hola_opts, verify_info;
    hola_opts["root_file"] = test_data_file("taylor_green.cycle_001860.root");
    ascent::hola("relay/blueprint/mesh", hola_opts, data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing Devil Ray");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_scalar_renderer");

    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "dray_project_2d";
    // filter knobs
    conduit::Node &params = pipelines["pl1/f1/params"];
    params["image_width"] = 512;
    params["image_height"] = 512;
    params["fields"].append() = "density";

    conduit::Node extracts;
    extracts["e1/type"]  = "relay";
    extracts["e1/pipeline"] = "pl1";

    extracts["e1/params/path"] = output_file;
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;
    // add the pipeline
    conduit::Node &add_pipelines= actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;

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
    std::string msg = "An example of using devil ray scalar rendering.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_devil_ray, test_scalar_rendering_plane)
{
    Node n;
    ascent::about(n);

    //
    // Create an example mesh.
    //
    Node data, hola_opts, verify_info;
    hola_opts["root_file"] = test_data_file("taylor_green.cycle_001860.root");
    ascent::hola("relay/blueprint/mesh", hola_opts, data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing Devil Ray");

    string output_path = prepare_output_dir();
    string output_file
      = conduit::utils::join_file_path(output_path,"tout_scalar_renderer_plane");

    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "dray_project_2d";
    // filter knobs
    conduit::Node &params = pipelines["pl1/f1/params"];
    params["image_width"] = 512;
    params["image_height"] = 512;

    double center[3] = {0.5, 0.0, -1.5};
    double up[3] = {0., 1., 0.};
    double normal[3] = {0., 0.25, 1.};
    params["plane/center"].set(center,3);
    params["plane/up"].set(up,3);
    params["plane/normal"].set(normal,3);
    params["plane/width"] = 1.5f;
    params["plane/height"] = 1.5f;

    params["fields"].append() = "density";

    conduit::Node extracts;
    extracts["e1/type"]  = "relay";
    extracts["e1/pipeline"] = "pl1";

    extracts["e1/params/path"] = output_file;
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;
    // add the pipeline
    conduit::Node &add_pipelines= actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;

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
    std::string msg = "An example of using devil ray scalar rendering using a plane.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);

    // allow override of the data size via the command line
    result = RUN_ALL_TESTS();
    return result;
}


