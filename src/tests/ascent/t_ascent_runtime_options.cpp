//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-716457
//
// All rights reserved.
//
// This file is part of Ascent.
//
// For details, see: http://software.llnl.gov/ascent/.
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
/// file: t_ascent_runtime_options.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <ascent.hpp>

#include <iostream>
#include <math.h>
#include <sstream>

#include <conduit_blueprint.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"


using namespace std;
using namespace conduit;
using namespace ascent;


index_t EXAMPLE_MESH_SIDE_DIM = 20;
//-----------------------------------------------------------------------------
TEST(ascent_runtime_options, verbose_msgs)
{
    //
    // Create example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("quads",100,100,0,data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    Node actions;
    Node &hello = actions.append();
    hello["action"]   = "hello!";
    actions.print();

    // we want the "empty" example pipeline
    Node open_opts;
    open_opts["runtime/type"] = "empty";
    open_opts["messages"] = "verbose";

    //
    // Run Ascent
    //
    Ascent ascent;
    ascent.open(open_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_runtime_options, quiet_msgs)
{
    //
    // Create example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("quads",100,100,0,data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    Node actions;
    Node &hello = actions.append();
    hello["action"]   = "hello!";
    actions.print();

    // we want the "empty" example pipeline
    Node open_opts;
    open_opts["runtime/type"] = "empty";
    open_opts["messages"] = "quiet";

    //
    // Run Ascent
    //
    Ascent ascent;
    ascent.open(open_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();
}


//-----------------------------------------------------------------------------
TEST(ascent_runtime_options, forward_exceptions)
{


    // invoke error by choosing bad runtime
    Node open_opts;
    open_opts["exceptions"] = "forward";
    open_opts["runtime/type"] = "BAD";


    //
    // Run Ascent
    //
    Ascent ascent;
    EXPECT_THROW(ascent.open(open_opts),conduit::Error);
    ascent.close();
}


//-----------------------------------------------------------------------------
TEST(ascent_runtime_options, catch_exceptions)
{


    // make sure bad runtime selection is caught
    Node open_opts;
    open_opts["exceptions"] = "catch";
    open_opts["runtime/type"] = "BAD";


    //
    // Run Ascent
    //
    Ascent ascent;
    ascent.open(open_opts);
    ascent.close();
}


//-----------------------------------------------------------------------------
TEST(ascent_runtime_options, test_errors)
{
    // invoke error cases caused by not initializing ascent

    Ascent ascent;
    Node n;
    // these will error to std::out, but we want to check they dont cras
    ascent.publish(n);
    ascent.publish(n);
    ascent.close();

    Node open_opts;
    open_opts["exceptions"] = "forward";
    ascent.open(open_opts);
    ascent.close();

    EXPECT_THROW(ascent.publish(n),conduit::Error);
    EXPECT_THROW(ascent.publish(n),conduit::Error);
    ascent.close();

}

//-----------------------------------------------------------------------------
TEST(ascent_runtime_options, test_error_actions_file)
{
    // invoke error cases caused by not initializing ascent

    Ascent ascent;
    Node n;

    Node open_opts;
    open_opts["exceptions"] = "forward";
    open_opts["actions_file"] = "bananas.yaml";
    ascent.open(open_opts);
    Node actions;

    EXPECT_THROW(ascent.execute(actions),conduit::Error);
    ascent.close();

}

//-----------------------------------------------------------------------------
TEST(ascent_runtime_options, test_timings)
{
    // the ascent runtime is currently our only rendering runtime
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

    ASCENT_INFO("Testing custom actions file");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_render_actions_img");
    string output_actions = conduit::utils::join_file_path(output_path,"tout_render_actions.json");

    string timings_file = "ascent_filter_times.csv";
    timings_file = conduit::utils::join_file_path(output_path,timings_file);

    // remove old images before rendering
    remove_test_image(output_file);
    remove_test_file(output_actions);
    remove_test_file(timings_file);


    //
    // Create the actions.
    //
    std::string actions_file = ""
                              "  [\n"
                              "    {\n"
                              "      \"action\": \"add_scenes\",\n"
                              "      \"scenes\": \n"
                              "      {\n"
                              "        \"s1\": \n"
                              "        {\n"
                              "          \"plots\":\n"
                              "          {\n"
                              "            \"p1\": \n"
                              "            {\n"
                              "              \"type\": \"pseudocolor\",\n"
                              "              \"field\": \"braid\"\n"
                              "            }\n"
                              "          },\n"
                              "          \"renders\": \n"
                              "          {\n"
                              "            \"r1\": \n"
                              "            {\n"
                              "              \"image_prefix\": \"" + output_file + "\"\n"
                              "            }\n"
                              "          }\n"
                              "        }\n"
                              "      }\n"
                              "    }\n"
                              "  ]\n";

    std::ofstream file(output_actions);
    file<<actions_file;
    file.close();
    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    //ascent_opts["ascent_info"] = "verbose";
    ascent_opts["runtime/type"] = "ascent";
    ascent_opts["actions_file"] = output_actions;
    ascent_opts["timings"] = "true";
    ascent_opts["default_dir"] = output_path;
    ascent.open(ascent_opts);
    ascent.publish(data);
    conduit::Node blank_actions;
    ascent.execute(blank_actions);
    ascent.close();
    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    EXPECT_TRUE(check_test_file(output_actions));
    // check to see if we created the timings
    EXPECT_TRUE(check_test_file(timings_file));
}

//-----------------------------------------------------------------------------
TEST(ascent_runtime_options, test_actions_file)
{
    // the ascent runtime is currently our only rendering runtime
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

    ASCENT_INFO("Testing custom actions file");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_render_actions_img");
    string output_actions = conduit::utils::join_file_path(output_path,"tout_render_actions.json");

    // remove old images before rendering
    remove_test_image(output_file);
    remove_test_file(output_actions);


    //
    // Create the actions.
    //
    std::string actions_file = ""
                              "  [\n"
                              "    {\n"
                              "      \"action\": \"add_scenes\",\n"
                              "      \"scenes\": \n"
                              "      {\n"
                              "        \"s1\": \n"
                              "        {\n"
                              "          \"plots\":\n"
                              "          {\n"
                              "            \"p1\": \n"
                              "            {\n"
                              "              \"type\": \"pseudocolor\",\n"
                              "              \"field\": \"braid\"\n"
                              "            }\n"
                              "          },\n"
                              "          \"renders\": \n"
                              "          {\n"
                              "            \"r1\": \n"
                              "            {\n"
                              "              \"image_prefix\": \"" + output_file + "\"\n"
                              "            }\n"
                              "          }\n"
                              "        }\n"
                              "      }\n"
                              "    }\n"
                              "  ]\n";

    std::ofstream file(output_actions);
    file<<actions_file;
    file.close();
    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    //ascent_opts["ascent_info"] = "verbose";
    ascent_opts["runtime/type"] = "ascent";
    ascent_opts["actions_file"] = output_actions;
    ascent.open(ascent_opts);
    ascent.publish(data);
    conduit::Node blank_actions;
    ascent.execute(blank_actions);
    ascent.close();
    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    EXPECT_TRUE(check_test_file(output_actions));
}



//-----------------------------------------------------------------------------
TEST(ascent_runtime_options, test_default_dir)
{
    // the ascent runtime is currently our only rendering runtime
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

    ASCENT_INFO("Testing custom actions yaml file");


    string output_path = prepare_output_dir();
    std::string dir_name = "bananas";
    output_path = conduit::utils::join_file_path(output_path,dir_name);
    if(!conduit::utils::is_directory(output_path))
    {
        conduit::utils::create_directory(output_path);
    }

    string image_name = "tout_render_actions_img_yaml";
    string actions_name = "tout_render_actions.yaml";
    // full paths
    string output_file = conduit::utils::join_file_path(output_path,image_name);
    string output_actions = conduit::utils::join_file_path(output_path,actions_name);


    // remove old images before rendering
    remove_test_image(output_file);
    remove_test_file(output_actions);


    //
    // Create the actions.
    //
    std::string actions_file = ""
                              "-\n"
                              "  action: add_scenes\n"
                              "  scenes:\n"
                              "        s1:\n"
                              "          plots:\n"
                              "            p1: \n"
                              "              type: pseudocolor\n"
                              "              field: braid\n"
                              "          renders:\n"
                              "            r1:\n"
                              "              image_prefix: " + image_name + "\n";


    std::ofstream file(output_actions);
    file<<actions_file;
    file.close();
    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    //ascent_opts["ascent_info"] = "verbose";
    ascent_opts["runtime/type"] = "ascent";
    ascent_opts["actions_file"] = output_actions;
    ascent_opts["default_dir"] = output_path;
    ascent.open(ascent_opts);
    ascent.publish(data);
    conduit::Node blank_actions;
    ascent.execute(blank_actions);
    ascent.close();
    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    EXPECT_TRUE(check_test_file(output_actions));
}

//-----------------------------------------------------------------------------
TEST(ascent_runtime_options, test_actions_yaml_file)
{
    // the ascent runtime is currently our only rendering runtime
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

    ASCENT_INFO("Testing custom actions yaml file");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_render_actions_img_yaml");
    string output_actions = conduit::utils::join_file_path(output_path,"tout_render_actions.yaml");

    // remove old images before rendering
    remove_test_image(output_file);
    remove_test_file(output_actions);


    //
    // Create the actions.
    //
    std::string actions_file = ""
                              "-\n"
                              "  action: add_scenes\n"
                              "  scenes:\n"
                              "        s1:\n"
                              "          plots:\n"
                              "            p1: \n"
                              "              type: pseudocolor\n"
                              "              field: braid\n"
                              "          renders:\n"
                              "            r1:\n"
                              "              image_prefix: " + output_file + "\n";


    std::ofstream file(output_actions);
    file<<actions_file;
    file.close();
    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    //ascent_opts["ascent_info"] = "verbose";
    ascent_opts["runtime/type"] = "ascent";
    ascent_opts["actions_file"] = output_actions;
    ascent.open(ascent_opts);
    ascent.publish(data);
    conduit::Node blank_actions;
    ascent.execute(blank_actions);
    ascent.close();
    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    EXPECT_TRUE(check_test_file(output_actions));
}

//-----------------------------------------------------------------------------
TEST(ascent_runtime_options, test_field_filtering)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent vtkm support disabled, skipping test");
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

    ASCENT_INFO("Testing field filtering");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_field_filtering");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";

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
    ascent_opts["field_filtering"] = "true";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of filtering fields not present in the actions file.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_runtime_options, test_field_filtering_error)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent vtkm support disabled, skipping test");
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

    ASCENT_INFO("Testing field filtering error");

    //
    // Create the actions.
    //

    conduit::Node extracts;
    extracts["e1/type"]  = "relay";

    extracts["e1/params/path"] = "will never happen";
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

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
    ascent_opts["field_filtering"] = "true";
    ascent_opts["exceptions"] = "forward";
    ascent.open(ascent_opts);
    ascent.publish(data);
    bool error = false;
    try
    {
       ascent.execute(actions);
    }
    catch(...)
    {
      error = true;
    }
    ascent.close();

    EXPECT_TRUE(error);
}
