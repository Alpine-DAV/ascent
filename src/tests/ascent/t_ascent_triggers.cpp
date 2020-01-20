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
/// file: t_ascent_cinema_a.cpp
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

index_t EXAMPLE_MESH_SIDE_DIM = 32;

//-----------------------------------------------------------------------------
TEST(ascent_triggers, simple_rick)
{
    // the vtkm runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);

    //
    // Create example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    string output_path = prepare_output_dir();
    string trigger_file = conduit::utils::join_file_path(output_path,"simple_trigger_actions");
    string output_file = conduit::utils::join_file_path(output_path,"tout_simple_trigger_actions");
    // remove old file
    if(conduit::utils::is_file(trigger_file))
    {
        conduit::utils::remove_file(trigger_file);
    }


    //
    // Create trigger actions.
    //
    Node trigger_actions;

    conduit::Node extracts;

    extracts["e1/type"]  = "relay";
    extracts["e1/params/path"] = output_file;
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

    conduit::Node &add_ext= trigger_actions.append();
    add_ext["action"] = "add_extracts";
    add_ext["extracts"] = extracts;

    trigger_actions.save(trigger_file, "json");

    //
    // Create the actions.
    //
    Node actions;

    std::string condition = "1 == 1";
    conduit::Node triggers;
    triggers["t1/params/condition"] = condition;
    triggers["t1/params/actions_file"] = trigger_file;

    conduit::Node &add_triggers= actions.append();
    add_triggers["action"] = "add_triggers";
    add_triggers["triggers"] = triggers;
    actions.print();

    //
    // Run Ascent
    //

    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);

    conduit::Node info;
    ascent.info(info);
    std::string path = "expressions/" + condition + "/100/value";
    EXPECT_TRUE(info[path].to_int32() == 1);
    std::string msg = "A simple example of triggering actions based on a boolean"
                      " expression.";
    ASCENT_ACTIONS_DUMP(actions, std::string("basic_trigger"), msg);

    ascent.close();

}

//-----------------------------------------------------------------------------
TEST(ascent_triggers, complex_trigger)
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
    // Create example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    string output_path = prepare_output_dir();
    string trigger_file = conduit::utils::join_file_path(output_path,"complex_trigger_actions");
    string output_file = conduit::utils::join_file_path(output_path,"tout_complex_trigger_actions");
    // remove old files
    if(conduit::utils::is_file(trigger_file))
    {
      conduit::utils::remove_file(trigger_file);
    }
    if(conduit::utils::is_file(output_file))
    {
      conduit::utils::remove_file(output_file);
    }

    //
    // Create trigger actions.
    //
    Node trigger_actions;

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "radial";
    scenes["s1/image_prefix"]   = output_file;

    conduit::Node &add_scenes= trigger_actions.append();
    add_scenes["action"] = "add_scenes";
    add_scenes["scenes"] = scenes;

    trigger_actions.save(trigger_file, "json");

    //
    // Create the actions.
    //
    Node actions;
    // this should always be true
    std::string condition = "magnitude(max(field('braid')).position - vector(0,0,0)) > 0";
    conduit::Node triggers;
    triggers["t1/params/condition"] = condition;
    triggers["t1/params/actions_file"] = trigger_file;

    conduit::Node &add_triggers= actions.append();
    add_triggers["action"] = "add_triggers";
    add_triggers["triggers"] = triggers;
    actions.print();

    //
    // Run Ascent
    //

    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);

    conduit::Node info;
    ascent.info(info);
    std::string path = "expressions/" + condition + "/100/value";
    EXPECT_TRUE(info[path].to_uint8() == 1);

    ascent.close();

    // check that we created an image from the trigger
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "A more complex trigger example using several functions"
                      " that evaluate positons on the mesh.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_triggers, trigger_extract)
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
    // Create example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    string output_path = prepare_output_dir();
    string trigger_file = conduit::utils::join_file_path(output_path,"trigger_extract_actions");
    string output_file = conduit::utils::join_file_path(output_path,"tout_trigger_extract");
    string output_root_file = output_file + ".cycle_000100.root";

    // remove old files
    if(conduit::utils::is_file(trigger_file))
    {
      conduit::utils::remove_file(trigger_file);
    }

    if(conduit::utils::is_file(output_root_file))
    {
      conduit::utils::remove_file(output_root_file);
    }

    //
    // Create trigger actions.
    //
    Node trigger_actions;

    conduit::Node extracts;

    extracts["e1/type"]  = "relay";
    extracts["e1/params/path"] = output_file;
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

    conduit::Node &add_ext= trigger_actions.append();
    add_ext["action"] = "add_extracts";
    add_ext["extracts"] = extracts;

    trigger_actions.save(trigger_file, "json");

    //
    // Create the actions.
    //
    Node actions;
    // this should always be true
    std::string condition = "magnitude(max(field('braid')).position - vector(0,0,0)) > 0";
    conduit::Node triggers;
    triggers["t1/params/condition"] = condition;
    triggers["t1/params/actions_file"] = trigger_file;

    conduit::Node &add_triggers= actions.append();
    add_triggers["action"] = "add_triggers";
    add_triggers["triggers"] = triggers;
    actions.print();

    //
    // Run Ascent
    //

    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image from the trigger
    EXPECT_TRUE(conduit::utils::is_file(output_root_file));
    std::string msg = "A more complex trigger example using several functions"
                      " that evaluate positons on the mesh.";
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


