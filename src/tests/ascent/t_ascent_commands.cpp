//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
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
bool example_bool = false;

//-----------------------------------------------------------------------------
void void_callback_1(conduit::Node &params, conduit::Node &output)
{
    output["param_was_passed"] = false;
    if (params.has_path("example_param"))
    {
        output["param_was_passed"] = true;
    }
}

//-----------------------------------------------------------------------------
void void_callback_2(conduit::Node &params, conduit::Node &output)
{
    if (example_bool)
    {
        example_bool = false;
    }
    else
    {
        example_bool = true;
    }
}

//-----------------------------------------------------------------------------
bool bool_callback_1()
{
    return example_bool;
}

//-----------------------------------------------------------------------------
bool bool_callback_2()
{
    return !example_bool;
}

//-----------------------------------------------------------------------------
TEST(ascent_commands, register_no_name_void)
{
    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);

    // An error should be thrown due to not including a name
    ascent::register_callback("", void_callback_1);

    Node actions;
    std::string msg = "An example of registering a void callback";
                      " without a callback name.";
    ASCENT_ACTIONS_DUMP(actions, std::string("register_no_name_void"), msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_commands, register_no_name_bool)
{
    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);

    // An error should be thrown due to not including a name
    ascent::register_callback("", bool_callback_1);

    Node actions;
        std::string msg = "An example of registering a bool callback";
                          " without a callback name.";
    ASCENT_ACTIONS_DUMP(actions, std::string("register_no_name_bool"), msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_commands, register_void_callbacks)
{
    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);

    // No error should be thrown
    ascent::register_callback("void_callback_1", void_callback_1);
    ascent::register_callback("void_callback_2", void_callback_2);

    Node actions;
    std::string msg = "An example of registering void callbacks.";
    ASCENT_ACTIONS_DUMP(actions, std::string("register_void_callbacks"), msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_commands, register_bool_callbacks)
{
    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);

    // No error should be thrown
    ascent::register_callback("bool_callback_1", bool_callback_1);
    ascent::register_callback("bool_callback_2", bool_callback_2);

    Node actions;
    std::string msg = "An example of registering bool callbacks.";
    ASCENT_ACTIONS_DUMP(actions, std::string("register_bool_callbacks"), msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_commands, register_same_callback_twice_mixed)
{
    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);

    // An error should be thrown for reregistering the callback name
    ascent::register_callback("test_callback", void_callback_1);
    ascent::register_callback("test_callback", bool_callback_1);

    Node actions;
    std::string msg = "An example of registering a void and bool callback with";
                      " the same names.";
    ASCENT_ACTIONS_DUMP(actions, std::string("register_same_callback_twice_mixed"), msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_commands, register_same_void_callback_twice)
{
    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);

    // An error should be thrown due to registering the same name twice
    ascent::register_callback("void_callback", void_callback_1);
    ascent::register_callback("void_callback", void_callback_2);

    Node actions;
    std::string msg = "An example of registering a void callback twice with";
                      " the same names.";
    ASCENT_ACTIONS_DUMP(actions, std::string("register_same_void_callback_twice"), msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_commands, register_same_bool_callback_twice)
{
    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);

    // An error should be thrown due to registering the same name twice
    ascent::register_callback("bool_callback", bool_callback_1);
    ascent::register_callback("bool_callback", bool_callback_2);

    Node actions;
    std::string msg = "An example of registering a bool callback twice with";
                      " the same names.";
    ASCENT_ACTIONS_DUMP(actions, std::string("register_same_bool_callback_twice"), msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_commands, direct_void_callback_invocation)
{
    // Register callbacks, should not require an Ascent instance
    ascent::register_callback("void_callback_1", void_callback_1);

    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);


    Node params;
    Node output;
    ascent::execute_callback("void_callback_1", params, output);

    // We didn't pass a parameter, so we expect param_was_passed to be false
    bool has_output = false;
    if (output.has_path("param_was_passed"))
    {
        has_output = params["param_was_passed"].to_uint8();
    }
    EXPECT_FALSE(has_output);

    params.reset();
    output.reset();
    params["example_param"] = 1234;
    ascent::execute_callback("void_callback_1", params, output);

    // We passed a parameter, so we expect param_was_passed to be true
    if (output.has_path("param_was_passed"))
    {
        has_output = params["param_was_passed"].to_uint8();
    }
    EXPECT_TRUE(has_output);

    Node actions;
    std::string msg = "An example of invoking void callbacks directly";
    ASCENT_ACTIONS_DUMP(actions, std::string("direct_void_callback_invocation"), msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_commands, actions_shell_command_invocation)
{
    string file_name = "actions_shell_command_invocation_test";
    string output_path = prepare_output_dir();
    string file_path = conduit::utils::join_file_path(output_path,file_name);
    // remove old file
    if(conduit::utils::is_file(file_path))
    {
        conduit::utils::remove_file(file_path);
    }

    //
    // Create the actions.
    //
    Node actions;

    conduit::Node commands;
    string shell_command = "touch " + file_path;
    commands["c1/params/shell_command"] = shell_command;
    commands["c1/params/mpi_behavior"] = "root";

    conduit::Node &add_commands = actions.append();
    add_commands["action"] = "add_commands";
    add_commands["commands"] = commands;
    actions.print();

    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.execute(actions);

    // This file should have been created by the shell command invocation
    EXPECT_TRUE(conduit::utils::is_file(file_path));

    std::string msg = "An example of directly invoking a shell command"
                      " from Ascent actions.";
    ASCENT_ACTIONS_DUMP(actions, std::string("actions_shell_command_invocation"), msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_commands, actions_void_callback_invocation)
{
    //
    // Create the actions.
    //
    Node actions;

    conduit::Node commands;
    commands["c1/params/callback"] = "void_callback_2";
    commands["c1/params/mpi_behavior"] = "all";

    conduit::Node &add_commands = actions.append();
    add_commands["action"] = "add_commands";
    add_commands["commands"] = commands;
    actions.print();

    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);

    ascent::register_callback("void_callback_2", void_callback_2);

    // void_callback_2 should make example_bool true
    EXPECT_FALSE(example_bool);
    ascent.execute(actions);
    EXPECT_TRUE(example_bool);

    std::string msg = "An example of invoking a void callback from Ascent actions.";
    ASCENT_ACTIONS_DUMP(actions, std::string("actions_void_callback_invocation"), msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_commands, bool_callback_trigger_condition)
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

    std::string condition = "bool_callback_1";
    conduit::Node triggers;
    triggers["t1/params/callback"] = condition;
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

    ascent::register_callback("bool_callback_1", bool_callback_1);

    ascent.publish(data);
    ascent.execute(actions);

    conduit::Node info;
    ascent.info(info);
    std::string path = "expressions/" + condition + "/100/value";
    info["expressions"].print();
    EXPECT_TRUE(info[path].to_int32() == 1);
    std::string msg = "An example of triggering actions using a bool callback.";
    ASCENT_ACTIONS_DUMP(actions, std::string("bool_callback_trigger_condition"), msg);

    ascent.close();
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


