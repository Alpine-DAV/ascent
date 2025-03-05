//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//-----------------------------------------------------------------------------
///
/// file: t_ascent_commands.cpp
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
bool example_bool_1 = false;
const bool example_bool_2 = true;

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
    if (example_bool_1)
    {
        example_bool_1 = false;
    }
    else
    {
        example_bool_1 = true;
    }
}

//-----------------------------------------------------------------------------
bool bool_callback_1()
{
    return example_bool_2;
}

//-----------------------------------------------------------------------------
bool bool_callback_2()
{
    return !example_bool_2;
}

//-----------------------------------------------------------------------------
TEST(ascent_commands, register_no_name_void)
{
    // Reset callbacks
    ascent::reset_callbacks();

    // An error should be thrown due to not including a name
    EXPECT_ANY_THROW(ascent::register_callback("", void_callback_1));
}

//-----------------------------------------------------------------------------
TEST(ascent_commands, register_no_name_bool)
{
    // Reset callbacks
    ascent::reset_callbacks();

    // An error should be thrown due to not including a name
    EXPECT_ANY_THROW(ascent::register_callback("", bool_callback_1));
}

//-----------------------------------------------------------------------------
TEST(ascent_commands, register_void_callbacks)
{
    // Reset callbacks
    ascent::reset_callbacks();

    // No error should be thrown
    EXPECT_NO_THROW(ascent::register_callback("void_callback_1", void_callback_1));
    EXPECT_NO_THROW(ascent::register_callback("void_callback_2", void_callback_2));
}

//-----------------------------------------------------------------------------
TEST(ascent_commands, register_bool_callbacks)
{
    // Reset callbacks
    ascent::reset_callbacks();

    // No error should be thrown
    EXPECT_NO_THROW(ascent::register_callback("bool_callback_1", bool_callback_1));
    EXPECT_NO_THROW(ascent::register_callback("bool_callback_2", bool_callback_2));
}

//-----------------------------------------------------------------------------
TEST(ascent_commands, register_same_callback_twice_mixed)
{
    // Reset callbacks
    ascent::reset_callbacks();

    // Register callbacks
    EXPECT_NO_THROW(ascent::register_callback("test_callback", void_callback_1));

    // An error should be thrown for registering the same callback name twice
    EXPECT_ANY_THROW(ascent::register_callback("test_callback", bool_callback_1));
}

//-----------------------------------------------------------------------------
TEST(ascent_commands, register_same_void_callback_twice)
{
    // Reset callbacks
    ascent::reset_callbacks();

    // Register callbacks
    EXPECT_NO_THROW(ascent::register_callback("void_callback", void_callback_1));

    // An error should be thrown registering the same callback name twice
    EXPECT_ANY_THROW(ascent::register_callback("void_callback", void_callback_2));
}

//-----------------------------------------------------------------------------
TEST(ascent_commands, register_same_bool_callback_twice)
{
    // Reset callbacks
    ascent::reset_callbacks();

    // Register callbacks
    EXPECT_NO_THROW(ascent::register_callback("bool_callback", bool_callback_1));

    // An error should be thrown registering the same callback name twice
    EXPECT_ANY_THROW(ascent::register_callback("bool_callback", bool_callback_2));
}

//-----------------------------------------------------------------------------
TEST(ascent_commands, direct_void_callback_invocation)
{
    // Reset callbacks
    ascent::reset_callbacks();

    // Register callbacks
    ascent::register_callback("void_callback_1", void_callback_1);

    Node params;
    Node output;
    ascent::execute_callback("void_callback_1", params, output);

    // We didn't pass a parameter, so we expect param_was_passed to be false
    bool has_output = false;
    if (output.has_path("param_was_passed"))
    {
        has_output = output["param_was_passed"].to_uint8();
    }
    EXPECT_FALSE(has_output);

    params.reset();
    output.reset();

    params["example_param"] = 1234;
    ascent::execute_callback("void_callback_1", params, output);

    // We passed a parameter, so we expect param_was_passed to be true
    if (output.has_path("param_was_passed"))
    {
        has_output = output["param_was_passed"].to_uint8();
    }
    EXPECT_TRUE(has_output);
}

//-----------------------------------------------------------------------------
TEST(ascent_commands, actions_shell_command_invocation)
{
    string file_name = "actions_shell_command_invocation_test";
    string output_path = prepare_output_dir();
    string file_path = conduit::utils::join_file_path(output_path, file_name);

    // remove old file
    if (conduit::utils::is_file(file_path))
    {
        conduit::utils::remove_file(file_path);
    }

    //
    // Create the actions.
    //
    Node actions;

    conduit::Node commands;
#ifdef ASCENT_PLATFORM_WINDOWS
    string shell_command = "type nul > " + file_path;
#else
    string shell_command = "touch " + file_path;
#endif
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
    // Reset callbacks
    ascent::reset_callbacks();

    // Register callbacks
    ascent::register_callback("void_callback_2", void_callback_2);

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

    // void_callback_2 should make example_bool_1 true
    EXPECT_FALSE(example_bool_1);
    ascent.execute(actions);
    EXPECT_TRUE(example_bool_1);

    std::string msg = "An example of invoking a void callback from Ascent actions.";
    ASCENT_ACTIONS_DUMP(actions, std::string("actions_void_callback_invocation"), msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_commands, bool_callback_trigger_condition)
{
    // Reset callbacks
    ascent::reset_callbacks();

    // Register callbacks
    ascent::register_callback("bool_callback_1", bool_callback_1);

    // the vtkm runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if (n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
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

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data, verify_info));

    string output_path = prepare_output_dir();
    string trigger_file = conduit::utils::join_file_path(output_path, "callback_trigger_actions.yaml");
    string output_file = conduit::utils::join_file_path(output_path, "tout_callback_trigger_actions");

    // remove old trigger file
    if (conduit::utils::is_file(trigger_file))
    {
        conduit::utils::remove_file(trigger_file);
    }

    // remove old output file
    if (conduit::utils::is_file(output_file))
    {
        conduit::utils::remove_file(output_file);
    }

    //
    // Create trigger actions.
    //
    Node trigger_actions;

    conduit::Node extracts;

    extracts["e1/type"] = "relay";
    extracts["e1/params/path"] = output_file;

    conduit::Node &add_ext = trigger_actions.append();
    add_ext["action"] = "add_extracts";
    add_ext["extracts"] = extracts;

    trigger_actions.save(trigger_file);

    //
    // Create the actions.
    //
    Node actions;

    conduit::Node triggers;
    triggers["t1/params/callback"] = "bool_callback_1";
    triggers["t1/params/actions_file"] = trigger_file;

    conduit::Node &add_triggers = actions.append();
    add_triggers["action"] = "add_triggers";
    add_triggers["triggers"] = triggers;
    actions.print();

    //
    // Run Ascent
    //
    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);

    // Verify that the trigger fired, and that the output file was created
    EXPECT_TRUE(conduit::utils::is_file(output_file));

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
