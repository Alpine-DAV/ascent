//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//-----------------------------------------------------------------------------
///
/// file: t_ascent_steering.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <ascent.hpp>
#include <conduit_blueprint.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"

using namespace std;
using namespace conduit;
using namespace ascent;

// Callback names
const std::string void_callback_1_name = "void_callback_1";
const std::string void_callback_2_name = "void_callback_2";
const std::string bool_callback_1_name = "bool_callback_1";
const std::string bool_callback_2_name = "bool_callback_2";

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
TEST(ascent_steering, get_void_callback_names_empty)
{
    // Reset callbacks
    ascent::reset_callbacks();

    // Empty vector of strings
    std::vector<std::string> callback_names;

    // Get void callback names
    ascent::get_void_callbacks(callback_names);

    // There should be no registered callbacks, meaning the vector is empty
    EXPECT_TRUE(callback_names.size() == 0);
}

//-----------------------------------------------------------------------------
TEST(ascent_steering, get_void_callback_names_nonempty)
{
    // Reset callbacks
    ascent::reset_callbacks();

    // Register some void callbacks
    ascent::register_callback(void_callback_1_name, void_callback_1);
    ascent::register_callback(void_callback_2_name, void_callback_2);

    // Empty vector of strings
    std::vector<std::string> callback_names;

    // Get void callback names
    ascent::get_void_callbacks(callback_names);

    // The vector should contain 2 strings with the exact names we expect
    EXPECT_TRUE(callback_names.size() == 2);
    EXPECT_TRUE(callback_names[0] == void_callback_1_name);
    EXPECT_TRUE(callback_names[1] == void_callback_2_name);
}

//-----------------------------------------------------------------------------
TEST(ascent_steering, get_bool_callback_names_empty)
{
    // Reset callbacks
    ascent::reset_callbacks();

    // Empty vector of strings
    std::vector<std::string> callback_names;

    // Get bool callback names
    ascent::get_bool_callbacks(callback_names);

    // There should be no registered callbacks, meaning the vector is empty
    EXPECT_TRUE(callback_names.size() == 0);
}

//-----------------------------------------------------------------------------
TEST(ascent_steering, get_bool_callback_names_nonempty)
{
    // Reset callbacks
    ascent::reset_callbacks();

    // Register some bool callbacks
    ascent::register_callback(bool_callback_1_name, bool_callback_1);
    ascent::register_callback(bool_callback_2_name, bool_callback_2);

    // Empty vector of strings
    std::vector<std::string> callback_names;

    // Get bool callback names
    ascent::get_bool_callbacks(callback_names);

    // The vector should contain 2 strings with the exact names we expect
    EXPECT_TRUE(callback_names.size() == 2);
    EXPECT_TRUE(callback_names[0] == bool_callback_1_name);
    EXPECT_TRUE(callback_names[1] == bool_callback_2_name);
}

//-----------------------------------------------------------------------------
TEST(ascent_steering, terminal_interface_null_params)
{
    //
    // Create the actions.
    //
    Node actions;

    conduit::Node extracts;
    extracts["e1/type"] = "steering";
    extracts["e1/params/explicit_command"] = "";
    // This erases the value without erasing the path, effectively passing
    // null as input
    extracts["e1/params/explicit_command"].reset();

    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;
    actions.print();

    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    EXPECT_NO_THROW(ascent.execute(actions));

    std::string msg = "An example of passing null input to the terminal"
                      " steering interface from Ascent actions.";
    ASCENT_ACTIONS_DUMP(actions,
                        std::string("terminal_interface_null_params"),
                        msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_steering, terminal_interface_blank_input)
{
    //
    // Create the actions.
    //
    Node actions;

    conduit::Node extracts;
    extracts["e1/type"] = "steering";
    extracts["e1/params/explicit_command"] = "";

    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;
    actions.print();

    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    EXPECT_NO_THROW(ascent.execute(actions));

    std::string msg = "An example of passing blank input to the terminal"
                      " steering interface from Ascent actions.";
    ASCENT_ACTIONS_DUMP(actions,
                        std::string("terminal_interface_blank_input"),
                        msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_steering, terminal_interface_numeric_input)
{
    //
    // Create the actions.
    //
    Node actions;

    conduit::Node extracts;
    extracts["e1/type"] = "steering";
    extracts["e1/params/explicit_command"] = 1234;

    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;
    actions.print();

    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    EXPECT_NO_THROW(ascent.execute(actions));

    std::string msg = "An example of passing numeric input to the terminal"
                      " steering interface from Ascent actions.";
    ASCENT_ACTIONS_DUMP(actions,
                        std::string("terminal_interface_numeric_input"),
                        msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_steering, terminal_interface_invalid_input)
{
    //
    // Create the actions.
    //
    Node actions;

    conduit::Node extracts;
    extracts["e1/type"] = "steering";
    extracts["e1/params/explicit_command"] = "this is an invalid command";

    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;
    actions.print();

    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    EXPECT_NO_THROW(ascent.execute(actions));

    std::string msg = "An example of passing invalid input to the terminal"
                      " steering interface from Ascent actions.";
    ASCENT_ACTIONS_DUMP(actions,
                        std::string("terminal_interface_invalid_input"),
                        msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_steering, terminal_interface_exit_input)
{
    //
    // Create the actions.
    //
    Node actions;

    conduit::Node extracts;
    extracts["e1/type"] = "steering";
    extracts["e1/params/explicit_command"] = "exit";

    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;
    actions.print();

    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    EXPECT_NO_THROW(ascent.execute(actions));

    std::string msg = "An example of passing 'exit' to the terminal steering"
                      " interface from Ascent actions.";
    ASCENT_ACTIONS_DUMP(actions,
                        std::string("terminal_interface_exit_input"),
                        msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_steering, terminal_interface_help_input)
{
    //
    // Create the actions.
    //
    Node actions;

    conduit::Node extracts;
    extracts["e1/type"] = "steering";
    extracts["e1/params/explicit_command"] = "help";

    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;
    actions.print();

    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    EXPECT_NO_THROW(ascent.execute(actions));

    std::string msg = "An example of passing 'help' to the terminal steering"
                      " interface from Ascent actions.";
    ASCENT_ACTIONS_DUMP(actions,
                        std::string("terminal_interface_help_input"),
                        msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_steering, terminal_interface_list_empty_input)
{
    // Reset callbacks
    ascent::reset_callbacks();

    //
    // Create the actions.
    //
    Node actions;

    conduit::Node extracts;
    extracts["e1/type"] = "steering";
    extracts["e1/params/explicit_command"] = "list";

    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;
    actions.print();

    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    EXPECT_NO_THROW(ascent.execute(actions));

    std::string msg = "An example of passing 'list' to the terminal steering"
                      " interface from Ascent actions.";
    ASCENT_ACTIONS_DUMP(actions,
                        std::string("terminal_interface_list_empty_input"),
                        msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_steering, terminal_interface_list_void_input)
{
    // Reset callbacks
    ascent::reset_callbacks();

    // Register some callbacks
    ascent::register_callback(void_callback_1_name, void_callback_1);
    ascent::register_callback(void_callback_2_name, void_callback_2);

    //
    // Create the actions.
    //
    Node actions;

    conduit::Node extracts;
    extracts["e1/type"] = "steering";
    extracts["e1/params/explicit_command"] = "list";

    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;
    actions.print();

    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    EXPECT_NO_THROW(ascent.execute(actions));

    std::string msg = "An example of passing 'list' to the terminal steering"
                      " interface from Ascent actions.";
    ASCENT_ACTIONS_DUMP(actions,
                        std::string("terminal_interface_list_void_input"),
                        msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_steering, terminal_interface_list_bool_input)
{
    // Reset callbacks
    ascent::reset_callbacks();

    // Register some callbacks
    ascent::register_callback(bool_callback_1_name, bool_callback_1);
    ascent::register_callback(bool_callback_2_name, bool_callback_2);

    //
    // Create the actions.
    //
    Node actions;

    conduit::Node extracts;
    extracts["e1/type"] = "steering";
    extracts["e1/params/explicit_command"] = "list";

    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;
    actions.print();

    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    EXPECT_NO_THROW(ascent.execute(actions));

    std::string msg = "An example of passing 'list' to the terminal steering"
                      " interface from Ascent actions.";
    ASCENT_ACTIONS_DUMP(actions,
                        std::string("terminal_interface_list_bool_input"),
                        msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_steering, terminal_interface_list_mixed_input)
{
    // Reset callbacks
    ascent::reset_callbacks();

    // Register some callbacks
    ascent::register_callback(void_callback_1_name, void_callback_1);
    ascent::register_callback(void_callback_2_name, void_callback_2);
    ascent::register_callback(bool_callback_1_name, bool_callback_1);
    ascent::register_callback(bool_callback_2_name, bool_callback_2);

    //
    // Create the actions.
    //
    Node actions;

    conduit::Node extracts;
    extracts["e1/type"] = "steering";
    extracts["e1/params/explicit_command"] = "list";

    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;
    actions.print();

    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    EXPECT_NO_THROW(ascent.execute(actions));

    std::string msg = "An example of passing 'list' to the terminal steering"
                      " interface from Ascent actions.";
    ASCENT_ACTIONS_DUMP(actions,
                        std::string("terminal_interface_list_mixed_input"),
                        msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_steering, terminal_interface_param_empty_input)
{
    //
    // Create the actions.
    //
    Node actions;

    conduit::Node extracts;
    extracts["e1/type"] = "steering";
    extracts["e1/params/explicit_command"] = "param";

    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;
    actions.print();

    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    EXPECT_NO_THROW(ascent.execute(actions));

    std::string msg = "An example of passing 'param' to the terminal steering"
                      " interface from Ascent actions.";
    ASCENT_ACTIONS_DUMP(actions,
                        std::string("terminal_interface_param_empty_input"),
                        msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_steering, terminal_interface_param_add_empty_input)
{
    //
    // Create the actions.
    //
    Node actions;

    conduit::Node extracts;
    extracts["e1/type"] = "steering";
    extracts["e1/params/explicit_command"] = "param velocity";

    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;
    actions.print();

    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    EXPECT_NO_THROW(ascent.execute(actions));

    std::string msg = "An example of passing 'param test' to the terminal"
                      " steering interface from Ascent actions.";
    ASCENT_ACTIONS_DUMP(actions,
                        std::string("terminal_interface_param_add_empty_input"),
                        msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_steering, terminal_interface_param_add_valid_numeric_input_1)
{
    //
    // Create the actions.
    //
    Node actions;

    conduit::Node extracts;
    extracts["e1/type"] = "steering";
    extracts["e1/params/explicit_command"] = "param velocity 2.2";

    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;
    actions.print();

    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    EXPECT_NO_THROW(ascent.execute(actions));

    std::string msg = "An example of passing 'param velocity 2.2' to the"
                      " terminal steering interface from Ascent actions.";
    ASCENT_ACTIONS_DUMP(actions,
                        std::string("terminal_interface_param_add_valid_numeric_input_1"),
                        msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_steering, terminal_interface_param_add_valid_numeric_input_2)
{
    //
    // Create the actions.
    //
    Node actions;

    conduit::Node extracts;
    extracts["e1/type"] = "steering";
    extracts["e1/params/explicit_command"] = "param velocity 2";

    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;
    actions.print();

    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    EXPECT_NO_THROW(ascent.execute(actions));

    std::string msg = "An example of passing 'param velocity 2' to the"
                      " terminal steering interface from Ascent actions.";
    ASCENT_ACTIONS_DUMP(actions,
                        std::string("terminal_interface_param_add_valid_numeric_input_2"),
                        msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_steering, terminal_interface_param_add_invalid_numeric_input)
{
    //
    // Create the actions.
    //
    Node actions;

    conduit::Node extracts;
    extracts["e1/type"] = "steering";
    extracts["e1/params/explicit_command"] = "param velocity 2.2e22";

    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;
    actions.print();

    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    EXPECT_NO_THROW(ascent.execute(actions));

    std::string msg = "An example of passing 'param velocity 2.2e22' to the"
                      " terminal steering interface from Ascent actions.";
    ASCENT_ACTIONS_DUMP(actions,
                        std::string("terminal_interface_param_add_invalid_numeric_input"),
                        msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_steering, terminal_interface_param_add_string_input)
{
    //
    // Create the actions.
    //
    Node actions;

    conduit::Node extracts;
    extracts["e1/type"] = "steering";
    extracts["e1/params/explicit_command"] = "param velocity high";

    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;
    actions.print();

    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    EXPECT_NO_THROW(ascent.execute(actions));

    std::string msg = "An example of passing 'param velocity high' to the"
                      " terminal steering interface from Ascent actions.";
    ASCENT_ACTIONS_DUMP(actions,
                        std::string("terminal_interface_param_add_string_input"),
                        msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_steering, terminal_interface_param_delete_input)
{
    //
    // Create the actions.
    //
    Node actions;

    conduit::Node extracts;
    extracts["e1/type"] = "steering";
    extracts["e1/params/explicit_command"] = "param velocity 2\nparam delete"
                                             " velocity";

    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;
    actions.print();

    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    EXPECT_NO_THROW(ascent.execute(actions));

    std::string msg = "An example of passing 'param velocity 2\nparam delete"
                      " velocity' to the terminal steering interface from"
                      " Ascent actions.";
    ASCENT_ACTIONS_DUMP(actions,
                        std::string("terminal_interface_param_delete_input"),
                        msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_steering, terminal_interface_param_reset_input)
{
    //
    // Create the actions.
    //
    Node actions;

    conduit::Node extracts;
    extracts["e1/type"] = "steering";
    extracts["e1/params/explicit_command"] = "param test1 1\nparam test2"
                                             " 2\nparam reset";

    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;
    actions.print();

    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    EXPECT_NO_THROW(ascent.execute(actions));

    std::string msg = "An example of passing 'param test1 1\nparam test2"
                      " 2\nparam reset' to the terminal steering interface"
                      " from Ascent actions.";
    ASCENT_ACTIONS_DUMP(actions,
                        std::string("terminal_interface_param_reset_input"),
                        msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_steering, terminal_interface_run_empty_input)
{
    // Reset callbacks
    ascent::reset_callbacks();

    //
    // Create the actions.
    //
    Node actions;

    conduit::Node extracts;
    extracts["e1/type"] = "steering";
    extracts["e1/params/explicit_command"] = "run";

    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;
    actions.print();

    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    EXPECT_NO_THROW(ascent.execute(actions));

    std::string msg = "An example of passing 'run' to the"
                      " terminal steering interface from Ascent actions.";
    ASCENT_ACTIONS_DUMP(actions,
                        std::string("terminal_interface_run_empty_input"),
                        msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_steering, terminal_interface_run_invalid_input)
{
    // Reset callbacks
    ascent::reset_callbacks();

    //
    // Create the actions.
    //
    Node actions;

    conduit::Node extracts;
    extracts["e1/type"] = "steering";
    extracts["e1/params/explicit_command"] = "run my_callback";

    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;
    actions.print();

    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    EXPECT_NO_THROW(ascent.execute(actions));

    std::string msg = "An example of passing 'run my_callback' to the"
                      " terminal steering interface from Ascent actions.";
    ASCENT_ACTIONS_DUMP(actions,
                        std::string("terminal_interface_run_invalid_input"),
                        msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_steering, terminal_interface_run_void_input)
{
    // Reset callbacks
    ascent::reset_callbacks();

    // Register some callbacks
    ascent::register_callback(void_callback_1_name, void_callback_1);
    ascent::register_callback(void_callback_2_name, void_callback_2);
    ascent::register_callback(bool_callback_1_name, bool_callback_1);
    ascent::register_callback(bool_callback_2_name, bool_callback_2);

    //
    // Create the actions.
    //
    Node actions;

    conduit::Node extracts;
    extracts["e1/type"] = "steering";
    extracts["e1/params/explicit_command"] = "run void_callback_1";

    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;
    actions.print();

    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    EXPECT_NO_THROW(ascent.execute(actions));

    std::string msg = "An example of passing 'run void_callback_1' to the"
                      " terminal steering interface from Ascent actions.";
    ASCENT_ACTIONS_DUMP(actions,
                        std::string("terminal_interface_run_void_input"),
                        msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_steering, terminal_interface_run_void_with_param_input)
{
    // Reset callbacks
    ascent::reset_callbacks();

    // Register some callbacks
    ascent::register_callback(void_callback_1_name, void_callback_1);
    ascent::register_callback(void_callback_2_name, void_callback_2);
    ascent::register_callback(bool_callback_1_name, bool_callback_1);
    ascent::register_callback(bool_callback_2_name, bool_callback_2);

    //
    // Create the actions.
    //
    Node actions;

    conduit::Node extracts;
    extracts["e1/type"] = "steering";
    extracts["e1/params/explicit_command"] = "param example_param true\nrun"
                                             " void_callback_1";

    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;
    actions.print();

    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    EXPECT_NO_THROW(ascent.execute(actions));

    std::string msg = "An example of passing 'param example_param true\nrun"
                      " void_callback_1' to the terminal steering interface"
                      " from Ascent actions.";
    ASCENT_ACTIONS_DUMP(actions,
                        std::string("terminal_interface_run_void_with_param_input"),
                        msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_steering, terminal_interface_run_bool_input)
{
    // Reset callbacks
    ascent::reset_callbacks();

    // Register some callbacks
    ascent::register_callback(void_callback_1_name, void_callback_1);
    ascent::register_callback(void_callback_2_name, void_callback_2);
    ascent::register_callback(bool_callback_1_name, bool_callback_1);
    ascent::register_callback(bool_callback_2_name, bool_callback_2);

    //
    // Create the actions.
    //
    Node actions;

    conduit::Node extracts;
    extracts["e1/type"] = "steering";
    extracts["e1/params/explicit_command"] = "run bool_callback_1";

    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;
    actions.print();

    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    EXPECT_NO_THROW(ascent.execute(actions));

    std::string msg = "An example of passing 'run bool_callback_1' to the"
                      " terminal steering interface from Ascent actions.";
    ASCENT_ACTIONS_DUMP(actions,
                        std::string("terminal_interface_run_bool_input"),
                        msg);

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
