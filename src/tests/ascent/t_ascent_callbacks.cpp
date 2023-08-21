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

bool example_bool = true;

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
TEST(ascent_callbacks, register_no_name_void)
{
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);

    // An error should be thrown due to not including a name
    ascent.register_callback("", void_callback_1);

    Node actions;
    std::string msg = "An example of registering a void callback";
                      " without a callback name.";
    ASCENT_ACTIONS_DUMP(actions, std::string("register_no_name_void"), msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_callbacks, register_no_name_bool)
{
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);

    // An error should be thrown due to not including a name
    ascent.register_callback("", bool_callback_1);

    Node actions;
        std::string msg = "An example of registering a bool callback";
                          " without a callback name.";
    ASCENT_ACTIONS_DUMP(actions, std::string("register_no_name_bool"), msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_callbacks, register_void_callbacks)
{
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);

    // No error should be thrown
    ascent.register_callback("void_callback_1", void_callback_1);
    ascent.register_callback("void_callback_2", void_callback_2);

    Node actions;
    std::string msg = "An example of registering void callbacks.";
    ASCENT_ACTIONS_DUMP(actions, std::string("register_void_callbacks"), msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_callbacks, register_bool_callbacks)
{
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);

    // No error should be thrown
    ascent.register_callback("bool_callback_1", bool_callback_1);
    ascent.register_callback("bool_callback_2", bool_callback_2);

    Node actions;
    std::string msg = "An example of registering bool callbacks.";
    ASCENT_ACTIONS_DUMP(actions, std::string("register_bool_callbacks"), msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_callbacks, register_same_callback_twice_mixed)
{
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);

    // An error should be thrown for reregistering the callback name
    ascent.register_callback("test_callback", void_callback_1);
    ascent.register_callback("test_callback", bool_callback_1);

    Node actions;
    std::string msg = "An example of registering a void and bool callback with";
                      " the same names.";
    ASCENT_ACTIONS_DUMP(actions, std::string("register_same_callback_twice_mixed"), msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_callbacks, register_same_void_callback_twice)
{
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);

    // An error should be thrown due to registering the same name twice
    ascent.register_callback("void_callback", void_callback_1);
    ascent.register_callback("void_callback", void_callback_2);

    Node actions;
    std::string msg = "An example of registering a void callback twice with";
                      " the same names.";
    ASCENT_ACTIONS_DUMP(actions, std::string("register_same_void_callback_twice"), msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_callbacks, register_same_bool_callback_twice)
{
    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);

    // An error should be thrown due to registering the same name twice
    ascent.register_callback("bool_callback", bool_callback_1);
    ascent.register_callback("bool_callback", bool_callback_2);

    Node actions;
    std::string msg = "An example of registering a bool callback twice with";
                      " the same names.";
    ASCENT_ACTIONS_DUMP(actions, std::string("register_same_bool_callback_twice"), msg);

    ascent.close();
}

//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);

    result = RUN_ALL_TESTS();
    return result;
}


