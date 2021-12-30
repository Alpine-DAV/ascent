//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_python_script_extract.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <ascent.hpp>

#include <iostream>
#include <conduit_blueprint.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"

using namespace std;
using namespace conduit;
using ascent::Ascent;

std::string py_script = "\n"
"# we treat everything as a multi_domain in ascent so grab child 0\n"
"v = ascent_data().child(0)\n"
"print(v['state'])\n"
"\n";

//-----------------------------------------------------------------------------
TEST(ascent_runtime, test_python_script_extract)
{
    //
    // Create the data.
    //
    Node data, verify_info;
    create_3d_example_dataset(data,32,0,1);
    data["state/cycle"] = 101;

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    //
    // Create the actions.
    //

    conduit::Node extracts;
    extracts["e1/type"]  = "python";
    extracts["e1/params/source"] = py_script;

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    actions.print();

    //
    // Run Ascent
    //

    Node ascent_opts;
    ascent_opts["ascent_info"] = "verbose";
    ascent_opts["exceptions"] = "forward";

    Ascent ascent;
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

}

//-----------------------------------------------------------------------------
TEST(ascent_runtime, test_python_script_extract_from_file)
{
    //
    // Create the data.
    //
    Node data, verify_info;
    create_3d_example_dataset(data,32,0,1);
    data["state/cycle"] = 101;

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    //
    // Create the actions.
    //

    // write out the test module to a file
    std::ofstream ofs;
    ofs.open("t_my_test_script.py");
    ofs << py_script;
    // in this case __file__ should be defined
    ofs << "assert __file__ == 't_my_test_script.py'\n";
    ofs.close();


    conduit::Node extracts;
    extracts["e1/type"]  = "python";
    extracts["e1/params/file"] = "t_my_test_script.py";

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    actions.print();

    //
    // Run Ascent
    //

    Node ascent_opts;
    ascent_opts["ascent_info"] = "verbose";
    ascent_opts["exceptions"] = "forward";

    Ascent ascent;
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

}


//-----------------------------------------------------------------------------
TEST(ascent_runtime, test_python_script_extract_from_bad_file)
{
    //
    // Create the data.
    //
    Node data, verify_info;
    create_3d_example_dataset(data,32,0,1);
    data["state/cycle"] = 101;

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    //
    // Create the actions.
    //

    // write out the test module to a file
    std::ofstream ofs;
    ofs.open("t_my_test_script.py");
    ofs << py_script;
    ofs.close();


    conduit::Node extracts;
    extracts["e1/type"]  = "python";
    extracts["e1/params/file"] = "/blarhg/very/bad/path";

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;
    actions.print();

    //
    // Run Ascent
    //

    Node ascent_opts;
    ascent_opts["ascent_info"] = "verbose";
    ascent_opts["exceptions"] = "forward";

    Ascent ascent;
    ascent.open(ascent_opts);
    ascent.publish(data);
    // we will get an error here
    EXPECT_THROW(ascent.execute(actions),
                 conduit::Error);
    ascent.close();

}




std::string py_script_mod_driver = "\n"
"import tout_my_test_module\n"
"print ('calling go from test module')\n"
"tout_my_test_module.go()\n";

std::string py_script_mod_src = "\n"
"import ascent_extract\n"
"\n"
"def go():\n"
"# we treat everything as a multi_domain in ascent so grab child 0\n"
"    v = ascent_extract.ascent_data().child(0)\n"
"    print(v['state'])\n"
"\n";

//-----------------------------------------------------------------------------
TEST(ascent_runtime, test_python_script_extract_import)
{
    //
    // Create the data.
    //
    Node data, verify_info;
    create_3d_example_dataset(data,32,0,1);
    data["state/cycle"] = 101;

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    // write out the test module to a file
    std::ofstream ofs;
    ofs.open("tout_my_test_module.py");
    ofs << py_script_mod_src;
    ofs.close();

    //
    // Create the actions.
    //

    conduit::Node extracts;
    extracts["e1/type"]  = "python";
    extracts["e1/params/source"] = py_script_mod_driver;

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    actions.print();

    //
    // Run Ascent
    //

    Node ascent_opts;
    ascent_opts["ascent_info"] = "verbose";
    ascent_opts["exceptions"] = "forward";

    Ascent ascent;
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

}


// This demos using the ascent python api inside of ascent ...
std::string py_script_inception = "\n"
"import conduit\n"
"import ascent\n"
"# we treat everything as a multi_domain in ascent so grab child 0\n"
"n_mesh = ascent_data().child(0)\n"
"a = ascent.Ascent()\n"
"a.open()\n"
"a.publish(n_mesh)\n"
"actions = conduit.Node()\n"
"scenes  = conduit.Node()\n"
"scenes['s1/plots/p1/type'] = 'pseudocolor'\n"
"scenes['s1/plots/p1/field'] = 'radial_vert'\n"
"scenes['s1/image_prefix'] = 'tout_python_extract_inception' \n"
"add_act =actions.append()\n"
"add_act['action'] = 'add_scenes'\n"
"add_act['scenes'] = scenes\n"
"a.execute(actions)\n"
"a.close()\n"
"\n";


//-----------------------------------------------------------------------------
TEST(ascent_runtime, test_python_extract_inception)
{
    //
    // Create the data.
    //
    Node data, verify_info;
    create_3d_example_dataset(data,32,0,1);
    data["state/cycle"] = 101;

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    //
    // Create the actions.
    //

    conduit::Node extracts;
    extracts["e1/type"]  = "python";
    extracts["e1/params/source"] = py_script_inception;

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    actions.print();

    //
    // Run Ascent
    //

    Node ascent_opts;
    ascent_opts["ascent_info"] = "verbose";
    ascent_opts["exceptions"] = "forward";

    Ascent ascent;
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

}

