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
    verify_info.print();

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
    verify_info.print();

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
    verify_info.print();

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
    verify_info.print();

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

