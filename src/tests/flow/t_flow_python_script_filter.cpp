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
/// file: t_flow_python_script_filter.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <flow.hpp>
#include <flow_python_script_filter.hpp>

#include <iostream>
#include <math.h>

#include "t_config.hpp"
#include "t_utils.hpp"



using namespace std;
using namespace conduit;
using namespace ascent;
using namespace flow;


//-----------------------------------------------------------------------------
class SrcFilter: public Filter
{
public:
    SrcFilter()
    : Filter()
    {}

    virtual ~SrcFilter()
    {}


    virtual void declare_interface(Node &i)
    {
        i["type_name"]   = "src";
        i["output_port"] = "true";
        i["port_names"] = DataType::empty();
        i["default_params"]["value"].set((int)0);
    }


    virtual void execute()
    {
        int val = params()["value"].value();

        // set output
        Node *res = new Node();
        res->set(val);
        set_output<Node>(res);

        // the registry will take care of deleting the data
        // when all consuming filters have executed.
        ASCENT_INFO("exec: " << name() << " result = " << res->to_json());
    }
};


//-----------------------------------------------------------------------------
TEST(flow_python_script_filter, simple_execute)
{
    flow::filters::register_builtin();

    Workspace::register_filter_type<SrcFilter>();

    Workspace w;

    Node src_params;
    src_params["value"] = 21;

    w.graph().add_filter("src","v",src_params);

    Node py_params;
    py_params["source"] = "val = flow_input().value() * 2\nprint(val)\nflow_set_output(val)";

    w.graph().add_filter("python_script","py", py_params);

    // // src, dest, port
    w.graph().connect("v","py","in");
    //
    w.print();
    //
    w.execute();

    Workspace::clear_supported_filter_types();
}

//-----------------------------------------------------------------------------
TEST(flow_python_script_filter, simple_execute_file)
{
    flow::filters::register_builtin();

    Workspace::register_filter_type<SrcFilter>();

    Workspace w;

    Node src_params;
    src_params["value"] = 21;

    string output_path = prepare_output_dir();

    string script_fname = conduit::utils::join_file_path(output_path,
                                                         "tout_test_flow_filter_script.py");

    ofstream ofs;
    ofs.open(script_fname);
    ofs << "val = flow_input().value() * 2\nprint(val)\nflow_set_output(val)";
    ofs.close();

    w.graph().add_filter("src","v",src_params);

    Node py_params;

    py_params["file"] = script_fname;

    w.graph().add_filter("python_script","py", py_params);

    // // src, dest, port
    w.graph().connect("v","py","in");
    //
    w.print();
    //
    w.execute();

    Workspace::clear_supported_filter_types();
}

//-----------------------------------------------------------------------------
TEST(flow_python_script_filter, simple_execute_bad_file)
{
    flow::filters::register_builtin();

    Workspace::register_filter_type<SrcFilter>();

    Workspace w;

    Node src_params;
    src_params["value"] = 21;

    string output_path = prepare_output_dir();

    string script_fname = "/blargh/path/to/bad/script.py";
        
    w.graph().add_filter("src","v",src_params);

    Node py_params;

    py_params["file"] = script_fname;

    w.graph().add_filter("python_script","py", py_params);

    // // src, dest, port
    w.graph().connect("v","py","in");
    //
    w.print();

    EXPECT_THROW(w.execute(),
                 conduit::Error);

    Workspace::clear_supported_filter_types();
}


//-----------------------------------------------------------------------------
TEST(flow_python_script_filter, exe_override_interface_func_names)
{
    flow::filters::register_builtin();

    Workspace::register_filter_type<SrcFilter>();

    Workspace w;

    Node src_params;
    src_params["value"] = 21;

    w.graph().add_filter("src","v",src_params);

    Node py_params;
    // test customized input() and set_output() names
    py_params["interface/input"] = "give_me_data";
    py_params["interface/set_output"] = "here_is_some_data";
    py_params["interpreter/reset"] = "true";
    py_params["source"] = "val = give_me_data().value() * 2\nprint(val)\nhere_is_some_data(val)";

    w.graph().add_filter("python_script","py", py_params);

    // // src, dest, port
    w.graph().connect("v","py","in");

    w.print();
    w.execute();

    Workspace::clear_supported_filter_types();
}


//-----------------------------------------------------------------------------
TEST(flow_python_script_filter, exe_override_interface_mod_and_func_names)
{
    flow::filters::register_builtin();

    Workspace::register_filter_type<SrcFilter>();

    Workspace w;

    Node src_params;
    src_params["value"] = 42;

    w.graph().add_filter("src","v",src_params);

    // we will change the name of the binding module to be called mine

    std::ostringstream py_src_oss;
    py_src_oss << "import mine\n"
               << "val = mine.give_me_data().value() * 2\n"
               << "print(val)\n"
               << "mine.here_is_some_data(val)\n";

    Node py_params;
    // test customized mod name, and input() and set_output() names
    py_params["interface/module"] = "mine";
    py_params["interface/input"] = "give_me_data";
    py_params["interface/set_output"] = "here_is_some_data";
    py_params["interpreter/reset"] = "true";
    py_params["source"] = py_src_oss.str();


    w.graph().add_filter("python_script","py", py_params);

    // // src, dest, port
    w.graph().connect("v","py","in");

    w.print();
    w.execute();

    Workspace::clear_supported_filter_types();
}







