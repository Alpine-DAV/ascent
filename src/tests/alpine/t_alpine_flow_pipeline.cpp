//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-716457
// 
// All rights reserved.
// 
// This file is part of Alpine. 
// 
// For details, see: http://software.llnl.gov/alpine/.
// 
// Please also read alpine/LICENSE
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
/// file: t_alpine_empty_pipeline.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <alpine.hpp>

#include <iostream>
#include <math.h>
#include <sstream>

#include <conduit_blueprint.hpp>

#include "t_config.hpp"
#include "t_alpine_test_utils.hpp"


using namespace std;
using namespace conduit;
using namespace alpine;


#include <alpine_flow.hpp>

// ----- //
// This tests that we can create a custom filter, register it and use it
// in the flow pipeline.
class InspectFilter: public flow::Filter
{
public:
    InspectFilter(): flow::Filter()
    {}
    ~InspectFilter()
    {}
        
    void declare_interface(Node &i)
    {
        i["type_name"] = "inspect";
        i["port_names"].append().set("in");
        i["output_port"] = "true";
    }
    
    
    void execute()
    {
        if(!input(0).check_type<Node>())
        {
            ALPINE_ERROR("Error, input is not a conduit node!");
        }
        
        Node *n = input<Node>(0);
        
        ALPINE_INFO("Total Strided Bytes = " << n->total_strided_bytes());
        
        set_output<Node>(n);
    }

};



//-----------------------------------------------------------------------------
TEST(alpine_flow_pipeline, test_flow_pipeline)
{
    
    if(!flow::Workspace::supports_filter_type<InspectFilter>())
    {
        flow::Workspace::register_filter_type<InspectFilter>();
    }

    //
    // Create example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("quads",100,100,0,data);
    
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));
    verify_info.print();
    
    Node actions;
    Node &a_add_insp = actions.append();
    a_add_insp["action"] = "add_filter";
    a_add_insp["type_name"]  = "inspect";
    a_add_insp["name"] = "fi";
    
    Node &a_conn = actions.append();
    
    a_conn["action"] = "connect";
    a_conn["src"]  = ":source";
    a_conn["dest"] = "fi";
    a_conn["port"] = "in";
    
    Node &a_exec = actions.append();
    a_exec["action"] = "execute";
    
    actions.print();

    // we want the "flow" pipeline
    Node open_opts;
    open_opts["pipeline/type"] = "flow";
    
    //
    // Run Alpine
    //
    Alpine alpine;
    alpine.Open(open_opts);
    alpine.Publish(data);
    alpine.Execute(actions);
    alpine.Close();
}




//-----------------------------------------------------------------------------
TEST(alpine_flow_pipeline, test_flow_pipeline_reuse_network)
{
    
    if(!flow::Workspace::supports_filter_type<InspectFilter>())
    {
        flow::Workspace::register_filter_type<InspectFilter>();
    }
    
    Node actions;
    Node &a_add_insp = actions.append();
    a_add_insp["action"] = "add_filter";
    a_add_insp["type_name"]  = "inspect";
    a_add_insp["name"] = "fi";
    
    Node &a_conn = actions.append();
    
    a_conn["action"] = "connect";
    a_conn["src"]  = ":source";
    a_conn["dest"] = "fi";
    a_conn["port"] = "in";
    
    Node &a_exec = actions.append();
    a_exec["action"] = "execute";
    
    actions.print();

    // we want the "flow" pipeline
    Node open_opts;
    open_opts["pipeline/type"] = "flow";
    
    //
    // Run Alpine
    //
    Alpine alpine;
    alpine.Open(open_opts);

    //
    // Create example mesh.
    //
    Node data;
    conduit::blueprint::mesh::examples::braid("quads",100,100,0,data);
    alpine.Publish(data);
    alpine.Execute(actions);

    // publish new data, but use the same data flow network.
    conduit::blueprint::mesh::examples::braid("quads",50,50,0,data);
    alpine.Publish(data);
    actions.reset();
    actions.append()["action"] = "execute";

    alpine.Publish(data);
    alpine.Execute(actions);
    
    alpine.Close();
}

