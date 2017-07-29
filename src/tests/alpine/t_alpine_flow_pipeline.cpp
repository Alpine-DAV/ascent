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
/// file: t_alpine_flow_pipeline.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <alpine.hpp>

#include <iostream>
#include <math.h>
#include <sstream>

#include <conduit_blueprint.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"


using namespace std;
using namespace conduit;
using namespace alpine;


#include <flow.hpp>

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
    actions.append();
    actions[0]["action"] = "add_filter";
    actions[0]["type_name"]  = "inspect";
    actions[0]["name"] = "fi";
    
    actions.append();
    actions[1]["action"] = "connect";
    actions[1]["src"]  = ":source";
    actions[1]["dest"] = "fi";
    actions[1]["port"] = "in";
    
    actions.append()["action"] = "execute";
    actions.print();

    // we want the "flow" pipeline
    Node open_opts;
    open_opts["pipeline/type"] = "flow";
    
    //
    // Run Alpine
    //
    Alpine alpine;
    alpine.open(open_opts);
    alpine.publish(data);
    alpine.execute(actions);
    alpine.close();
}




//-----------------------------------------------------------------------------
TEST(alpine_flow_pipeline, test_flow_pipeline_reuse_network)
{
    
    if(!flow::Workspace::supports_filter_type<InspectFilter>())
    {
        flow::Workspace::register_filter_type<InspectFilter>();
    }
    
    Node actions;
    
    actions.append();
    actions[0]["action"] = "add_filter";
    actions[0]["type_name"]  = "inspect";
    actions[0]["name"] = "fi";
    
    actions.append();
    actions[1]["action"] = "connect";
    actions[1]["src"]  = ":source";
    actions[1]["dest"] = "fi";
    
    actions.append()["action"] = "execute";
    actions.print();

    // we want the "flow" pipeline
    Node open_opts;
    open_opts["pipeline/type"] = "flow";
    
    //
    // Run Alpine
    //
    Alpine alpine;
    alpine.open(open_opts);

    //
    // Create example mesh.
    //
    Node data;
    conduit::blueprint::mesh::examples::braid("quads",100,100,0,data);
    alpine.publish(data);
    alpine.execute(actions);

    // publish new data, but use the same data flow network.
    conduit::blueprint::mesh::examples::braid("quads",50,50,0,data);
    alpine.publish(data);
    actions.reset();
    actions.append()["action"] = "execute";

    alpine.publish(data);
    alpine.execute(actions);
    
    alpine.close();
}

//-----------------------------------------------------------------------------
TEST(alpine_flow_pipeline, test_flow_pipeline_relay_save)
{
    
    if(!flow::Workspace::supports_filter_type<InspectFilter>())
    {
        flow::Workspace::register_filter_type<InspectFilter>();
    }
    
    Node actions;
    actions.append();
    actions[0]["action"] = "add_filter";
    actions[0]["type_name"]  = "inspect";
    actions[0]["name"] = "fi";

    string output_file = conduit::utils::join_file_path(output_dir(),
                                                        "tout_flow_pipeline_relay_save.json");


    actions.append();
    actions[1]["action"] = "add_filter";
    actions[1]["type_name"]  = "relay_io_save";
    actions[1]["name"] = "out";
    actions[1]["params/path"] = "test.json";

    
    actions.append();
    actions[2]["action"] = "connect";
    actions[2]["src"]  = ":source";
    actions[2]["dest"] = "fi";
    
    
    actions.append();
    actions[3]["action"] = "connect";
    actions[3]["src"]  = "fi";
    actions[3]["dest"] = "out";
    
    actions.append()["action"] = "execute";
    actions.print();

    // we want the "flow" pipeline
    Node open_opts;
    open_opts["pipeline/type"] = "flow";

    //
    // Create example mesh.
    //
    Node data;
    conduit::blueprint::mesh::examples::braid("quads",10,10,0,data);
    
    //
    // Run Alpine
    //
    Alpine alpine;
    alpine.open(open_opts);
    alpine.publish(data);
    alpine.execute(actions);    
    alpine.close();
}

//-----------------------------------------------------------------------------
TEST(alpine_flow_pipeline, test_flow_pipeline_blueprint_verify)
{
    Node actions;
    
    actions.append();
    actions[0]["action"] = "add_filter";
    actions[0]["type_name"]  = "blueprint_verify";
    actions[0]["name"] = "v";
    actions[0]["params/protocol"] = "mesh";
    
    
    actions.append();
    actions[1]["action"] = "connect";
    actions[1]["src"]  = ":source";
    actions[1]["dest"] = "v";
    
    actions.append()["action"] = "execute";
    actions.print();

    // we want the "flow" pipeline
    Node open_opts;
    open_opts["pipeline/type"] = "flow";

    //
    // Create example mesh.
    //
    Node data;
    conduit::blueprint::mesh::examples::braid("quads",10,10,0,data);
    
    //
    // Run Alpine
    //
    Alpine alpine;
    alpine.open(open_opts);
    alpine.publish(data);
    alpine.execute(actions);

    // catch exception ... when verify fails
    
    data.reset();
    alpine.publish(data);
    EXPECT_THROW(alpine.execute(actions),conduit::Error);
    
    alpine.close();
    
}

//-----------------------------------------------------------------------------
TEST(alpine_flow_pipeline, test_flow_vtkm)
{
    Node actions;
    
    actions.append();
    actions[0]["action"] = "add_filter";
    actions[0]["type_name"]  = "blueprint_verify";
    actions[0]["name"] = "verify";
    actions[0]["params/protocol"] = "mesh";

    actions.append();
    actions[1]["action"] = "add_filter";
    actions[1]["type_name"]  = "ensure_vtkm";
    actions[1]["name"] = "vtkm_data";

    actions.append();
    actions[2]["action"] = "connect";
    actions[2]["src"]  = ":source";
    actions[2]["dest"] = "verify";

    actions.append();
    actions[3]["action"] = "connect";
    actions[3]["src"]  = "verify";
    actions[3]["dest"] = "vtkm_data";

    
    actions.append()["action"] = "execute";
    actions.print();

    // we want the "flow" pipeline
    Node open_opts;
    open_opts["pipeline/type"] = "flow";

    //
    // Create example mesh.
    //
    Node data;
    conduit::blueprint::mesh::examples::braid("quads",10,10,0,data);
    
    //
    // Run Alpine
    //
    Alpine alpine;
    alpine.open(open_opts);
    alpine.publish(data);
    
    Node n;
    alpine::about(n);

    // expect an error if we don't have vtkm support 
    if(n["pipelines/vtkm/status"].as_string() == "disabled")
    {
        EXPECT_THROW(alpine.execute(actions),conduit::Error);
    }
    else
    {
        alpine.execute(actions);
    }

    
    alpine.close();
    
}

//-----------------------------------------------------------------------------
TEST(alpine_flow_pipeline, test_flow_vtkh_render)
{   
    
    Node n;
    alpine::about(n);
    
    // vtk-h requires vtk-m, don't run this test if we don't have vtk-m support
    if(n["pipelines/vtkm/status"].as_string() == "disabled")
    {
        
        return;
    }
    
    
    Node actions;
    actions.append();
    actions[0]["action"] = "add_graph";
    Node &graph = actions[0]["graph"];

    graph["filters/verify/type_name"] = "blueprint_verify";
    graph["filters/verify/params/protocol"] = "mesh";

    graph["filters/vtkh_data/type_name"]    = "ensure_vtkh";

    graph["filters/vtkh_render/type_name"]  = "vtkh_raytracer";
    graph["filters/vtkh_render/params/field"]  = "braid";

    
    graph["connections"].append();
    graph["connections"][0]["src"] = ":source";
    graph["connections"][0]["dest"] = "verify";

    graph["connections"].append();
    graph["connections"][1]["src"] = "verify";
    graph["connections"][1]["dest"] = "vtkh_data";

    graph["connections"].append();
    graph["connections"][2]["src"] = "vtkh_data";
    graph["connections"][2]["dest"] = "vtkh_render";

    actions.append()["action"] = "execute";
    actions.print();

    // we want the "flow" pipeline
    Node open_opts;
    open_opts["pipeline/type"] = "flow";

    //
    // Create example mesh.
    //
    Node data;
    conduit::blueprint::mesh::examples::braid("hexs",10,10,10,data);
    
    //
    // Run Alpine
    //
    Alpine alpine;
    alpine.open(open_opts);
    alpine.publish(data);
    alpine.execute(actions);
    alpine.close();
    
}

//-----------------------------------------------------------------------------
TEST(alpine_flow_pipeline, test_flow_vtkh_filter)
{   
    
    Node n;
    alpine::about(n);
    
    // vtk-h requires vtk-m, don't run this test if we don't have vtk-m support
    if(n["pipelines/vtkm/status"].as_string() == "disabled")
    {
        
        return;
    }
    
    
    Node actions;
    actions.append();
    actions[0]["action"] = "add_graph";
    Node &graph = actions[0]["graph"];

    graph["filters/verify/type_name"] = "blueprint_verify";
    graph["filters/verify/params/protocol"] = "mesh";

    graph["filters/vtkh_data/type_name"]    = "ensure_vtkh";

    graph["filters/vtkh_isov/type_name"]  = "vtkh_marchingcubes";
    graph["filters/vtkh_isov/params/field"]  = "braid";
    
    Node &isov = graph["filters/vtkh_isov/params/iso_values"];
    isov.set_float64(0.0);

    graph["filters/vtkh_render/type_name"]  = "vtkh_raytracer";
    graph["filters/vtkh_render/params/field"]  = "braid";

    
    graph["connections"].append();
    graph["connections"][0]["src"] = ":source";
    graph["connections"][0]["dest"] = "verify";

    graph["connections"].append();
    graph["connections"][1]["src"] = "verify";
    graph["connections"][1]["dest"] = "vtkh_data";

    graph["connections"].append();
    graph["connections"][2]["src"] = "vtkh_data";
    graph["connections"][2]["dest"] = "vtkh_isov";


    graph["connections"].append();
    graph["connections"][3]["src"] = "vtkh_isov";
    graph["connections"][3]["dest"] = "vtkh_render";

    actions.append()["action"] = "execute";
    actions.print();

    // we want the "flow" pipeline
    Node open_opts;
    open_opts["pipeline/type"] = "flow";

    //
    // Create example mesh.
    //
    Node data;
    conduit::blueprint::mesh::examples::braid("hexs",10,10,10,data);
    
    //
    // Run Alpine
    //
    Alpine alpine;
    alpine.open(open_opts);
    alpine.publish(data);
    alpine.execute(actions);
    alpine.close();
    
}



//-----------------------------------------------------------------------------
TEST(alpine_flow_pipeline, test_flow_mesh_blueprint_hdf5_output)
{   
    
    Node n;
    alpine::about(n);
    
    
    Node actions;
    actions.append();
    actions[0]["action"] = "add_graph";
    Node &graph = actions[0]["graph"];

    graph["filters/verify/type_name"] = "blueprint_verify";
    graph["filters/verify/params/protocol"] = "mesh";

    graph["filters/save/type_name"] = "relay_io_save";
    graph["filters/save/params/protocol"]  = "blueprint/mesh/hdf5";
    graph["filters/save/params/path"]      = "tout_flow_mesh_bp_test";
    
    graph["connections"].list_of(Schema(DataType::empty()),2);
    graph["connections"][0]["src"]  = ":source";
    graph["connections"][0]["dest"] = "verify";

    graph["connections"][1]["src"]  = "verify";
    graph["connections"][1]["dest"] = "save";

    actions.append()["action"] = "execute";
    actions.print();

    // we want the "flow" pipeline
    Node open_opts;
    open_opts["pipeline/type"] = "flow";

    //
    // Create example mesh.
    //
    Node data;
    conduit::blueprint::mesh::examples::braid("hexs",10,10,10,data);
    
    //
    // Run Alpine
    //
    Alpine alpine;
    alpine.open(open_opts);
    alpine.publish(data);
    alpine.execute(actions);
    alpine.close();
    
}


//-----------------------------------------------------------------------------
TEST(alpine_flow_pipeline, test_flow_bulk_actions_1)
{
    // test add_graph, add_filters, and add_connections
    
        if(!flow::Workspace::supports_filter_type<InspectFilter>())
        {
            flow::Workspace::register_filter_type<InspectFilter>();
        }
    
        Node actions;
    
        Node graph;
        graph["filters/fi/type_name"] = "inspect";
        graph["connections"].append();

        graph["connections"][0]["src"] = ":source";
        graph["connections"][0]["dest"] = "fi";
    
    
        actions.append();
        actions[0]["action"] = "add_filters";
        actions[0]["filters"] = graph["filters"];
    
        actions.append();
        actions[1]["action"] = "add_connections";
        actions[1]["connections"] = graph["connections"];

        actions.append()["action"] = "execute";
        actions.print();

        // we want the "flow" pipeline
        Node open_opts;
        open_opts["pipeline/type"] = "flow";
    
        //
        // Run Alpine
        //
        Alpine alpine;
        alpine.open(open_opts);

        //
        // Create example mesh.
        //
        Node data;
        conduit::blueprint::mesh::examples::braid("quads",100,100,0,data);
        alpine.publish(data);
        alpine.execute(actions);
        alpine.close();
    
}


//-----------------------------------------------------------------------------
TEST(alpine_flow_pipeline, test_flow_bulk_actions_2)
{
    // test add_graph, add_filters, and add_connections
    
        if(!flow::Workspace::supports_filter_type<InspectFilter>())
        {
            flow::Workspace::register_filter_type<InspectFilter>();
        }
    
        Node actions;
        
    
        actions.append();
        actions[0]["action"] = "add_graph";
        Node &graph = actions[0]["graph"];

        graph["filters/fi/type_name"] = "inspect";

        graph["connections"].append();
        graph["connections"][0]["src"] = ":source";
        graph["connections"][0]["dest"] = "fi";

        actions.append()["action"] = "execute";
        actions.print();

        // we want the "flow" pipeline
        Node open_opts;
        open_opts["pipeline/type"] = "flow";

        //
        // Run Alpine
        //
        Alpine alpine;
        alpine.open(open_opts);

        //
        // Create example mesh.
        //
        Node data;
        conduit::blueprint::mesh::examples::braid("quads",100,100,0,data);
        alpine.publish(data);
        alpine.execute(actions);
        alpine.close();
    
}



//-----------------------------------------------------------------------------
TEST(alpine_flow_pipeline, test_flow_load_and_save_graph)
{
    // test add_graph, add_filters, and add_connections
    
        if(!flow::Workspace::supports_filter_type<InspectFilter>())
        {
            flow::Workspace::register_filter_type<InspectFilter>();
        }
    
        Node actions;
        string graph_ofile = conduit::utils::join_file_path(output_dir(),
                                                            "tout_flow_pipeline_load_and_save.json");
    
        actions.append();
        actions[0]["action"] = "add_graph";
        Node &graph = actions[0]["graph"];

        graph["filters/fi/type_name"] = "inspect";

        graph["connections"].append();
        graph["connections"][0]["src"] = ":source";
        graph["connections"][0]["dest"] = "fi";

        actions.append()["action"] = "execute";
        
        actions.append();
        actions[2]["action"] = "save_graph";
        actions[2]["path"]   = graph_ofile;
        
        actions.print();

        // we want the "flow" pipeline
        Node open_opts;
        open_opts["pipeline/type"] = "flow";

        //
        // Run Alpine
        //
        Alpine alpine;
        alpine.open(open_opts);

        //
        // Create example mesh.
        //
        Node data;
        conduit::blueprint::mesh::examples::braid("quads",100,100,0,data);
        alpine.publish(data);
        alpine.execute(actions);
        
        actions.reset();
        actions.append()["action"] = "reset";
        actions.append();
        actions[1]["action"] = "load_graph";
        actions[1]["path"] = graph_ofile;
        alpine.execute(actions);
    
        
        alpine.close();
    
}



