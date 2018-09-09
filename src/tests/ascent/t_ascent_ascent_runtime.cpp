//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
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
/// file: t_ascent_flow_pipeline.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <ascent.hpp>

#include <ascent_main_runtime.hpp>

#include <iostream>
#include <math.h>
#include <sstream>

#include <conduit_blueprint.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"


index_t EXAMPLE_MESH_SIDE_DIM = 50;

using namespace std;
using namespace conduit;
using namespace ascent;
//-----------------------------------------------------------------------------
TEST(ascent_pipeline, test_render_3d_main_pipeline)
{
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
    string output_file = conduit::utils::join_file_path(output_path, 
                                                        "tout_render_3d_ascent_pipeline");
    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/params/field"] = "braid";
    scenes["s1/image_prefix"] = output_file;
 
 
    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;
    conduit::Node &execute  = actions.append();
    execute["action"] = "execute";
    actions.print();
    
    //
    // Run Ascent
    //
    
    Ascent ascent;

    Node ascent_opts;
    Node ascent_info;
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.info(ascent_info);
    EXPECT_EQ(ascent_info["runtime/type"].as_string(),"ascent");
    ascent_info.print();
    ascent.close();
    //
    // // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
}


//-----------------------------------------------------------------------------
TEST(ascent_pipeline, test_error_for_mpi_vs_non_mpi)
{
    Ascent ascent;
    Node ascent_opts;
    ascent_opts["mpi_comm"] = 1;
    ascent_opts["exceptions"] = "forward";
    // we throw an error if an mpi_comm is provided to a non-mpi ver of ascent
    EXPECT_THROW(ascent.open(ascent_opts),conduit::Error);
}

//-----------------------------------------------------------------------------
// Check that we can register and call extracts and transforms from outslide
// of ascent, and use them.
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
class MyExtract: public ::flow::Filter
{
    public:
        static bool s_was_called;

        static void reset_was_called(){ s_was_called = false;}
        static bool was_called(){ return s_was_called;}
        
        MyExtract():Filter()
        {}
        ~MyExtract()
        {}
        
        void declare_interface(Node &i)
        {
            i["type_name"]   = "my_noop_extract";
            i["port_names"].append() = "in";
            i["output_port"] = "true";
        }
        
        void execute()
        {
            s_was_called = true;
            set_output(input(0));
        }
};

bool MyExtract::s_was_called = false;

//-----------------------------------------------------------------------------
TEST(ascent_pipeline, test_register_extract)
{
    AscentRuntime::register_filter_type<MyExtract>("extracts","my_extract");
    

    conduit::Node extracts;
    extracts["e1/type"]  = "my_extract";
    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    conduit::Node &execute  = actions.append();
    execute["action"] = "execute";
    
    Node data, info;
    conduit::blueprint::mesh::examples::braid("quads",
                                               5,
                                               5,
                                               0,
                                               data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,info));

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    MyExtract::reset_was_called();
    EXPECT_FALSE(MyExtract::was_called());
    ascent.execute(actions);
    EXPECT_TRUE(MyExtract::was_called());
    ascent.info(info);
    info.print();
    ascent.close();
    
    
}

//-----------------------------------------------------------------------------
class MyXForm: public ::flow::Filter
{
    public:
        static bool s_was_called;

        static void reset_was_called(){ s_was_called = false;}
        static bool was_called(){ return s_was_called;}
        
        MyXForm():Filter()
        {}
        ~MyXForm()
        {}
        
        void declare_interface(Node &i)
        {
            i["type_name"]   = "my_noop_xform";
            i["port_names"].append() = "in";
            i["output_port"] = "true";
        }
        
        void execute()
        {
            s_was_called = true;
            set_output(input(0));
        }
};

bool MyXForm::s_was_called = false;


//-----------------------------------------------------------------------------
TEST(ascent_pipeline, test_register_transform)
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

    AscentRuntime::register_filter_type<MyXForm>("transforms","my_xform");
    
    Node data, info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                               5,
                                               5,
                                               5,
                                               data);
    
    
    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_reg_xform");
    
    // remove old images before rendering
    remove_test_image(output_file);
    
    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "my_xform";

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/params/field"] = "radial";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/image_prefix"] = output_file;
 
    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines= actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;
    // add the scenes
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    add_scenes["scenes"] = scenes;
    // execute
    conduit::Node &execute  = actions.append();
    execute["action"] = "execute";
    
    
    Ascent ascent;
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,info));
    ascent.open();
    ascent.publish(data);
    MyXForm::reset_was_called();
    EXPECT_FALSE(MyXForm::was_called());
    ascent.execute(actions);
    EXPECT_TRUE(MyXForm::was_called());
    ascent.info(info);
    info.print();
    ascent.close();
    
    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
}

