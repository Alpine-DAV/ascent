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
/// file: t_alpine_web.cpp
///
//-----------------------------------------------------------------------------


#include "gtest/gtest.h"

#include <alpine.hpp>

#include <iostream>
#include <math.h>

#include <conduit_blueprint.hpp>

#include "t_config.hpp"
#include "t_alpine_test_utils.hpp"


using namespace std;
using namespace conduit;
using namespace alpine;


const float64 PI_VALUE = 3.14159265359;

bool launch_server = false;

//-----------------------------------------------------------------------------
TEST(alpine_web, test_alpine_web_launch)
{
    // this test launches a web server and infinitely streams images from 
    // alpine we  only run it if we passed proper command line arg
    if(!launch_server)
    {
        return;
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",100,100,100,data);
    
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));
    verify_info.print();


    //
    // Create the actions.
    //

    Node actions;
    
    Node &plot = actions.append();
    plot["action"]     = "add_plot";
    plot["field_name"] = "braid";
    
    Node &opts = plot["render_options"];
    opts["width"]  = 500;
    opts["height"] = 500;
    actions.append()["action"] = "draw_plots";
    
    //
    // Launch alpine with webserver
    //

    
    Node open_opts;
    open_opts["web/stream"] = "true";

    Alpine alpine;
    alpine.open(open_opts);
    
    uint64  *cycle_ptr = data["state/cycle"].value();
    float64 *time_ptr  = data["state/time"].value();
    while(true)
    {
        // publish the same mesh data, but update the state info
        alpine.publish(data);
        alpine.execute(actions);
        conduit::utils::sleep(100);
        cycle_ptr[0]+=1;
        time_ptr[0] = PI_VALUE * cycle_ptr[0];
    }
    
    alpine.close();
}


//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);

    for(int i=0; i < argc ; i++)
    {
        std::string arg_str(argv[i]);
        if(arg_str == "launch")
        {
            launch_server = true;;
        }
    }

    result = RUN_ALL_TESTS();
    return result;
}


