//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_web.cpp
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


const float64 PI_VALUE = 3.14159265359;

bool launch_server = false;
bool use_doc_root  = false;
std::string doc_root = "";

#include <flow.hpp>

//-----------------------------------------------------------------------------
TEST(ascent_web, test_ascent_main_web_launch)
{
    // this test launches a web server and infinitely streams images from
    // ascent we  only run it if we passed proper command line arg
    if(!launch_server)
    {
        return;
    }


    //
    // Create example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",100,100,100,data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_render_3d_web_main_runtime");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/image_prefix"] = output_file;


    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;
    actions.print();

    // we want the "flow" runtime
    Node open_opts;
    open_opts["runtime/type"] = "ascent";
    open_opts["web/stream"] = "true";
    if(use_doc_root)
    {
        open_opts["web/document_root"] = doc_root;
    }
    open_opts["ascent_info"] = "verbose";

    Ascent ascent;
    ascent.open(open_opts);

    uint64  *cycle_ptr = data["state/cycle"].value();
    float64 *time_ptr  = data["state/time"].value();

    ascent.publish(data);
    ascent.execute(actions);

    while(true)
    {
        cycle_ptr[0]+=1;
        time_ptr[0] = PI_VALUE * cycle_ptr[0];
        ASCENT_INFO(data["state"].to_json());
        // publish the same mesh data, but update the state info
        actions.reset();
        ascent.publish(data);
        ascent.execute(actions);
        conduit::utils::sleep(1000);
    }

    ascent.close();
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
        else if(arg_str == "doc_root" && (i+1 < argc) )
        {
            use_doc_root = true;
            doc_root = std::string(argv[i+1]);
            i++;
        }
    }

    result = RUN_ALL_TESTS();
    return result;
}


