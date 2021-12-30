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


using namespace std;
using namespace conduit;
using namespace ascent;


const float64 PI_VALUE = 3.14159265359;

bool launch_server = false;
bool use_doc_root  = false;
std::string doc_root = "";

#include <flow.hpp>

// ----- //
// This tests that we can create a custom filter, register it and use it
// in the flow runtime.
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
            ASCENT_ERROR("Error, input is not a conduit node!");
        }

        Node *n = input<Node>(0);

        ASCENT_INFO("Total Strided Bytes = " << n->total_strided_bytes());

        set_output<Node>(n);
    }

};


//-----------------------------------------------------------------------------
TEST(ascent_web, test_ascent_web_launch)
{
    // this test launches a web server and infinitely streams images from
    // ascent we  only run it if we passed proper command line arg
    if(!launch_server)
    {
        return;
    }

    flow::Workspace::register_filter_type<InspectFilter>();


    //
    // Create example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("quads",100,100,0,data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    Node actions;
    actions.append();
    actions[0]["action"] = "add_filter";
    actions[0]["type_name"]  = "inspect";
    actions[0]["name"] = "fi";

    actions.append();
    actions[1]["action"] = "connect";
    actions[1]["src"]  = "source";
    actions[1]["dest"] = "fi";
    actions[1]["port"] = "in";

    actions.print();

    // we want the "flow" runtime
    Node open_opts;
    open_opts["runtime/type"] = "flow";
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
        ASCENT_INFO(actions.to_json());
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


