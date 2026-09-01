//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_query_example1.cpp
///
//-----------------------------------------------------------------------------

#include <iostream>
#include <sstream>

#include "ascent.hpp"
#include "conduit_blueprint.hpp"

#include "ascent_tutorial_cpp_utils.hpp"

using namespace ascent;
using namespace conduit;

const int EXAMPLE_MESH_SIDE_DIM = 20;

int main(int argc, char **argv)
{
    Node mesh;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              mesh);

    // Use Ascent to bin an input mesh in a few ways
    Ascent a;

    // open ascent
    a.open();

    // publish mesh to ascent
    a.publish(mesh);

    // setup actions
    Node actions;

    // Add a sampling pipeline
    Node &add_sample_act = actions.append();
    add_sample_act["action"] = "add_pipelines";

    Node &sample_pipe = add_sample_act["pipelines"];
    sample_pipe["pl1/f1/type"] = "sample";
    sample_pipe["pl1/f1/params/fields"] = {"braid"};

    // Define the bounding box
    sample_pipe["pl1/f1/params/box/dims/i"] = 25.0;
    sample_pipe["pl1/f1/params/box/dims/j"] = 25.0;
    sample_pipe["pl1/f1/params/box/dims/k"] = 25.0;

    sample_pipe["pl1/f1/params/box/min/x"] = 0.0;
    sample_pipe["pl1/f1/params/box/min/y"] = 0.0;
    sample_pipe["pl1/f1/params/box/min/z"] = 0.0;

    sample_pipe["pl1/f1/params/box/max/x"] = "max";
    sample_pipe["pl1/f1/params/box/max/y"] = "max";
    sample_pipe["pl1/f1/params/box/max/z"] = "max";

    sample_pipe["pl1/f1/params/invalid_value"] = -10.0;

    // Add a scene that renders the sampled result.
    Node &add_act = actions.append();
    add_act["action"] = "add_scenes";

    // declare a queries to ask some questions
    Node &scenes = add_act["scenes"];
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/image_name"] = "sample_bounding_box";

    // print our full actions tree
    std::cout << actions.to_yaml() << std::endl;

    // execute the actions
    a.execute(actions);

    // retrieve the info node that contains the query results
    Node info;
    a.info(info);

    // close ascent
    a.close();

    //
    // We can also examine when the results by looking at the expressions
    // results in the output info
    //
    std::cout << info["expressions"].to_yaml() << std::endl;
}