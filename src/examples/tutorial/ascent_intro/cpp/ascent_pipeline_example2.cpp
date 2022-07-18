//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_pipeline_example2.cpp
///
//-----------------------------------------------------------------------------
#include <iostream>
#include "ascent.hpp"
#include "conduit_blueprint.hpp"

using namespace ascent;
using namespace conduit;

int main(int argc, char **argv)
{
    //create example mesh using the conduit blueprint braid helper
    
    Node mesh;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              25,
                                              25,
                                              25,
                                              mesh);
    Ascent a;

    // open ascent
    a.open();

    // publish mesh to ascent
    a.publish(mesh);

    // setup actions
    Node actions;
    Node &add_act = actions.append();
    add_act["action"] = "add_pipelines";
    Node &pipelines = add_act["pipelines"];

    // create a  pipeline (pl1) with a threshold filter (f1)
    // and a clip filter (f2)

    // add our threshold (f1)
    pipelines["pl1/f1/type"] = "threshold";
    Node &thresh_params = pipelines["pl1/f1/params"];
    // set threshold parameters
    // keep elements with values between 0.0 and 0.5
    thresh_params["field"]  = "braid";
    thresh_params["min_value"] = 0.0;
    thresh_params["max_value"] = 0.5;

    // add our clip (f2)
    pipelines["pl1/f2/type"]   = "clip";
    Node &clip_params = pipelines["pl1/f2/params"];
    // set clip parameters
    // use spherical clip
    clip_params["sphere/center/x"] = 0.0;
    clip_params["sphere/center/y"] = 0.0;
    clip_params["sphere/center/z"] = 0.0;
    clip_params["sphere/radius"]   = 12;

    //  declare a scene to render the pipeline results
    Node &add_act2 = actions.append();
    add_act2["action"] = "add_scenes";
    Node &scenes = add_act2["scenes"];

    // add a scene (s1) with one pseudocolor plot (p1) that 
    // will render the result of our pipeline (pl1)
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/plots/p1/field"] = "braid";
    // set the output file name (ascent will add ".png")
    scenes["s1/image_name"] = "out_pipeline_ex2_thresh_clip";

    // print our full actions tree
    std::cout << actions.to_yaml() << std::endl;

    // execute the actions
    a.execute(actions);

    a.close();
}

