//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_pipeline_example3.cpp
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

    // create our first pipeline (pl1) 
    // with a contour filter (f1)
    pipelines["pl1/f1/type"] = "contour";
    // extract contours where braid variable
    // equals 0.2 and 0.4
    Node &contour_params = pipelines["pl1/f1/params"];
    contour_params["field"] = "braid";
    double iso_vals[2] = {0.2, 0.4};
    contour_params["iso_values"].set(iso_vals,2);

    // create our second pipeline (pl2) with a threshold filter (f1)
    // and a clip filter (f2)

    // add our threshold (pl2 f1)
    pipelines["pl2/f1/type"] = "threshold";
    Node &thresh_params = pipelines["pl2/f1/params"];
    // set threshold parameters
    // keep elements with values between 0.0 and 0.5
    thresh_params["field"]  = "braid";
    thresh_params["min_value"] = 0.0;
    thresh_params["max_value"] = 0.5;

    // add our clip (pl2 f2)
    pipelines["pl2/f2/type"]   = "clip";
    Node &clip_params = pipelines["pl2/f2/params"];
    // set clip parameters
    // use spherical clip
    clip_params["sphere/center/x"] = 0.0;
    clip_params["sphere/center/y"] = 0.0;
    clip_params["sphere/center/z"] = 0.0;
    clip_params["sphere/radius"]   = 12;

    // declare a scene to render our pipeline results
    Node &add_act2 = actions.append();
    add_act2["action"] = "add_scenes";
    Node &scenes = add_act2["scenes"];

    // add a scene (s1) with two pseudocolor plots 
    // (p1 and p2) that will render the results 
    // of our pipelines (pl1 and pl2)## Pipeline Example 2:

    // plot (p1) to render our first pipeline (pl1)
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/plots/p1/field"] = "braid";
    // plot (p2) to render our second pipeline (pl2)
    scenes["s1/plots/p2/type"] = "pseudocolor";
    scenes["s1/plots/p2/pipeline"] = "pl2";
    scenes["s1/plots/p2/field"] = "braid";
    // set the output file name (ascent will add ".png")
    scenes["s1/image_name"] = "out_pipeline_ex3_two_plots";

    // print our full actions tree
    std::cout << actions.to_yaml() << std::endl;

    // execute the actions
    a.execute(actions);

    a.close();
}



