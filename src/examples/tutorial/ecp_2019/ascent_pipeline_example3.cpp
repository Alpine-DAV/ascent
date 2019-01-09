#include <iostream>
#include "ascent.hpp"
#include "conduit_blueprint.hpp"

using namespace ascent;
using namespace conduit;

int main(int argc, char **argv)
{
    Ascent a;

    // open ascent
    a.open();

    Node n_mesh;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              25,
                                              25,
                                              25,
                                              n_mesh);

    // publish mesh to ascent
    a.publish(n_mesh);

    conduit::Node pipelines;

    // pipeline 1
    pipelines["pl1/f1/type"] = "contour";
    conduit::Node contour_params;
    contour_params["field"] = "braid";
    double iso_vals[2] = {0.2, 0.4};
    contour_params["iso_values"].set_external(iso_vals,2);
    pipelines["pl1/f1/params"] = contour_params;

    // pipeline 2
    conduit::Node thresh_params;
    pipelines["pl2/f1/type"] = "threshold";
    thresh_params["field"]  = "braid";
    thresh_params["min_value"] = 0.0;
    thresh_params["max_value"] = 0.5;
    pipelines["pl2/f1/params"] = thresh_params;
    pipelines["pl2/f2/type"]   = "clip";
    conduit::Node clip_params;
    clip_params["topology"] = "mesh";
    clip_params["sphere/center/x"] = 0.0;
    clip_params["sphere/center/y"] = 0.0;
    clip_params["sphere/center/z"] = 0.0;
    clip_params["sphere/radius"]   = 12;
    pipelines["pl2/f2/params/"] = clip_params;

    // setup actions
    Node actions;
    Node &add_act = actions.append();
    add_act["action"] = "add_pipelines";
    add_act["pipelines"] = pipelines;

    // declare a scene to render the dataset
    Node scenes;
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/plots/p2/type"] = "pseudocolor";
    scenes["s1/plots/p2/pipeline"] = "pl2";
    scenes["s1/plots/p2/field"] = "braid";
    // set the output file name (ascent will add ".png")
    scenes["s1/image_prefix"] = "two_pipelines";

    Node &add_act2 = actions.append();
    add_act2["action"] = "add_scenes";
    add_act2["scenes"] = scenes;

    actions.append()["action"] = "execute";

    // execute
    a.execute(actions);

    // close ascent
    a.close();
}



