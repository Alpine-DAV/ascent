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

    Node pipelines;
    pipelines["pl1/f1/type"] = "contour";
    Node contour_params;
    contour_params["field"] = "braid";
    double iso_vals[2] = {0.2, 0.4};
    contour_params["iso_values"].set_external(iso_vals,2);
    pipelines["pl1/f1/params"] = contour_params;

    // setup actions
    Node actions;
    Node &add_act = actions.append();
    add_act["action"] = "add_pipelines";
    add_act["pipelines"] = pipelines;

    Node extracts;
    extracts["e1/type"] = "relay";
    extracts["e1/pipeline"]  = "pl1";
    extracts["e1/params/path"] = "braid_contour";
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

    // setup actions
    Node &add_act2 = actions.append();
    add_act2["action"] = "add_extracts";
    add_act2["extracts"] = extracts;

    actions.append()["action"] = "execute";

    // execute
    a.execute(actions);

    // close ascent
    a.close();
}



