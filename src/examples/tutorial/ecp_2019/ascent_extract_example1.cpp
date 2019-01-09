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

    Node extracts;
    extracts["e1/type"] = "relay";
    extracts["e1/params/path"] = "braid";
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

    // setup actions
    Node actions;
    Node &add_act = actions.append();
    add_act["action"] = "add_extracts";
    add_act["extracts"] = extracts;

    actions.append()["action"] = "execute";

    // execute
    a.execute(actions);

    // close ascent
    a.close();
}



