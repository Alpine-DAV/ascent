//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_render_example.cpp
///
//-----------------------------------------------------------------------------

#include <ascent.hpp>
#include <catalyst.h>
#include <conduit_blueprint.hpp>

#include <iostream>

using namespace ascent;
using namespace conduit;

int main(int argc, char **argv)
{
    conduit_node *initialize = conduit_node_create();
    conduit_node_set_path_char8_str(initialize, "catalyst_load/implementation", "ascent");
    conduit_node_print(initialize);
    catalyst_initialize(initialize);

    conduit_node *about = conduit_node_create();
    catalyst_about(about);
    conduit_node_print(about);

    // Ascent a;

    // // open ascent
    // a.open();

    // // create example mesh using conduit blueprint
    // Node n_mesh;
    // conduit::blueprint::mesh::examples::braid("hexs",
    //                                           10,
    //                                           10,
    //                                           10,
    //                                           n_mesh);
    // // publish mesh to ascent
    // a.publish(n_mesh);

    // // declare a scene to render the dataset
    // Node scenes;
    // scenes["s1/plots/p1/type"] = "pseudocolor";
    // scenes["s1/plots/p1/field"] = "braid";
    // // Set the output file name (ascent will add ".png")
    // scenes["s1/image_prefix"] = "out_ascent_render_3d";

    // // setup actions
    // Node actions;
    // Node &add_act = actions.append();
    // add_act["action"] = "add_scenes";
    // add_act["scenes"] = scenes;

    // // execute
    // a.execute(actions);

    // // close ascent
    // a.close();
}
