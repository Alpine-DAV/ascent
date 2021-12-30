//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_scene_example4.cpp
///
//-----------------------------------------------------------------------------

#include <iostream>
#include "ascent.hpp"
#include "conduit_blueprint.hpp"

#include "ascent_tutorial_cpp_utils.hpp"

using namespace ascent;
using namespace conduit;

int main(int argc, char **argv)
{
    Node mesh;
    // (call helper to create example tet mesh as in blueprint example 2)
    tutorial_tets_example(mesh);

    // Use Ascent to render pseudocolor plots with different color tables

    Ascent a;

    // open ascent
    a.open();

    // publish mesh to ascent
    a.publish(mesh);

    // setup actions
    Node actions;
    Node &add_act = actions.append();
    add_act["action"] = "add_scenes";

    //
    // declare a two scenes (s1 and s2) to render the dataset
    // using different color tables
    // 
    // See the Color Tables docs for more details on what is supported:
    // https://ascent.readthedocs.io/en/latest/Actions/Scenes.html#color-tables
    //
    Node &scenes = add_act["scenes"];

    // the first scene (s1) will render a pseudocolor 
    // plot using Viridis color table
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "var1";
    scenes["s1/plots/p1/color_table/name"] = "Viridis";
    scenes["s1/image_name"] = "out_scene_ex4_render_viridis";

    // the first scene (s2) will render a pseudocolor 
    // plot using Inferno color table
    scenes["s2/plots/p1/type"] = "pseudocolor";
    scenes["s2/plots/p1/field"] = "var1";
    scenes["s2/plots/p1/color_table/name"] = "Inferno";
    scenes["s2/image_name"] = "out_scene_ex4_render_inferno";

    // print our full actions tree
    std::cout << actions.to_yaml() << std::endl;

    // execute the actions
    a.execute(actions);

    a.close();
}



