//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_extract_example4.cpp
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

    // Use Ascent to create a Cinema Image Database
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

    // create a  pipeline (pl1) with a contour filter (f1)
    pipelines["pl1/f1/type"] = "contour";

    // extract contours where braid variable
    // equals 0.2 and 0.4
    Node &contour_params = pipelines["pl1/f1/params"];
    contour_params["field"] = "braid";

    double iso_vals[2] = {0.2, 0.4};
    contour_params["iso_values"].set(iso_vals,2);

    // declare a scene to render several angles of
    // the pipeline result (pl1) to a Cinema Image
    // database

    Node &add_act2 = actions.append();
    add_act2["action"] = "add_scenes";
    Node &scenes = add_act2["scenes"];

    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/plots/p1/field"] = "braid";
    // select cinema path
    scenes["s1/renders/r1/type"] = "cinema";
    // use 5 renders in phi
    scenes["s1/renders/r1/phi"] = 5;
    // and 5 renders in theta
    scenes["s1/renders/r1/theta"] = 5;
    // setup to output database to:
    //  cinema_databases/out_extract_cinema_contour
    // you can view using a webserver to open:
    //  cinema_databases/out_extract_cinema_contour/index.html
    scenes["s1/renders/r1/db_name"] = "out_extract_cinema_contour";

    // print our full actions tree
    std::cout << actions.to_yaml() << std::endl;

    // execute the actions
    a.execute(actions);

    // close ascent
    a.close();
}



