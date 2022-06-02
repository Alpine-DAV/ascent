//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_extract_example3.cpp
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

    // Use Ascent to export contours to blueprint flavored hdf5 files
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

    // add an extract to capture the pipeline result
    Node &add_act2 = actions.append();
    add_act2["action"] = "add_extracts";
    Node &extracts = add_act2["extracts"];

    // add an relay extract (e1) to export the pipeline result
    // (pl1) to blueprint hdf5 files
    extracts["e1/type"] = "relay";
    extracts["e1/pipeline"]  = "pl1";
    extracts["e1/params/path"] = "out_extract_braid_contour";
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

    // print our full actions tree
    std::cout << actions.to_yaml() << std::endl;

    // execute the actions
    a.execute(actions);

    // close ascent
    a.close();
}



