//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_extract_example1.cpp
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

    // Use Ascent to export our mesh to blueprint flavored hdf5 files
    Ascent a;

    // open ascent
    a.open();

    // publish mesh to ascent
    a.publish(mesh);

    // setup actions
    Node actions;
    Node &add_act = actions.append();
    add_act["action"] = "add_extracts";

    // add a relay extract that will write mesh data to 
    // blueprint hdf5 files
    Node &extracts = add_act["extracts"];
    extracts["e1/type"] = "relay";
    extracts["e1/params/path"] = "out_export_braid_one_field";
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

    //
    // add fields parameter to limit the export to the just 
    // the `braid` field
    //
    extracts["e1/params/fields"].append().set("braid");

    // print our full actions tree
    std::cout << actions.to_yaml() << std::endl;

    // execute the actions
    a.execute(actions);

    // close ascent
    a.close();
}



