//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_catalyst_example.cpp
///
//-----------------------------------------------------------------------------

#include <catalyst.h>
#include <conduit_blueprint.hpp>
#include <conduit_cpp_to_c.hpp>

#include <iostream>

using namespace conduit;

int main(int argc, char **argv)
{
    // Custom catalyst implementations require that we use conduit's C API
    conduit_node *initialize = conduit_node_create();
    conduit_node_set_path_char8_str(initialize, "catalyst_load/implementation", "ascent");
    conduit_node_set_path_char8_str(initialize, "catalyst/scripts/my_actions", "ascent_actions.yaml");
    catalyst_initialize(initialize);

    conduit_node *about = conduit_node_create();
    catalyst_about(about);
    conduit_node_print(about);

    // create an example dataset
    conduit::Node mesh;
    conduit::blueprint::mesh::examples::braid("uniform", 10, 10, 10, mesh);

    // We can still use conduit's C++ API
    conduit::Node data;
    data["catalyst/channels/grid/data"].set_external(mesh);

    // We just have to convert cpp_nodes into c_nodes before passing them to catalyst
    conduit_node *data_converted = c_node(&data);
    catalyst_execute(data_converted);

    conduit_node *finalize = conduit_node_create();
    catalyst_finalize(finalize);
}
