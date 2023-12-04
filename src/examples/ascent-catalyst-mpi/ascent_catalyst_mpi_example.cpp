//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_catalyst_mpi_example.cpp
///
//-----------------------------------------------------------------------------

#include <catalyst.h>
#include <conduit_blueprint.hpp>
#include <conduit_cpp_to_c.hpp>

#include <mpi.h>
#include <iostream>

using namespace conduit;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    int par_rank, par_size;
    MPI_Comm_rank(mpi_comm, &par_rank);
    MPI_Comm_size(mpi_comm, &par_size);

    // Custom catalyst implementations require that we use conduit's C API
    conduit_node *initialize = conduit_node_create();
    conduit_node_set_path_char8_str(initialize, "catalyst_load/implementation", "ascent");
    conduit_node_set_path_char8_str(initialize, "catalyst/scripts/my_actions", "ascent_actions.yaml");
    catalyst_initialize(initialize);

    if (par_rank == 0)
    {
        conduit_node *about = conduit_node_create();
        catalyst_about(about);
        conduit_node_print(about);
    }

    // create an example dataset
    conduit::Node mesh;
    conduit::blueprint::mesh::examples::braid("uniform", 10, 10, 10, mesh);

    // to make a multi domain example, offset the x coords by par_rank
    mesh["coordsets/coords/origin/x"] = -10 + 20 * par_rank;

    // We can still use conduit's C++ API
    conduit::Node data;
    data["catalyst/channels/grid/data"].set_external(mesh);

    // We just have to convert cpp_nodes into c_nodes before passing them to catalyst
    conduit_node *data_converted = c_node(&data);
    catalyst_execute(data_converted);

    conduit_node *finalize = conduit_node_create();
    catalyst_finalize(finalize);

    MPI_Finalize();
}
