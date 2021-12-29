// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.

//-----------------------------------------------------------------------------
///
/// file: ascent_mpi_render_example.cpp
///
//-----------------------------------------------------------------------------


#include <ascent.hpp>
#include <conduit_blueprint.hpp>

#include <mpi.h>
#include <iostream>

using namespace ascent;
using namespace conduit;


int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    int par_rank, par_size;
    MPI_Comm_rank(mpi_comm, &par_rank);
    MPI_Comm_size(mpi_comm, &par_size);

    //much ado about `about`
    if(par_rank == 0)
    {
        conduit::Node about;
        ascent::about(about);
        std::cout << about.to_yaml() << std::endl;
    }

    // create an example dataset
    conduit::Node mesh;
    conduit::blueprint::mesh::examples::braid("uniform",10,10,10,mesh);

    // to make a multi domain example, offset the x coords by par_rank
    mesh["coordsets/coords/origin/x"] = -10 + 20 * par_rank;

    // setup actions
    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    // declare a scene to render the dataset
    add_plots["action"] = "add_scenes";
    conduit::Node  & scenes = add_plots["scenes"];
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/image_name"]   = "out_ascent_mpi_example";

    if(par_rank == 0)
    {
        actions.print();
    }

    // Run Ascent
    Ascent ascent;

    // - setup options -
    // we use the mpi handle provided by the fortran interface
    // since it is simply an integer
    Node ascent_opts;
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(mpi_comm);
    ascent.open(ascent_opts);

    // publish mesh to ascent
    ascent.publish(mesh);

    // execute
    ascent.execute(actions);

    // shutdown our instance
    ascent.close();

    MPI_Finalize();
}


