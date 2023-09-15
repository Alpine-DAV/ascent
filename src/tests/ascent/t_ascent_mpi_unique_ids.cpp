//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_partition.cpp
///
//-----------------------------------------------------------------------------


#include "gtest/gtest.h"

#include <ascent.hpp>
#include <ascent_logging.hpp>

#include <iostream>
#include <math.h>
#include <mpi.h>

#include <conduit_blueprint.hpp>
#include <conduit_blueprint_mpi_mesh_examples.hpp>
#include <conduit_relay.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"


using namespace std;
using namespace conduit;
using namespace ascent;

//-----------------------------------------------------------------------------
TEST(ascent_partition, test_indiv_rank_non_unique)
{
    Node n;
    ascent::about(n);

    //
    // Create an example mesh.
    //
    Node data, verify_info;

    //
    //Set Up MPI
    //
    int par_rank;
    int par_size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &par_rank);
    MPI_Comm_size(comm, &par_size);

    // use spiral , with 20 domains
    conduit::blueprint::mpi::mesh::examples::spiral_round_robin(20,data,comm);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    int root = 0;
    if(par_rank == root)
    	ASCENT_INFO("Testing local non unique IDs");

    int num_domains = data.number_of_children();
    for(int i = 0; i < num_domains; i++)
    {
      conduit::Node &dom = data.child(i);
      dom["state/domain_id"] = 0;
    }
    
    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime"] = "ascent";
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);
    ascent_opts["exceptions"] = "forward";
    ascent.open(ascent_opts);

    EXPECT_THROW(ascent.publish(data),conduit::Error);
}
//-----------------------------------------------------------------------------
TEST(ascent_partition, test_global_ranks_non_unique)
{
    Node n;
    ascent::about(n);

    //
    // Create an example mesh.
    //
    Node data, verify_info;

    //
    //Set Up MPI
    //
    int par_rank;
    int par_size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &par_rank);
    MPI_Comm_size(comm, &par_size);

    // use spiral , with 20 domains
    conduit::blueprint::mpi::mesh::examples::spiral_round_robin(20,data,comm);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    int root = 0;
    if(par_rank == root)
    	ASCENT_INFO("Testing global non unique IDs");

    int num_domains = data.number_of_children();
    for(int i = 0; i < num_domains; i++)
    {
      conduit::Node &dom = data.child(i);
      dom["state/domain_id"] = i;
    }
    
    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime"] = "ascent";
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);
    ascent_opts["exceptions"] = "forward";
    ascent.open(ascent_opts);

    EXPECT_THROW(ascent.publish(data),conduit::Error);
}

//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}


