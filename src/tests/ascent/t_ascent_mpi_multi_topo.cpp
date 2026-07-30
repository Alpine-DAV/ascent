//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_mpi_render_2d.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <ascent.hpp>

#include <iostream>
#include <math.h>
#include <mpi.h>

#include <conduit_blueprint.hpp>
#include <conduit_relay.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"

using namespace std;
using namespace conduit;
using ascent::Ascent;

//-----------------------------------------------------------------------------
// note: this example was a reproducer for tricky case
// involving multiple topos, pipelines + rendering.
TEST(ascent_mpi_mult_topo, test_multi_semi_madness)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping MPI multi topo "
                      "runtime test");
        return;
    }


    //
    // Set Up MPI
    //
    int par_rank;
    int par_size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &par_rank);
    MPI_Comm_size(comm, &par_size);

    ASCENT_INFO("Rank "
                  << par_rank
                  << " of "
                  << par_size
                  << " reporting");
    //
    // Create the data.
    //
    Node data, verify_info;
    create_example_multi_domain_multi_topo_dataset(data,par_rank,par_size);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    // make sure the _output dir exists
    string output_path = "";
    if(par_rank == 0)
    {
        output_path = prepare_output_dir();
    }
    else
    {
        output_path = output_dir();
    }

    string output_file = conduit::utils::join_file_path(output_path,
                            "tout_render_mpi_multi_domain_multi_topo");

    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //

    conduit::Node actions;

    conduit::Node &add_plines = actions.append();
    add_plines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_plines["pipelines"];
    pipelines["pl1/f1/type"] = "threshold";
    pipelines["pl1/f1/params/field"] = "ele_example";
    pipelines["pl1/f1/params/min_value"] = 2.0;
    pipelines["pl1/f1/params/max_value"] = 11112.0;
 
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    conduit::Node &scenes = add_plots["scenes"];
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "ele_example";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/plots/p2/type"]  = "pseudocolor";
    scenes["s1/plots/p2/field"] = "braid";
    scenes["s1/image_prefix"] = output_file;


    std::cout << actions.to_yaml() << std::endl;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    // we use the mpi handle provided by the fortran interface
    // since it is simply an integer
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);
    ascent_opts["runtime"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    MPI_Barrier(comm);
    // check that we created an image
    // EXPECT_TRUE(check_test_image(output_file));
}

//-----------------------------------------------------------------------------
TEST(ascent_mpi_mult_topo, sample_rank_zero_topology)
{
    Node n;
    ascent::about(n);
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping MPI topology sample test");
        return;
    }

    int par_rank;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &par_rank);

    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs", 10, 10, 10, data);
    data["state/domain_id"] = par_rank;
    data["state/cycle"] = 100;

    if(par_rank == 0)
    {
        Node plane;
        conduit::blueprint::mesh::examples::braid("quads", 5, 5, 0, plane);
        data["topologies/sample_plane"] = plane["topologies/mesh"];
        data["topologies/sample_plane/coordset"] = "sample_plane_coords";
        data["coordsets/sample_plane_coords"] = plane["coordsets/coords"];
    }

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data, verify_info));

    string output_path = "";
    if(par_rank == 0)
    {
        output_path = prepare_output_dir();
    }
    else
    {
        output_path = output_dir();
    }

    string output_file = conduit::utils::join_file_path(output_path,
                                                        "tout_mpi_sample_rank_zero_topology");
    string output_root = output_file + ".cycle_000100.root";
    if(par_rank == 0 && conduit::utils::is_file(output_root))
    {
        conduit::utils::remove_file(output_root);
    }

    conduit::Node actions;
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_pipelines["pipelines"];
    pipelines["pl1/f1/type"] = "sample";
    pipelines["pl1/f1/params/fields"].append() = "braid";
    pipelines["pl1/f1/params/topology"] = "sample_plane";
    pipelines["pl1/f1/params/invalid_value"] = -10.0;

    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts = add_extracts["extracts"];
    extracts["e1/pipeline"] = "pl1";
    extracts["e1/type"] = "relay";
    extracts["e1/params/path"] = output_file;
    extracts["e1/params/protocol"] = "blueprint/mesh/yaml";

    Ascent ascent;
    Node ascent_opts;
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);
    ascent_opts["runtime"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    MPI_Barrier(comm);

    if(par_rank == 0)
    {
        EXPECT_TRUE(conduit::utils::is_file(output_root));

        Node read_mesh, read_verify_info;
        conduit::relay::io::blueprint::load_mesh(output_root, read_mesh);
        EXPECT_TRUE(conduit::blueprint::mesh::verify(read_mesh, read_verify_info));
        EXPECT_EQ(conduit::blueprint::mesh::number_of_domains(read_mesh), 1);

        const Node &dom = read_mesh.child(0);
        EXPECT_TRUE(dom.has_path("topologies/sample_plane"));
        EXPECT_TRUE(dom.has_path("fields/braid"));
        EXPECT_EQ(dom["fields/braid/topology"].as_string(), "sample_plane");
        EXPECT_GT(dom["fields/braid/values"].dtype().number_of_elements(), 0);
    }
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

