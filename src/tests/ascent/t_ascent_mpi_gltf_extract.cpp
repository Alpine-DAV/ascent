//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_mpi_gltf_extract.cpp
///
//-----------------------------------------------------------------------------


#include "gtest/gtest.h"

#include <ascent.hpp>
#include <mpi.h>

#include <fstream>
#include <iostream>
#include <math.h>

#include <conduit_blueprint.hpp>
#include <conduit_relay.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"


using namespace std;
using namespace conduit;
using namespace ascent;


//-----------------------------------------------------------------------------
bool
check_glb_magic(const std::string &path)
{
    char magic[4] = {0, 0, 0, 0};
    std::ifstream ifs(path.c_str(), std::ios::binary);
    ifs.read(magic, 4);
    return ifs.good() && std::string(magic, 4) == "glTF";
}

//-----------------------------------------------------------------------------
TEST(ascent_mpi_gltf_extract, uneven_multi_domain_field)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent viskores support disabled, skipping test");
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
    // Create an example mesh with an uneven domain split:
    // rank 0 has domains 0 and 1, rank 1 has domain 2.
    //
    Node data, verify_info;
    if(par_rank == 0)
    {
        create_3d_example_dataset(data["domain_000000"], 10, 0, 3);
        create_3d_example_dataset(data["domain_000001"], 10, 1, 3);
    }
    else // par_rank == 1
    {
        create_3d_example_dataset(data["domain_000002"], 10, 2, 3);
    }

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing mpi gltf extract with uneven domains and a field");

    conduit::Node actions;
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_pipelines["pipelines"];
    pipelines["pl1/f1/type"] = "contour";
    pipelines["pl1/f1/params/field"] = "radial_vert";
    pipelines["pl1/f1/params/iso_values"] = 100.0;

    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts = add_extracts["extracts"];

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,
                                             "tout_mpi_gltf_extract_uneven_domains_{cycle:06d}");
    string output_file_formatted = conduit::utils::join_file_path(output_path,
                                                "tout_mpi_gltf_extract_uneven_domains_000100");

    if(par_rank == 0)
    {
        // remove output files from prior runs so stale files can't
        // satisfy checks
        remove_test_file(conduit::utils::join_file_path(output_file_formatted,
                                                        "manifest.json"));
        string domains_dir = conduit::utils::join_file_path(output_file_formatted,
                                                            "domains");
        remove_test_file(conduit::utils::join_file_path(domains_dir,
                                                        "domain_00000000.glb"));
        remove_test_file(conduit::utils::join_file_path(domains_dir,
                                                        "domain_00000001.glb"));
        remove_test_file(conduit::utils::join_file_path(domains_dir,
                                                        "domain_00000002.glb"));
    }

    MPI_Barrier(comm);

    // add the extract
    extracts["e1/type"] = "gltf";
    extracts["e1/pipeline"] = "pl1";
    extracts["e1/params/path"] = output_file;
    extracts["e1/params/field"] = "radial_vert";

    std::cout << actions.to_yaml() << std::endl;

    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    MPI_Barrier(comm);

    if(par_rank == 0)
    {
        // check the manifest
        string manifest_file = conduit::utils::join_file_path(output_file_formatted,
                                                              "manifest.json");
        EXPECT_TRUE(conduit::utils::is_file(manifest_file));

        Node manifest;
        conduit::relay::io::load(manifest_file, "json", manifest);
        EXPECT_EQ(manifest["protocol"].as_string(), "ascent-gltf");
        EXPECT_EQ(manifest["field"].as_string(), "radial_vert");
        EXPECT_EQ(manifest["domains"].number_of_children(), 3);

        // check all glb files
        string domains_dir = conduit::utils::join_file_path(output_file_formatted,
                                                            "domains");
        for(int domain_id = 0; domain_id < 3; domain_id++)
        {
            Node &record = manifest["domains"][domain_id];
            EXPECT_EQ(record["domain_id"].to_int64(), domain_id);
            string glb_file = conduit::utils::join_file_path(domains_dir,
                                 record["uri"].as_string().substr(string("domains/").size()));
            EXPECT_TRUE(conduit::utils::is_file(glb_file));
            EXPECT_TRUE(check_glb_magic(glb_file));
        }
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
