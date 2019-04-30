//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-716457
//
// All rights reserved.
//
// This file is part of Ascent.
//
// For details, see: http://ascent.readthedocs.io/.
//
// Please also read ascent/LICENSE
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_python_script_extract.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <ascent.hpp>

#include <iostream>
#include <mpi.h>
#include <conduit_blueprint.hpp>


#include "t_config.hpp"
#include "t_utils.hpp"

using namespace std;
using namespace conduit;
using ascent::Ascent;

std::string py_script = "\n"
"# Treat everything is mutli domian so grabe the state of the first\n"
"v = ascent_data().child(0)\n"
"print(v['state'])\n"
"\n"
"from mpi4py import MPI\n"
"print(ascent_mpi_comm_id())\n"
"comm = MPI.Comm.f2py(ascent_mpi_comm_id())\n"
"print('COMM SIZE = {}'.format(comm.Get_size()))\n"
"\n";

//-----------------------------------------------------------------------------
TEST(ascent_mpi_runtime, test_pyhton_script_extract_src)
{
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
    create_3d_example_dataset(data,32,par_rank,par_size);
    data["state/cycle"] = 101;

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    //
    // Create the actions.
    //

    conduit::Node extracts;
    extracts["e1/type"]  = "python";
    extracts["e1/params/source"] = py_script;

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    conduit::Node &execute  = actions.append();
    execute["action"] = "execute";

    actions.print();

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

}
//-----------------------------------------------------------------------------
TEST(ascent_mpi_runtime, test_python_script_extract_file)
{
    // same as above, however we read the script from a file

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
    create_3d_example_dataset(data,32,par_rank,par_size);
    data["state/cycle"] = 101;

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


    //
    // Create the actions.
    //

    string script_fname = conduit::utils::join_file_path(output_path,
                                                         "tout_mpi_py_script_test.py");

    if(par_rank == 0)
    {
        // write the script to a file
        ofstream script_file;
        script_file.open(script_fname.c_str());
        script_file << py_script;
        script_file.close();
    }

    conduit::Node extracts;
    extracts["e1/type"]  = "python";
    extracts["e1/params/file"] = script_fname;

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    conduit::Node &execute  = actions.append();
    execute["action"] = "execute";

    actions.print();

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

}
//
// This demos using the ascent python api inside of ascent ...
std::string py_script_inception = "\n"
"import conduit\n"
"import ascent.mpi\n"
"# we treat everything as a multi_domain in ascent so grab child 0\n"
"n_mesh = ascent_data().child(0)\n"
"ascent_opts = conduit.Node()\n"
"ascent_opts['mpi_comm'].set(ascent_mpi_comm_id())\n"
"a = ascent.mpi.Ascent()\n"
"a.open(ascent_opts)\n"
"a.publish(n_mesh)\n"
"actions = conduit.Node()\n"
"scenes  = conduit.Node()\n"
"scenes['s1/plots/p1/type'] = 'pseudocolor'\n"
"scenes['s1/plots/p1/field'] = 'radial_vert'\n"
"scenes['s1/image_prefix'] = 'tout_python_mpi_extract_inception' \n"
"add_act =actions.append()\n"
"add_act['action'] = 'add_scenes'\n"
"add_act['scenes'] = scenes\n"
"actions.append()['action'] = 'execute'\n"
"a.execute(actions)\n"
"a.close()\n"
"\n";


//-----------------------------------------------------------------------------
TEST(ascent_mpi_runtime, test_python_extract_inception)
{
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
    create_3d_example_dataset(data,32,par_rank,par_size);
    data["state/cycle"] = 101;

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    //
    // Create the actions.
    //

    conduit::Node extracts;
    extracts["e1/type"]  = "python";
    extracts["e1/params/source"] = py_script_inception;

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    conduit::Node &execute  = actions.append();
    execute["action"] = "execute";

    actions.print();

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

