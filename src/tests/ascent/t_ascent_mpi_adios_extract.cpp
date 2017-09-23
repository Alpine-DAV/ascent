//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-716457
// 
// All rights reserved.
// 
// This file is part of Ascent. 
// 
// For details, see: http://software.llnl.gov/ascent/.
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
/// file: ascent_mpi_render_2d.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <ascent.hpp>

#include <iostream>
#include <math.h>
#include <mpi.h>

#include <conduit_blueprint.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"

using namespace std;
using namespace conduit;
using ascent::Ascent;

//-----------------------------------------------------------------------------
TEST(ascent_mpi_runtime, test_render_mpi_2d_main_runtime)
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
    create_3d_example_dataset(data,par_rank,par_size);
    
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));
    verify_info.print();
    
    //
    // Create the actions.
    //

    conduit::Node extracts;
    extracts["e1/type"]  = "adios";
    // populate some param examples
    extracts["e1/params/important_param"] = "dave";
    extracts["e1/params/int"] = 1;
    extracts["e1/params/float"] = 1.f;
    extracts["e1/params/double"] = 1.;
    
    std::vector<float> values;
    values.push_back(1.f);
    values.push_back(2.f);
    values.push_back(3.f);
    //zero copy == set_external
    extracts["e1/params/float_values"].set_external(values);
    
    const int num_vals = 3;
    double d_values[num_vals] = {1., 2., 3.};
    extracts["e1/params/double_values"].set_external(d_values, num_vals);
  
    
    extracts["e1/params/actions"] = "actions";
    //
    // we can tell adios to do actions with the published data
    // if we use the same api as ascent all we have to do
    // is translate it in the adios filter
    //
    conduit::Node &contour = extracts["e1/params/actions"].append();
    contour["type"]  = "contour";
    contour["params/field"] = "braid";
    contour["params/iso_values"] = 0.3;
 
    conduit::Node actions;
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
   
    MPI_Barrier(comm);
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


