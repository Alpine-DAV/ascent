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
/// file: ascent_probing_runtime.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_probing_runtime.hpp"

// standard lib includes
#include <string.h>

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

// conduit includes
#include <conduit_blueprint.hpp>

// mpi related includes
#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
// -- conduit relay mpi
#include <conduit_relay_mpi.hpp>
#endif

using namespace conduit;
using namespace std;


//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//
// Creation and Destruction
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
ProbingRuntime::ProbingRuntime()
:Runtime()
{

}

//-----------------------------------------------------------------------------
ProbingRuntime::~ProbingRuntime()
{
    Cleanup();
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//
// Main runtime interface methods called by the ascent interface.
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void
ProbingRuntime::Initialize(const conduit::Node &options)
{
#if ASCENT_MPI_ENABLED
    if(!options.has_child("mpi_comm") ||
       !options["mpi_comm"].dtype().is_integer())
    {
        ASCENT_ERROR("Missing Ascent::open options missing MPI communicator (mpi_comm)");
    }
#endif
    // check for probing options (?)

    m_runtime_options = options;
}

//-----------------------------------------------------------------------------
void
ProbingRuntime::Info(conduit::Node &out)
{
    out.reset();
    out["runtime/type"] = "probing";
}


//-----------------------------------------------------------------------------
void
ProbingRuntime::Cleanup()
{

}

//-----------------------------------------------------------------------------
void
ProbingRuntime::Publish(const conduit::Node &data)
{
    Node verify_info;
    bool verify_ok = conduit::blueprint::mesh::verify(data,verify_info);

#if ASCENT_MPI_ENABLED

    MPI_Comm mpi_comm = MPI_Comm_f2c(m_runtime_options["mpi_comm"].to_int());

    // parallel reduce to find if there were any verify errors across mpi tasks
    // use an mpi sum to check if all is ok
    Node n_src, n_reduce;

    if(verify_ok)
        n_src = (int)0;
    else
        n_src = (int)1;

    conduit::relay::mpi::sum_all_reduce(n_src,
                                        n_reduce,
                                        mpi_comm);

    int num_failures = n_reduce.value();
    if(num_failures != 0)
    {
        ASCENT_ERROR("Mesh Blueprint Verify failed on "
                       << num_failures
                       << " MPI Tasks");

        // you could use mpi to find out where things went wrong ...
    }



#else
    if(!verify_ok)
    {
         ASCENT_ERROR("Mesh Blueprint Verify failed!"
                        << std::endl
                        << verify_info.to_json());
    }
#endif

    // create our own tree, with all data zero copied.
    m_data.set_external(data);
}

//-----------------------------------------------------------------------------
void
ProbingRuntime::Execute(const conduit::Node &actions)
{
    std::cout << "===== execute probing runtime" << std::endl;
    
    // // Loop over the actions
    // for (int i = 0; i < actions.number_of_children(); ++i)
    // {
    //     const Node &action = actions.child(i);
    //     string action_name = action["action"].as_string();
    //     // implement action
    // }

    // copy actions for probing
    Node ascent_opt = m_runtime_options;
    int world_rank = 0;
    int size = 1;
    int rank_split = 0;

#if ASCENT_MPI_ENABLED
    // split comm into sim and vis nodes
    MPI_Comm comm_world  = MPI_Comm_f2c(m_runtime_options["mpi_comm"].to_int());
    MPI_Comm_rank(comm_world, &world_rank);
    MPI_Comm_size(comm_world, &size);

    // number of sim nodes: 3/4 * # nodes
    // TODO: change to dynamic ratio (based on parameter given in ascent_actions.yaml)
    rank_split = int(size*0.75 + 0.5); 
    int color = 0;
    if (world_rank >= rank_split)
      color = 1;

    MPI_Comm sim_comm;
    MPI_Comm_split(comm_world, color, 0, &sim_comm);
    ascent_opt["mpi_comm"] = sim_comm;
#endif  // ASCENT_MPI_ENABLED

    if (world_rank < rank_split)
    {
        ascent_opt["runtime/type"] = "ascent";      // set to main runtime
        
        // TODO: manipulate actions to only include probe renderings

        // all sim nodes run probing in a new ascent instance
        Ascent ascent_probing;
        ascent_probing.open(ascent_opt); 
        ascent_probing.publish(m_data);     // pass on data pointer
        ascent_probing.execute(actions);    // pass on actions

        // get execution time info
        // conduit::Node info;
        // ascent_probing.info(info);
        // info.print();

        ascent_probing.close();

        // -- the data should now contain the rendering times
        // std::cout << ".......probing runtime: " << std::endl;
        // m_data.print();

        // TODO: run the (former) split filter
    }

}






//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------



