//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-716457
// 
// All rights reserved.
// 
// This file is part of Strawman. 
// 
// For details, see: http://software.llnl.gov/strawman/.
// 
// Please also read strawman/LICENSE
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
/// file: strawman_blueprint_hdf5_pipeline.cpp
///
//-----------------------------------------------------------------------------

#include "alpine_blueprint_hdf5_pipeline.hpp"
#include <alpine_file_system.hpp>

// standard lib includes
#include <iostream>
#include <string.h>
#include <limits.h>
#include <cstdlib>

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

// conduit includes
#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>

// mpi related includes
#ifdef PARALLEL
#include <mpi.h>
// -- conduit relay mpi
#include <conduit_relay_mpi.hpp>
#endif

#include <sstream>

using namespace conduit;
using namespace conduit::relay;

using namespace std;


//-----------------------------------------------------------------------------
// -- begin strawman:: --
//-----------------------------------------------------------------------------
namespace strawman
{

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// Internal Class that coordinates writing.
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
class BlueprintHDF5Pipeline::IOManager
{
public:

    // construction and destruction 
#ifdef PARALLEL
     IOManager(MPI_Comm mpi_comm);
#else
     IOManager();
#endif
    ~IOManager();

    // main call to create hdf5 file set
    void SaveToHDF5FileSet(const Node &data, const Node &options);

//-----------------------------------------------------------------------------
// private vars for MPI case
//-----------------------------------------------------------------------------
private:
    int m_rank;

//-----------------------------------------------------------------------------
// private vars for MPI case
//-----------------------------------------------------------------------------
#ifdef PARALLEL
    MPI_Comm            m_mpi_comm;
    int                 m_mpi_size;
#endif 

};

#ifdef PARALLEL
//-----------------------------------------------------------------------------
BlueprintHDF5Pipeline::IOManager::IOManager(MPI_Comm mpi_comm)
:m_rank(0),
 m_mpi_comm(mpi_comm)
{
    MPI_Comm_rank(m_mpi_comm, &m_rank);
    MPI_Comm_size(m_mpi_comm, &m_mpi_size);
}
#else
//-----------------------------------------------------------------------------
BlueprintHDF5Pipeline::IOManager::IOManager()
:m_rank(0)
{
    
}
#endif


//-----------------------------------------------------------------------------
BlueprintHDF5Pipeline::IOManager::~IOManager()
{
    // empty
}

//-----------------------------------------------------------------------------
void
BlueprintHDF5Pipeline::IOManager::SaveToHDF5FileSet(const Node &data,
                                                    const Node &options)
{
    // get cycle and domain id from the mesh

    uint64 domain = data["state/domain_id"].to_value();
    uint64 cycle  = data["state/cycle"].to_value();

    STRAWMAN_INFO("rank: "   << m_rank << 
                  " cycle: " << cycle << 
                  " domain:" << domain);

    char fmt_buff[64];
    snprintf(fmt_buff, sizeof(fmt_buff), "%06lu",cycle);
    
    std::string output_base_path = options["output_path"].as_string();
    
        
    ostringstream oss;
    oss << output_base_path << ".cycle_" << fmt_buff;
    string output_dir  =  oss.str();
    
    snprintf(fmt_buff, sizeof(fmt_buff), "%06lu",domain);
    oss.str("");
    oss << "domain_" << fmt_buff << ".hdf5";
    string output_file  = conduit::utils::join_file_path(output_dir,oss.str());


    bool dir_ok = false;

    // let rank zero handle dir creation
    if(m_rank == 0)
    {
        // check of the dir exists
        dir_ok = directory_exists(output_dir);
        if(!dir_ok)
        {
            // if not try to let rank zero create it
            dir_ok = create_directory(output_dir);
        }
    }
    
    int num_domains = 1;
#ifdef PARALLEL
    num_domains = m_mpi_size;
    
    // use an mpi sum to check if the dir exists
    Node n_src, n_reduce;
    
    if(dir_ok)
        n_src = (int)1;
    else
        n_src = (int)0;

    mpi::all_reduce(n_src,
                    n_reduce,
                    MPI_INT,
                    MPI_MAX,
                    m_mpi_comm);

    // error out if something went wrong.
    if(n_reduce.as_int() != 1)
    {
        STRAWMAN_ERROR("Error: failed to create directory " << output_dir);
    } 
#else
    if(!dir_ok)
    {
        STRAWMAN_ERROR("Error: failed to create directory " << output_dir);
    }
#endif

    relay::io::save(data,output_file);

    // let rank zero write out the root file
    if(m_rank == 0)
    {
        snprintf(fmt_buff, sizeof(fmt_buff), "%06lu",cycle);

        oss.str("");
        oss << options["output_path"].as_string() 
            << ".cycle_" 
            << fmt_buff 
            << ".root";

        string root_file = oss.str();

        string output_dir_base, output_dir_path;

        // TODO: Fix for windows
        conduit::utils::rsplit_string(output_dir,
                                      "/",
                                      output_dir_base,
                                      output_dir_path);

        string output_file_pattern = conduit::utils::join_file_path(output_dir_base,
                                                                    "domain_%06d.hdf5");


        Node root;
        Node &bp_idx = root["blueprint_index"];

        blueprint::mesh::generate_index(data,
                                        "",
                                        num_domains,
                                        bp_idx["mesh"]);
            
        root["protocol/name"]    = "conduit_hdf5";
        root["protocol/version"] = "0.2.1";

        root["number_of_files"]  = num_domains;
        root["number_of_trees"]  = num_domains;
        // TODO: make sure this is relative 
        root["file_pattern"]     = output_file_pattern;
        root["tree_pattern"]     = "/";

        CONDUIT_INFO("Creating: " << root_file);
        relay::io::save(root,root_file,"hdf5");

    }
}



//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//
// Creation and Destruction
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
BlueprintHDF5Pipeline::BlueprintHDF5Pipeline()
:Pipeline(), 
 m_io(NULL)
{

}

//-----------------------------------------------------------------------------
BlueprintHDF5Pipeline::~BlueprintHDF5Pipeline()
{
    Cleanup();
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//
// Main pipeline interface methods, which are used by the strawman interface.
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void
BlueprintHDF5Pipeline::Initialize(const conduit::Node &options)
{
#if PARALLEL
    if(!options.has_child("mpi_comm"))
    {
        STRAWMAN_ERROR("Missing Strawman::Open options missing MPI communicator (mpi_comm)");
    }

    int mpi_handle = options["mpi_comm"].value();
    MPI_Comm mpi_comm = MPI_Comm_f2c(mpi_handle);
    
    m_io = new IOManager(mpi_comm);
#else
    m_io = new IOManager();
#endif

}


//-----------------------------------------------------------------------------
void
BlueprintHDF5Pipeline::Cleanup()
{
    if(m_io != NULL)
    {
        delete m_io;
    }

    m_io = NULL;
}

//-----------------------------------------------------------------------------
void
BlueprintHDF5Pipeline::Publish(const conduit::Node &data)
{
    m_data.set_external(data);
}

//-----------------------------------------------------------------------------
void
BlueprintHDF5Pipeline::Execute(const conduit::Node &actions)
{
    //
    // Loop over the actions
    //
    for (int i = 0; i < actions.number_of_children(); ++i)
    {
        const Node &action = actions.child(i);
        STRAWMAN_INFO("Executing " << action["action"].as_string());
        
        if (action["action"].as_string() == "save")
        {
            m_io->SaveToHDF5FileSet(m_data,action);
        }
        else
        {
            STRAWMAN_INFO("Warning : unknown action "
                          << action["action"].as_string());
        }
    }
}






//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end strawman:: --
//-----------------------------------------------------------------------------



