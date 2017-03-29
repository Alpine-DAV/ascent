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

#include <adios.h>
#include <adios_types.h>
#include "strawman_adios_pipeline.hpp"
#include <strawman_file_system.hpp>

// standard lib includes
#include <iostream>
#include <string.h>
#include <limits.h>
#include <cstdlib>
#include <conduit.hpp>
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
class AdiosPipeline::IOManager
{
public:

    // construction and destruction 
#ifdef PARALLEL
     IOManager(MPI_Comm mpi_comm);
#else
     IOManager();
#endif
    ~IOManager();

    // main call to create adios file set
    void SaveToAdiosFormat(const Node &data, const Node &options);
    void NodeTraverse(const Node &data, char* path);
//-----------------------------------------------------------------------------
// private vars for MPI case
//-----------------------------------------------------------------------------
private:
    int m_rank;
    int64_t       m_adios_group;
    int64_t       m_adios_file;
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
AdiosPipeline::IOManager::IOManager(MPI_Comm mpi_comm)
:m_rank(0),
 m_mpi_comm(mpi_comm)
{
    MPI_Comm_rank(m_mpi_comm, &m_rank);
    MPI_Comm_size(m_mpi_comm, &m_mpi_size);
}
#else
//-----------------------------------------------------------------------------
AdiosPipeline::IOManager::IOManager()
:m_rank(0)
{
    
}
#endif


//-----------------------------------------------------------------------------
AdiosPipeline::IOManager::~IOManager()
{
    // empty
}

void AdiosPipeline::IOManager::NodeTraverse(const Node &data, char* path)
{
/*
int64_t var_id;
 NodeConstIterator itr = data.children();
 itr.to_front();
 if (itr.has_next())
 { bool looptag=true;
   while(looptag)
   {
     Node &n = itr.next();
     NodeTraverse(n,path+itr.name())
     if (!itr.has_next())
      looptag=false;
     
   }
 }
 else
 {   
   var_ids = adios_define_var (m_adios_group, path,"", adios_double,l_str, g_str, o_str);
   adios_write_byid(m_adios_file, var_id, data.value());

 }
*/

}
//-----------------------------------------------------------------------------
void
AdiosPipeline::IOManager::SaveToAdiosFormat(const Node &data,
                                                    const Node &options)
{
   #ifdef PARALLEL
    adios_init_noxml (m_mpi_comm);
    adios_set_max_buffer_size (10);

    int par_size;
    MPI_Comm_size(m_mpi_comm, &par_size);

    const Node &x_data = data[options["selected_vars"].as_string()];
    int child_count=x_data.number_of_children(); 
    std::cout<<"child_count "<<child_count<<"\n";

    adios_declare_group (&m_adios_group,"test_data", "iter", adios_stat_default);
    adios_select_method (m_adios_group, "MPI", "", "");
   
    
    char        filename [100];    
    strcpy (filename, options["output_path"].as_char8_str());
    adios_open (&m_adios_file, "test_data", filename, "w", m_mpi_comm);

    for (int i=0;i<child_count;i++)
    {        
	    char var_name[50]="",t[5]="";
            Node child_node=x_data.child(i);
	    DataType x_dt=child_node.dtype();
	    int num_ele=x_dt.number_of_elements();
	    int ele_size=x_dt.element_bytes();
	    //std::cout<<"element number "<<num_ele<<"\n";
	    //std::cout<<"element size "<<ele_size<<"\n";
	    //int64_t       var_ids[nblocks];
	    
	    sprintf(t, "%d", i);
            strcat(var_name, "var");
            strcat(var_name, t);
	    char         l_str[100], g_str[100],o_str[100];
	    sprintf (g_str, "%d", num_ele);
	    sprintf (l_str, "%d", num_ele/par_size);
	    
	    int offset=m_rank*(num_ele/par_size);
	    sprintf (o_str, "%d", offset);
	    std::cout<<g_str<<" vbn "<<l_str<<"  "<<o_str<<"  "<< m_rank<<"\n";
	    
	    int64_t var_id = adios_define_var (m_adios_group, var_name,"", adios_double, l_str,g_str,o_str);
	    adios_set_transform (var_id, "none");
	   
	   
	    adios_write_byid(m_adios_file, var_id, (void *)child_node.as_float64_ptr());
   
     }
     
     adios_close (m_adios_file);

     MPI_Barrier (m_mpi_comm);

     adios_finalize (m_rank);

    
    
       /*NodeTraverse(const Node &data, "");
	int NX = 10;
	double t[NX];
	// ADIOS variables declaration 
	int64_t handle;
	uint64_t group_size , total_size ;
	// data initialization 
   
	for (int i =0; i < NX ; i ++)
	t[i] = i * (m_rank+1) + 0.1; // ADIOS routines
	adios_init( "/home/dongliang/StrawMan/strawman-0.1.0/src/strawman/pipelines/config.xml", m_mpi_comm);
	adios_open (&handle , "temperature" , "data.bp" , "w" , m_mpi_comm);
	group_size = sizeof(int) + sizeof(double)*NX; // double array t
	//adios_group_size (handle, 4, &total_size );
	adios_write (handle, "NX" , &NX);*/
	//adios_write (handle, "temperature" , t );
	//adios_close (m_adios_file);
	//adios_finalize(m_rank);
   #endif

  
   

   
}



//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//
// Creation and Destruction
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
AdiosPipeline::AdiosPipeline()
:Pipeline(), 
 m_io(NULL)
{

}

//-----------------------------------------------------------------------------
AdiosPipeline::~AdiosPipeline()
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
AdiosPipeline::Initialize(const conduit::Node &options)
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
AdiosPipeline::Cleanup()
{
    if(m_io != NULL)
    {
        delete m_io;
    }

    m_io = NULL;
}

//-----------------------------------------------------------------------------
void
AdiosPipeline::Publish(const conduit::Node &data)
{
    m_data.set_external(data);
}

//-----------------------------------------------------------------------------
void
AdiosPipeline::Execute(const conduit::Node &actions)
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
            m_io->SaveToAdiosFormat(m_data,action);
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



