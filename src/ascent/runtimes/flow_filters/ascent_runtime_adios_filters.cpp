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
/// file: ascent_runtime_adios_filters.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_adios_filters.hpp"

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

// conduit includes
#include <conduit.hpp>

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_logging.hpp>
#include <ascent_file_system.hpp>

#include <flow_graph.hpp>
#include <flow_workspace.hpp>

// mpi related includes
#ifdef PARALLEL
#include <mpi.h>
#endif

using namespace std;
using namespace conduit;
using namespace flow;

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime --
//-----------------------------------------------------------------------------
namespace runtime
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime::filters --
//-----------------------------------------------------------------------------
namespace filters
{

//-----------------------------------------------------------------------------
ADIOS::ADIOS()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
ADIOS::~ADIOS()
{
// empty
}

//-----------------------------------------------------------------------------
void 
ADIOS::declare_interface(Node &i)
{
    i["type_name"]   = "adios";
    i["port_names"].append() = "in";
    i["output_port"] = "false";
}

//-----------------------------------------------------------------------------
bool
ADIOS::verify_params(const conduit::Node &params,
                           conduit::Node &info)
{
    bool res = true;
    
    if( !params.has_child("important_param") ) 
    {
        info["errors"].append() = "missing required entry 'important_param'";
        res = false;
    }
    return res;
}
//-----------------------------------------------------------------------------
void 
ADIOS::execute()
{
    int par_rank = 0;  
    int par_size = 1;
     
#ifdef PARALLEL
    MPI_Comm mpi_comm = MPI_Comm_f2c(Workspace::default_mpi_comm());
    MPI_Comm_rank(mpi_comm, &par_rank);
    MPI_Comm_size(mpi_comm, &par_size);
#endif

    std::string important_param;
    important_param = params()["important_param"].as_string();
    if(par_rank == 0)
    {
      std::cout<<"The important param is "<<important_param<<"\n";
    }

    if(params().has_child("int"))
    {
        int the_int = params()["int"].as_int32();
    }

    if(params().has_child("float"))
    {
        float the_float = params()["float"].as_float32();
    }

    if(params().has_child("double"))
    {
        double the_double = params()["double"].as_float64();
    }

    if(params().has_child("float_values"))
    {
       float *vals = params()["float_values"].as_float32_ptr();
    }

    if(params().has_child("double_values"))
    {
       double *vals = params()["double_values"].as_float64_ptr();
    }

    if(params().has_child("actions"))
    {
      const conduit::Node actions = params()["actions"];
      if(par_rank == 0)
      {
        std::cout<<"Actions passed to adios filter:\n";
        actions.print();
      }
    }

    if(!input("in").check_type<Node>())
    {
        // error
        ASCENT_ERROR("adios requires a conduit::Node input");
    }

    
    Node *blueprint_data = input<Node>("in");

}

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::filters --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------





