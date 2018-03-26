//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
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
/// file: ascent_runtime_relay_filters.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_relay_filters.hpp"

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

// conduit includes
#include <conduit.hpp>
#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_logging.hpp>
#include <ascent_file_system.hpp>

#include <flow_graph.hpp>
#include <flow_workspace.hpp>

// mpi related includes
#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
// -- conduit relay mpi
#include <conduit_relay_mpi.hpp>
#endif

// std includes
#include <limits>

using namespace std;
using namespace conduit;
using namespace conduit::relay;
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
// -- begin ascent::runtime::detail --
//-----------------------------------------------------------------------------
namespace detail
{
// 
// This expects a single or multi_domain blueprint mesh and will iterate
// through all domains to see if they are valid. Returns true
// if it contains valid data and false if there is no valid
// data.
//
// This is needed because after pipelines, it is possible to
// have no data left in a domain because of something like a 
// clip
//
bool clean_mesh(const conduit::Node &data, conduit::Node &output)
{
  const int potential_doms = data.number_of_children();
  bool maybe_multi_dom = true;

  if(!data.dtype().is_object() && !data.dtype().is_list())
  {
    maybe_multi_dom = false;
  }

  if(maybe_multi_dom)
  {
    // check all the children for valid domains
    for(int i = 0; i < potential_doms; ++i)
    {
      conduit::Node info;
      const conduit::Node &child = data.child(i);
      bool is_valid = blueprint::mesh::verify(child, info); 
      if(is_valid)
      {
        conduit::Node &dest_dom = output.append();
        dest_dom.set_external(child);
      }
    }
  }
  // if there is nothing in the outut, lets see if it is a
  // valid single domain
  if(output.number_of_children() == 0)
  {
    // check to see if this is a single valid domain
    conduit::Node info;
    bool is_valid = blueprint::mesh::verify(data, info); 
    if(is_valid)
    {
      conduit::Node &dest_dom = output.append();
      dest_dom.set_external(data);
    }
  }

  return output.number_of_children() > 0;
}
//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::detail --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// helper shared by io save and load
//-----------------------------------------------------------------------------
bool
verify_io_params(const conduit::Node &params,
                 conduit::Node &info)
{
    bool res = true;
    
    if( !params.has_child("path") ) 
    {
        info["errors"].append() = "missing required entry 'path'";
        res = false;
    }
    else if(!params["path"].dtype().is_string())
    {
        info["errors"].append() = "'path' must be a string";
        res = false;
    }
    else if(params["path"].as_string().empty())
    {
        info["errors"].append() = "'path' is an empty string";
        res = false;
    }
    
    
    if( params.has_child("protocol") ) 
    {
        if(!params["protocol"].dtype().is_string())
        {
            info["errors"].append() = "optional entry 'protocol' must be a string";
            res = false;
        }
        else if(params["protocol"].as_string().empty())
        {
            info["errors"].append() = "'protocol' is an empty string";
            res = false;
        }
        else
        {
            info["info"].append() = "includes 'protocol'";
        }
    }

    return res;
}


//-----------------------------------------------------------------------------
void mesh_blueprint_save(const Node &data,
                         const std::string &path,
                         const std::string &file_protocol)
{
    // The assumption here is that everything is multi domain    

    Node multi_dom; 
    bool is_valid = detail::clean_mesh(data, multi_dom);
    
    int par_rank = 0;  
    int par_size = 1;
    // we may not have any domains so init to max
    int cycle = std::numeric_limits<int>::max();

    int local_boolean = is_valid ? 1 : 0; 
    int global_boolean = local_boolean;
#ifdef ASCENT_MPI_ENABLED
    MPI_Comm mpi_comm = MPI_Comm_f2c(Workspace::default_mpi_comm());
    MPI_Comm_rank(mpi_comm, &par_rank);
    MPI_Comm_size(mpi_comm, &par_size);

    // check to see if any valid data exist
    MPI_Allreduce((void *)(&local_boolean),
                  (void *)(&global_boolean),
                  1,
                  MPI_INT,
                  MPI_SUM,
                  mpi_comm);
#endif
    
    if(global_boolean == 0)
    {
      ASCENT_INFO("Blueprint save: no valid data exists. Skipping save");
      //return;
    }

    int num_domains = multi_dom.number_of_children();
    // figure out what cycle we are
    if(num_domains > 0 && is_valid)
    {
      Node dom = multi_dom.child(0);
      if(!dom.has_path("state/cycle"))
      {
        ASCENT_ERROR("Cannot blueprint save without 'state/cycle'");
      }
      cycle = dom["state/cycle"].to_int();
    }
    
#ifdef ASCENT_MPI_ENABLED
    Node n_cycle, n_min;
    
    n_cycle = (int)cycle;

    mpi::min_all_reduce(n_cycle,
                        n_min,
                        mpi_comm);

    cycle = n_min.as_int();
#endif
    // setup the directory
    char fmt_buff[64];
    snprintf(fmt_buff, sizeof(fmt_buff), "%06d",cycle);
    
    std::string output_base_path = path;
    
    ostringstream oss;
    oss << output_base_path << ".cycle_" << fmt_buff;
    string output_dir  =  oss.str();

    bool dir_ok = false;

    // let rank zero handle dir creation
    if(par_rank == 0)
    {
        // check of the dir exists
        dir_ok = directory_exists(output_dir);
        if(!dir_ok)
        {
            // if not try to let rank zero create it
            dir_ok = create_directory(output_dir);
        }
    }
    int global_domains = num_domains;
#ifdef ASCENT_MPI_ENABLED
    // TODO:
    // This a reduce to check for an error ... 
    // it will be a common pattern, how do we make this easy?
    
    // use an mpi sum to check if the dir exists
    Node n_src, n_reduce;
    
    if(dir_ok)
        n_src = (int)1;
    else
        n_src = (int)0;

    mpi::sum_all_reduce(n_src,
                        n_reduce,
                        mpi_comm);

    dir_ok = (n_reduce.as_int() == 1);

    n_src = num_domains;

    mpi::sum_all_reduce(n_src,
                        n_reduce,
                        mpi_comm);

    global_domains = n_reduce.as_int();
#endif
  
    if(global_domains == 0)
    {
      if(par_rank == 0)
      {
          ASCENT_WARN("There no data to save. Doing nothing");
      }
      return;
    }

    if(!dir_ok)
    {
        ASCENT_ERROR("Error: failed to create directory " << output_dir);
    }
    // write out each domain
    for(int i = 0; i < num_domains; ++i)
    {
        Node dom = multi_dom.child(i); 
        uint64 domain = dom["state/domain_id"].to_uint64();

        snprintf(fmt_buff, sizeof(fmt_buff), "%06lu",domain);
        oss.str("");
        oss << "domain_" << fmt_buff << "." << file_protocol;
        string output_file  = conduit::utils::join_file_path(output_dir,oss.str());
        relay::io::save(dom, output_file);
    }
    
    int root_file_writer = 0;
    if(num_domains == 0)
    {
      root_file_writer = -1; 
    }
#ifdef ASCENT_MPI_ENABLED
    // Rank 0 could have an empty domain, so we have to check
    // to find someone with a data set to write out the root file.
    Node out;
    out = num_domains;
    Node rcv;
     
    mpi::all_gather_using_schema(out, rcv, mpi_comm);
    root_file_writer = -1;
    int* res_ptr = (int*)rcv.data_ptr();
    for(int i = 0; i < par_size; ++i)
    {
        if(res_ptr[i] != 0)
        {
            root_file_writer = i;
            break;
        }
    }

    MPI_Barrier(mpi_comm);
#endif

    if(root_file_writer == -1)
    {
        // this should not happen. global doms is already 0
        ASCENT_WARN("Relay: there are no domains to write out");
    }
    // let rank zero write out the root file
    if(par_rank == root_file_writer)
    {
        snprintf(fmt_buff, sizeof(fmt_buff), "%06d",cycle);

        oss.str("");
        oss << path
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
                                                                    "domain_%06d." + file_protocol);


        Node root;
        Node &bp_idx = root["blueprint_index"];

        blueprint::mesh::generate_index(multi_dom.child(0),
                                        "",
                                        global_domains,
                                        bp_idx["mesh"]);
            
        root["protocol/name"]    = "conduit_" + file_protocol;
        root["protocol/version"] = "0.2.1";

        root["number_of_files"]  = global_domains;
        // for now we will save one file per domain, so trees == files
        root["number_of_trees"]  = global_domains;
        // TODO: make sure this is relative 
        root["file_pattern"]     = output_file_pattern;
        root["tree_pattern"]     = "/";
        relay::io::save(root,root_file,file_protocol);
    }
}



//-----------------------------------------------------------------------------
RelayIOSave::RelayIOSave()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
RelayIOSave::~RelayIOSave()
{
// empty
}

//-----------------------------------------------------------------------------
void 
RelayIOSave::declare_interface(Node &i)
{
    i["type_name"]   = "relay_io_save";
    i["port_names"].append() = "in";
    i["output_port"] = "false";
}

//-----------------------------------------------------------------------------
bool
RelayIOSave::verify_params(const conduit::Node &params,
                           conduit::Node &info)
{
    return verify_io_params(params,info);
}


//-----------------------------------------------------------------------------
void 
RelayIOSave::execute()
{
    std::string path, protocol;
    path = params()["path"].as_string();
    // TODO check if we need to expand the path (MPI) case for std protocols
    if(params().has_child("protocol"))
    {
        protocol = params()["protocol"].as_string();
    }

    if(!input("in").check_type<Node>())
    {
        // error
        ASCENT_ERROR("relay_io_save requires a conduit::Node input");
    }
    
    Node *in = input<Node>("in");

    if(protocol.empty())
    {
        conduit::relay::io::save(*in,path);
    }
    else if( protocol == "blueprint/mesh/hdf5")
    {
        mesh_blueprint_save(*in,path,"hdf5");
    }
    else
    {
        conduit::relay::io::save(*in,path,protocol);
    }

}



//-----------------------------------------------------------------------------
RelayIOLoad::RelayIOLoad()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
RelayIOLoad::~RelayIOLoad()
{
// empty
}

//-----------------------------------------------------------------------------
void 
RelayIOLoad::declare_interface(Node &i)
{
    i["type_name"]   = "relay_io_load";
    i["port_names"] = DataType::empty();
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
RelayIOLoad::verify_params(const conduit::Node &params,
                           conduit::Node &info)
{
    return verify_io_params(params,info);
}


//-----------------------------------------------------------------------------
void 
RelayIOLoad::execute()
{
    std::string path, protocol;
    path = params()["path"].as_string();
    
    // TODO check if we need to expand the path (MPI) case
    if(params().has_child("protocol"))
    {
        protocol = params()["protocol"].as_string();
    }

    Node *res = new Node();
    
    if(protocol.empty())
    {
        conduit::relay::io::load(path,*res);
    }
    else
    {
        conduit::relay::io::load(path,protocol,*res);
    }
    
    set_output<Node>(res);

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





