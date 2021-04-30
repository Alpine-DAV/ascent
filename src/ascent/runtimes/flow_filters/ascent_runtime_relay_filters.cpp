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
#include <ascent_data_object.hpp>
#include <ascent_logging.hpp>
#include <ascent_file_system.hpp>
#include <ascent_mpi_utils.hpp>
#include <ascent_runtime_utils.hpp>
#include <ascent_runtime_param_check.hpp>

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
#include <set>

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

//-------------------------------------------------------------------------
void
mesh_bp_generate_index(const conduit::Node &mesh,
                       const std::string &ref_path,
                       int num_domains,
                       Node &index_out)
{
    if(blueprint::mesh::is_multi_domain(mesh))
    {
        NodeConstIterator itr = mesh.children();

        while(itr.has_next())
        {
            Node curr_idx;
            const Node &cld = itr.next();
            blueprint::mesh::generate_index(cld,
                                            ref_path,
                                            num_domains,
                                            curr_idx);
            // add any new entries to the running index
            index_out.update(curr_idx);
        }
    }
    else
    {
        blueprint::mesh::generate_index(mesh,
                                        ref_path,
                                        num_domains,
                                        index_out);
    }
}

#ifdef ASCENT_MPI_ENABLED
//-------------------------------------------------------------------------
void
mesh_bp_generate_index(const conduit::Node &mesh,
                       const std::string &ref_path,
                       Node &index_out,
                       MPI_Comm comm)
{
    int par_rank = relay::mpi::rank(comm);
    int par_size = relay::mpi::size(comm);

    // we need a list of all possible topos, coordsets, etc
    // for the blueprint index in the root file.
    //
    // across ranks, domains may be sparse
    //  for example: a topo may only exist in one domain
    // so we union all local mesh indices, and then
    // se an all gather and union the results together
    // to create an accurate global index.

    index_t local_num_domains = blueprint::mesh::number_of_domains(mesh);
    // note:
    // find global # of domains w/o conduit_blueprint_mpi for now
    // since we aren't yet linking conduit_blueprint_mpi
    Node n_src, n_reduce;
    n_src = local_num_domains;

    mpi::sum_all_reduce(n_src,
                        n_reduce,
                        comm);

    index_t global_num_domains = n_reduce.to_int();

    index_out.reset();

    Node local_idx, gather_idx;

    if(local_num_domains > 0)
    {
        mesh_bp_generate_index(mesh,
                               ref_path,
                               global_num_domains,
                               local_idx);
    }

    relay::mpi::all_gather_using_schema(local_idx,
                                        gather_idx,
                                        comm);

    NodeConstIterator itr = gather_idx.children();
    while(itr.has_next())
    {
        const Node &curr = itr.next();
        index_out.update(curr);
    }
}

#endif


//
// recalculate domain ids so that we are consistant.
// Assumes that domains are valid
//
void make_domain_ids(conduit::Node &domains)
{
  int num_domains = domains.number_of_children();

  int domain_offset = 0;

#ifdef ASCENT_MPI_ENABLED
  int comm_id = flow::Workspace::default_mpi_comm();
  int comm_size = 1;
  int rank = 0;

  MPI_Comm mpi_comm = MPI_Comm_f2c(comm_id);
  MPI_Comm_rank(mpi_comm,&rank);

  MPI_Comm_size(mpi_comm, &comm_size);
  int *domains_per_rank = new int[comm_size];

  MPI_Allgather(&num_domains, 1, MPI_INT, domains_per_rank, 1, MPI_INT, mpi_comm);

  for(int i = 0; i < rank; ++i)
  {
    domain_offset += domains_per_rank[i];
  }
  delete[] domains_per_rank;
#endif

  for(int i = 0; i < num_domains; ++i)
  {
    conduit::Node &dom = domains.child(i);
    dom["state/domain_id"] = domain_offset + i;
  }
}
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
  output.reset();

  conduit::Node info;
  if(blueprint::mesh::verify(data, info))
  {
    const auto domains = blueprint::mesh::domains(data);
    for(auto it = domains.cbegin(); it != domains.cend(); ++it)
    {
      const conduit::Node &src_dom = **it;
      conduit::Node &dest_dom = output.append();
      dest_dom.set_external(src_dom);
    }
  }

  make_domain_ids(output);
  return output.number_of_children() > 0;
}
// mfem needs these special fields so look for them
void check_for_attributes(const conduit::Node &input,
                          std::vector<std::string> &list)
{
  const int num_doms = input.number_of_children();
  std::set<std::string> specials;
  for(int d = 0; d < num_doms; ++d)
  {
    const conduit::Node &dom = input.child(d);
    if(dom.has_path("fields"))
    {
      const conduit::Node &fields = dom["fields"];
      std::vector<std::string> fnames = fields.child_names();
      for(int i = 0; i < fnames.size(); ++i)
      {
        if(fnames[i].find("_attribute") != std::string::npos)
        {
          specials.insert(fnames[i]);
        }
      }
    }
  }

  for(auto it = specials.begin(); it != specials.end(); ++it)
  {
    list.push_back(*it);
  }
}

void filter_fields(const conduit::Node &input,
                   conduit::Node &output,
                   std::vector<std::string> fields,
                   flow::Graph &graph)
{
  // assume this is multi-domain
  //
  check_for_attributes(input, fields);

  const int num_doms = input.number_of_children();
  for(int d = 0; d < num_doms; ++d)
  {
    const conduit::Node &dom = input.child(d);
    conduit::Node &out_dom = output.append();
    std::set<std::string> topos;
    std::set<std::string> matsets;

    for(int f = 0; f < fields.size(); ++f)
    {
      const std::string fname = fields[f];
      if(dom.has_path("fields/" + fname))
      {
        const std::string fpath = "fields/" + fname;
        out_dom[fpath].set_external(dom[fpath]);
        // check for topologies
        const std::string topo = dom[fpath + "/topology"].as_string();
        const std::string tpath = "topologies/" + topo;
        topos.insert(topo);

        // check for matset
        if(dom.has_path(fpath + "/matset"))
        {
          const std::string mopo = dom[fpath + "/matset"].as_string();
          matsets.insert(mopo);
        }

        if(!out_dom.has_path(tpath))
        {
          out_dom[tpath].set_external(dom[tpath]);
          if(dom.has_path(tpath + "/grid_function"))
          {
            const std::string gf_name = dom[tpath + "/grid_function"].as_string();
            const std::string gf_path = "fields/" + gf_name;
            out_dom[gf_path].set_external(dom[gf_path]);
          }
          if(dom.has_path(tpath + "/boundary_topology"))
          {
            const std::string bname = dom[tpath + "/boundary_topology"].as_string();
            const std::string bpath = "topologies/" + bname;
            out_dom[bpath].set_external(dom[bpath]);
          }
        }
        // check for coord sets
        const std::string coords = dom[tpath + "/coordset"].as_string();
        const std::string cpath = "coordsets/" + coords;
        if(!out_dom.has_path(cpath))
        {
          out_dom[cpath].set_external(dom[cpath]);
        }
      }

    }

    // add nestsets associated with referenced topologies
    if(dom.has_path("nestsets"))
    {
      const int num_nestsets = dom["nestsets"].number_of_children();
      const std::vector<std::string> nest_names = dom["nestsets"].child_names();
      for(int i = 0; i < num_nestsets; ++i)
      {
        const conduit::Node &nestset = dom["nestsets"].child(i);
        const std::string nest_topo = nestset["topology"].as_string();
        if(topos.find(nest_topo) != topos.end())
        {
          out_dom["nestsets/"+nest_names[i]].set_external(nestset);
        }
      }
    }

    // add nestsets associated with referenced topologies
    if(dom.has_path("matsets"))
    {
      const int num_matts = dom["matsets"].number_of_children();
      const std::vector<std::string> matt_names = dom["matsets"].child_names();
      for(int i = 0; i < num_matts; ++i)
      {
        const conduit::Node &matt = dom["matsets"].child(i);
        if(matsets.find(matt_names[i]) != matsets.end())
        {
          out_dom["matsets/"+matt_names[i]].set_external(matt);
        }
      }
    }

    // auto save out ghost fields from subset of topologies
    Node * meta = graph.workspace().registry().fetch<Node>("metadata");
    if(meta->has_path("ghost_field"))
    {
      const conduit::Node ghost_list = (*meta)["ghost_field"];
      const int num_ghosts = ghost_list.number_of_children();

      for(int i = 0; i < num_ghosts; ++i)
      {
        std::string ghost = ghost_list.child(i).as_string();

        if(dom.has_path("fields/"+ghost) &&
           !out_dom.has_path("fields/"+ghost))
        {
          const conduit::Node &ghost_field = dom["fields/"+ghost];
          const std::string ghost_topo = ghost_field["topology"].as_string();
          if(topos.find(ghost_topo) != topos.end())
          {
            out_dom["fields/"+ghost].set_external(ghost_field);
          }
        }
      }
    }

    if(dom.has_path("state"))
    {
      out_dom["state"].set_external(dom["state"]);
    }
  }

  const int num_out_doms = output.number_of_children();
  bool has_data = false;
  // check to see if this resulted in any data
  for(int d = 0; d < num_out_doms; ++d)
  {
    const conduit::Node &dom = output.child(d);
    if(dom.has_path("fields"))
    {
      int fsize = dom["fields"].number_of_children();
      if(fsize != 0)
      {
        has_data = true;
        break;
      }
    }
  }

  has_data = global_someone_agrees(has_data);
  if(!has_data)
  {
    ASCENT_ERROR("Relay: field selection resulted in no data."
                 "This can occur if the fields did not exist "
                 "in the simulaiton data or if the fields were "
                 "created as a result of a pipeline, but the "
                 "relay extract did not recieve the result of "
                 "a pipeline");
  }

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

    if( params.has_child("num_files") )
    {
        if(!params["num_files"].dtype().is_integer())
        {
            info["errors"].append() = "optional entry 'num_files' must be an integer";
            res = false;
        }
        else
        {
            info["info"].append() = "includes 'num_files'";
        }
    }

    std::vector<std::string> valid_paths;
    std::vector<std::string> ignore_paths;
    valid_paths.push_back("path");
    valid_paths.push_back("protocol");
    valid_paths.push_back("fields");
    valid_paths.push_back("num_files");
    ignore_paths.push_back("fields");

    std::string surprises = surprise_check(valid_paths, ignore_paths, params);

    if(surprises != "")
    {
      res = false;
      info["errors"].append() = surprises;
    }

    return res;
}

//-----------------------------------------------------------------------------
void gen_domain_to_file_map(int num_domains,
                            int num_files,
                            Node &out)
{
    int num_domains_per_file = num_domains / num_files;
    int left_overs = num_domains % num_files;

    out["global_domains_per_file"].set(DataType::int32(num_files));
    out["global_domain_offsets"].set(DataType::int32(num_files));
    out["global_domain_to_file"].set(DataType::int32(num_domains));

    int32_array v_domains_per_file = out["global_domains_per_file"].value();
    int32_array v_domains_offsets  = out["global_domain_offsets"].value();
    int32_array v_domain_to_file   = out["global_domain_to_file"].value();

    // setup domains per file
    for(int f=0; f < num_files; f++)
    {
        v_domains_per_file[f] = num_domains_per_file;
        if( f < left_overs)
            v_domains_per_file[f]+=1;
    }

    // prefix sum to calc offsets
    for(int f=0; f < num_files; f++)
    {
        v_domains_offsets[f] = v_domains_per_file[f];
        if(f > 0)
            v_domains_offsets[f] += v_domains_offsets[f-1];
    }

    // do assignment, create simple map
    int f_idx = 0;
    for(int d=0; d < num_domains; d++)
    {
        if(d >= v_domains_offsets[f_idx])
            f_idx++;
        v_domain_to_file[d] = f_idx;
    }
}

//-----------------------------------------------------------------------------
void mesh_blueprint_save(const Node &data,
                         const std::string &path,
                         const std::string &file_protocol,
                         int num_files,
                         std::string &root_file_out)
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
      return;
    }

    int local_num_domains = multi_dom.number_of_children();
    // figure out what cycle we are
    if(local_num_domains > 0 && is_valid)
    {
      Node dom = multi_dom.child(0);
      if(!dom.has_path("state/cycle"))
      {
        static std::map<string,int> counters;
        ASCENT_INFO("Blueprint save: no 'state/cycle' present."
                    " Defaulting to counter");
        cycle = counters[path];
        counters[path]++;
      }
      else
      {
        cycle = dom["state/cycle"].to_int();
      }
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
    char fmt_buff[64] = {0};
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

    int global_num_domains = local_num_domains;

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

    n_src = local_num_domains;

    mpi::sum_all_reduce(n_src,
                        n_reduce,
                        mpi_comm);

    global_num_domains = n_reduce.as_int();
#endif


    if(global_num_domains == 0)
    {
      if(par_rank == 0)
      {
          ASCENT_WARN("There no data to save. Doing nothing.");
      }
      return;
    }


    // zero or negative (default cases), use one file per domain
    if(num_files <= 0)
    {
        num_files = global_num_domains;
    }

    // if global domains > num_files, warn and use one file per domain
    if(global_num_domains < num_files)
    {
        ASCENT_INFO("Requested more files than actual domains, "
                    "writing one file per domain");
        num_files = global_num_domains;
    }

    if(!dir_ok)
    {
        ASCENT_ERROR("Error: failed to create directory " << output_dir);
    }

    if(global_num_domains == num_files)
    {
        // write out each domain
        for(int i = 0; i < local_num_domains; ++i)
        {
            const Node &dom = multi_dom.child(i);
            uint64 domain = dom["state/domain_id"].to_uint64();

            snprintf(fmt_buff, sizeof(fmt_buff), "%06llu",domain);
            oss.str("");
            oss << "domain_" << fmt_buff << "." << file_protocol;
            string output_file  = conduit::utils::join_file_path(output_dir,oss.str());
            relay::io::save(dom, output_file);
        }
    }
    else // more complex case
    {
        //
        // recall: we have re-labeled domain ids from 0 - > N-1, however
        // some mpi tasks may have no data.
        //

        // books we keep:

        Node books;
        books["local_domain_to_file"].set(DataType::int32(local_num_domains));
        books["local_domain_status"].set(DataType::int32(local_num_domains));
        books["local_file_batons"].set(DataType::int32(num_files));
        books["global_file_batons"].set(DataType::int32(num_files));

        // local # of domains
        int32_array local_domain_to_file = books["local_domain_to_file"].value();
        int32_array local_domain_status  = books["local_domain_status"].value();
        // num total files
        int32_array local_file_batons    = books["local_file_batons"].value();
        int32_array global_file_batons   = books["global_file_batons"].value();

        Node d2f_map;
        gen_domain_to_file_map(global_num_domains,
                               num_files,
                               books);

        int32_array global_d2f = books["global_domain_to_file"].value();

        // init our local map and status array
        for(int d = 0; d < local_num_domains; ++d)
        {
            const Node &dom = multi_dom.child(d);
            uint64 domain = dom["state/domain_id"].to_uint64();
            // local domain index to file map
            local_domain_to_file[d] = global_d2f[domain];
            local_domain_status[d] = 1; // pending (1), vs done (0)
        }

        //
        // Round and round we go, will we deadlock I believe no :-)
        //
        // Here is how this works:
        //  At each round, if a rank has domains pending to write to a file,
        //  we put the rank id in the local file_batons vec.
        //  This vec is then mpi max'ed, and the highest rank
        //  that needs access to each file will write this round.
        //
        //  When a rank does not need to write to a file, we
        //  put -1 for this rank.
        //
        //  During each round, max of # files writers are participating
        //
        //  We are done when the mpi max of the batons is -1 for all files.
        //

        bool another_twirl = true;
        int twirls = 0;
        while(another_twirl)
        {
            // update baton requests
            for(int f = 0; f < num_files; ++f)
            {
                for(int d = 0; d < local_num_domains; ++d)
                {
                    if(local_domain_status[d] == 1)
                        local_file_batons[f] = par_rank;
                    else
                        local_file_batons[f] = -1;
                }
            }

            // mpi max file batons array
            #ifdef ASCENT_MPI_ENABLED
                mpi::max_all_reduce(books["local_file_batons"],
                                    books["global_file_batons"],
                                    mpi_comm);
            #else
                global_file_batons.set(local_file_batons);
            #endif

            // we now have valid batons (global_file_batons)
            for(int f = 0; f < num_files; ++f)
            {
                // check if this rank has the global baton for this file
                if( global_file_batons[f] == par_rank )
                {
                    // check the domains this rank has pending
                    for(int d = 0; d < local_num_domains; ++d)
                    {
                        // reuse this handle for all domains in the file
                        relay::io::IOHandle hnd;
                        if(local_domain_status[d] == 1 &&  // pending
                           local_domain_to_file[d] == f) // destined for this file
                        {
                            // now is the time to write!
                            // pattern is:
                            //  file_%06llu.{protocol}:/domain_%06llu/...
                            const Node &dom = multi_dom.child(d);
                            uint64 domain_id = dom["state/domain_id"].to_uint64();
                            // construct file name
                            snprintf(fmt_buff, sizeof(fmt_buff), "%06d",f);
                            oss.str("");
                            oss << "file_" << fmt_buff << "." << file_protocol;
                            std::string file_name = oss.str();
                            oss.str("");
                            // and now domain id
                            snprintf(fmt_buff, sizeof(fmt_buff), "%06llu",domain_id);
                            oss << "domain_" << fmt_buff;

                            std::string path = oss.str();
                            string output_file = conduit::utils::join_file_path(output_dir,file_name);

                            if(!hnd.is_open())
                            {
                                hnd.open(output_file);
                            }

                            hnd.write(dom,path);
                            ASCENT_INFO("rank " << par_rank << " output_file"
                                      << output_file << " path " << path);

                            // update status, we are done with this doman
                            local_domain_status[d] = 0;
                        }
                    }
                }
            }

            // If you  need to debug the baton alog:
            // std::cout << "[" << par_rank << "] "
            //              << " twirls: " << twirls
            //              << " details\n"
            //              << books.to_yaml();

            // check if we have another round
            // stop when all batons are -1
            another_twirl = false;
            for(int f = 0; f < num_files && !another_twirl; ++f)
            {
                // if any entry is not -1, we still have more work to do
                if(global_file_batons[f] != -1)
                {
                    another_twirl = true;
                    twirls++;
                }
            }
        }
    }

    int root_file_writer = 0;
    if(local_num_domains == 0)
    {
      root_file_writer = -1;
    }
#ifdef ASCENT_MPI_ENABLED
    // Rank 0 could have an empty domain, so we have to check
    // to find someone with a data set to write out the root file.
    Node out;
    out = local_num_domains;
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

    snprintf(fmt_buff, sizeof(fmt_buff), "%06d",cycle);

    oss.str("");
    oss << path
        << ".cycle_"
        << fmt_buff
        << ".root";

    string root_file = oss.str();

    // return this via out var
    root_file_out = root_file;

    // --------
    // create blueprint index
    // --------

    // all ranks participate in the index gen
    Node bp_idx;
#ifdef ASCENT_MPI_ENABLED
        // mpi tasks may have diff fields, topos, etc
        //
        detail::mesh_bp_generate_index(multi_dom,
                                       "",
                                       bp_idx["mesh"],
                                       mpi_comm);
#else
        detail::mesh_bp_generate_index(multi_dom,
                                       "",
                                       global_num_domains,
                                       bp_idx["mesh"]);
#endif

    // let selected rank write out the root file
    if(par_rank == root_file_writer)
    {
        string output_dir_base, output_dir_path;

        // TODO: Fix for windows
        conduit::utils::rsplit_string(output_dir,
                                      "/",
                                      output_dir_base,
                                      output_dir_path);

        string output_tree_pattern;
        string output_file_pattern;

        if(global_num_domains == num_files)
        {
            output_tree_pattern = "/";
            output_file_pattern = conduit::utils::join_file_path(output_dir_base,
                                                                 "domain_%06d." + file_protocol);
        }
        else
        {
            output_tree_pattern = "/domain_%06d";
            output_file_pattern = conduit::utils::join_file_path(output_dir_base,
                                                                 "file_%06d." + file_protocol);
        }


        Node root;
        root["blueprint_index"] = bp_idx;

        // work around conduit and manually add state fields
        if(multi_dom.child(0).has_path("state/cycle"))
        {
          bp_idx["mesh/state/cycle"] = multi_dom.child(0)["state/cycle"].to_int32();
        }

        if(multi_dom.child(0).has_path("state/time"))
        {
          bp_idx["mesh/state/time"] = multi_dom.child(0)["state/time"].to_double();
        }

        root["protocol/name"]    = file_protocol;
        root["protocol/version"] = "0.6.0";

        root["number_of_files"]  = num_files;
        root["number_of_trees"]  = global_num_domains;
        // TODO: make sure this is relative
        root["file_pattern"]     = output_file_pattern;
        root["tree_pattern"]     = output_tree_pattern;

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
    path = output_dir(path, graph());

    if(params().has_child("protocol"))
    {
        protocol = params()["protocol"].as_string();
    }

    if(!input("in").check_type<DataObject>())
    {
        // error
        ASCENT_ERROR("relay_io_save requires a DataObject input");
    }

    DataObject *data_object  = input<DataObject>("in");
    if(!data_object->is_valid())
    {
      return;
    }
    std::shared_ptr<Node> n_input = data_object->as_node();

    Node *in = n_input.get();

    Node selected;
    conduit::Node test;
    if(params().has_path("fields"))
    {
      std::vector<std::string> field_selection;
      const conduit::Node &flist = params()["fields"];
      const int num_fields = flist.number_of_children();
      if(num_fields == 0)
      {
        ASCENT_ERROR("relay_io_save field selection list must be non-empty");
      }
      for(int i = 0; i < num_fields; ++i)
      {
        const conduit::Node &f = flist.child(i);
        if(!f.dtype().is_string())
        {
           ASCENT_ERROR("relay_io_save field selection list values must be a string");
        }
        field_selection.push_back(f.as_string());
      }
      detail::filter_fields(*in, selected, field_selection, graph());
    }
    else
    {
      // select all fields
      selected.set_external(*in);
    }

    int num_files = -1;

    if(params().has_path("num_files"))
    {
        num_files = params()["num_files"].to_int();
    }

    std::string result_path;
    if(protocol.empty())
    {
        conduit::relay::io::save(selected,path);
        result_path = path;
    }
    else if( protocol == "blueprint/mesh/hdf5")
    {
        mesh_blueprint_save(selected,
                            path,
                            "hdf5",
                            num_files,
                            result_path);
    }
    else if( protocol == "blueprint/mesh/json")
    {
        mesh_blueprint_save(selected,
                            path,
                            "hdf5",
                            num_files,
                            result_path);

    }
    else if( protocol == "blueprint/mesh/yaml")
    {
        mesh_blueprint_save(selected,
                            path,
                            "hdf5",
                            num_files,
                            result_path);

    }
    else
    {
        conduit::relay::io::save(selected,path,protocol);
        result_path = path;
    }

    // add this to the extract results in the registry
    if(!graph().workspace().registry().has_entry("extract_list"))
    {
      conduit::Node *extract_list = new conduit::Node();
      graph().workspace().registry().add<Node>("extract_list",
                                               extract_list,
                                               -1); // TODO keep forever?
    }

    conduit::Node *extract_list = graph().workspace().registry().fetch<Node>("extract_list");

    Node &einfo = extract_list->append();
    einfo["type"] = "relay";
    if(!protocol.empty())
        einfo["protocol"] = protocol;
    einfo["path"] = result_path;
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





